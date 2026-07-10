# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — Warehouse Allocation Agent
# MAGIC
# MAGIC Reads from Silver, applies business rules from
# MAGIC `Rules_and_priority_of_warehouse.xlsx`, and produces the final allocation output.
# MAGIC
# MAGIC ---
# MAGIC ## Allocation logic (2 steps per order line)
# MAGIC
# MAGIC **Step 1 — Check PT12, the default storage location for every customer**
# MAGIC (per the `priority` sheet). If it has enough stock → **CASE 1**,
# MAGIC no change, auto-confirm. No authorization needed.
# MAGIC
# MAGIC **Step 2 — If the starting location is short**, walk the customer's priority
# MAGIC fallback list (from the `priority` sheet) and take the first location with
# MAGIC enough stock → **CASE 2**. Look up that Customer + Warehouse pair in the
# MAGIC `Authorization` sheet to decide if it needs UL approval before the PEGA ticket.
# MAGIC
# MAGIC **If nothing can fulfil it → CASE 3**, drop the line.
# MAGIC
# MAGIC ## Priority sequence (priority sheet)
# MAGIC | Customer | Default | Fallback 1 | Fallback 2 |
# MAGIC |---|---|---|---|
# MAGIC | Walmart | PT12(1000) | PT11 | PT15 |
# MAGIC | Cencosud | PT12(1000) | PT15 | PT11 |
# MAGIC | DMart / TentA / others | PT12(1000) | PT11 | PT15 |
# MAGIC
# MAGIC ## Authorization sheet (Customer + Warehouse → auth needed?)
# MAGIC | Customer | PT12 | PT11 | PT15 |
# MAGIC |---|---|---|---|
# MAGIC | Walmart | no | no | **yes** |
# MAGIC | Cencosud | no | **yes** | no |
# MAGIC | DMart | no | **yes** | **yes** |
# MAGIC | TentA | no | **yes** | **yes** |

# COMMAND ----------

CATALOG       = "warehouse_allocation"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, BooleanType
from datetime import datetime

PROCESS_TS = datetime.utcnow().isoformat()

# COMMAND ----------

# MAGIC %md ### Rules encoded from Rules_and_priority_of_warehouse.xlsx

# COMMAND ----------

# Priority fallback sequence per customer group (first entry = default, always PT12)
PRIORITY_RULES = {
    "Walmart":  ["PT12", "PT11", "PT15"],
    "Cencosud": ["PT12", "PT15", "PT11"],
    "DMart":    ["PT12", "PT11", "PT15"],
    "TentA":    ["PT12", "PT11", "PT15"],
    "Default":  ["PT12", "PT11", "PT15"],
}

# Authorization sheet — Customer + Warehouse -> requires auth (True/False)
AUTH_RULES = {
    "Walmart":  {"PT12": False, "PT11": False, "PT15": True},
    "Cencosud": {"PT12": False, "PT11": True,  "PT15": False},
    "DMart":    {"PT12": False, "PT11": True,  "PT15": True},
    "TentA":    {"PT12": False, "PT11": True,  "PT15": True},
    "Default":  {"PT12": False, "PT11": True,  "PT15": True},
}

# Carrusel sheet — sold-to parties that ALLOW incomplete pallets (yes)
ALLOW_INCOMPLETE_PALLET_NO = {
    "30279964", "30724028", "30597757", "30791641",
    "30280027", "30280062", "30280078", "30829327",
}

# Normalises customer name variants (e.g. "Demart" in rules file vs "DMart" in order data)
CUSTOMER_NAME_NORMALISE = {
    "WALMART":  "Walmart",
    "CENCOSUD": "Cencosud",
    "DEMART":   "DMart",
    "DMART":    "DMart",
    "TENTA":    "TentA",
}

print("Business rules loaded.")

# COMMAND ----------

# MAGIC %md ## 1 — Load Silver tables

# COMMAND ----------

df_orders = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.valid_orders")
df_stock  = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.stock_filtered")
df_exceptions_silver = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.exceptions")

print(f"Valid orders  : {df_orders.count()}")
print(f"Stock records : {df_stock.count()}")

# COMMAND ----------

# MAGIC %md ## 2 — Enrich orders: normalise customer group, pallet rule, starting location

# COMMAND ----------

@udf(StringType())
def normalise_customer_group(customer_name):
    if customer_name is None:
        return "Default"
    key = customer_name.strip().upper()
    return CUSTOMER_NAME_NORMALISE.get(key, "Default")

@udf(StringType())
def get_pallet_rule(sold_to_party):
    sold_to_str = str(sold_to_party) if sold_to_party is not None else ""
    return "FULL_PALLET_ONLY" if sold_to_str in ALLOW_INCOMPLETE_PALLET_NO else "INCOMPLETE_OK"

df_orders_enriched = (
    df_orders
    .withColumn("customer_group", normalise_customer_group(F.col("customer_name")))
    .withColumn("pallet_rule", get_pallet_rule(F.col("sold_to_party")))
    # Per the priority sheet, "Default Storage Location" is always PT12 (1000) for
    # every customer group — this is NOT the order's own SAP storage_location field.
    .withColumn("starting_location", F.lit("PT12"))
)

print("Customer group distribution:")
df_orders_enriched.groupBy("customer_group").count().show()

# COMMAND ----------

# MAGIC %md ## 3 — Join orders with stock

# COMMAND ----------

df_joined = (
    df_orders_enriched
    .join(
        df_stock.select(
            "material_number", "plant",
            "stock_PT12", "stock_PT11", "stock_PT15",
            "has_stock_PT12", "has_stock_PT11", "has_stock_PT15",
            "available_locations_count", "available_locations_list",
            "warehouse_type", "packaging_unit", "palletization_indicator",
            "total_stock"
        ),
        on="material_number",
        how="left"
    )
)

df_no_stock_at_all = df_joined.filter(F.col("total_stock").isNull())
df_has_stock = df_joined.filter(F.col("total_stock").isNotNull())

print(f"Orders with some stock  : {df_has_stock.count()}")
print(f"Orders with zero stock  : {df_no_stock_at_all.count()}")

# COMMAND ----------

# MAGIC %md ## 4 — Allocation Agent
# MAGIC
# MAGIC Step 1: check the order's own starting location.
# MAGIC Step 2: if short, walk the customer's priority fallback list.
# MAGIC Step 3: check Authorization sheet for the chosen Customer + Warehouse.

# COMMAND ----------

allocation_schema = StructType([
    StructField("allocated_location", StringType(), True),
    StructField("requires_auth",      StringType(), True),   # "true"/"false"
    StructField("case_number",        StringType(), True),
    StructField("case_reason",        StringType(), True),
    StructField("location_changed",   StringType(), True),   # "true"/"false"
])

@udf(allocation_schema)
def allocate(customer_group, starting_location, order_qty,
             stock_PT12, stock_PT11, stock_PT15,
             has_PT12, has_PT11, has_PT15,
             pallet_rule, packaging_unit):

    stock_map = {
        "PT12": (float(stock_PT12 or 0), bool(has_PT12)),
        "PT11": (float(stock_PT11 or 0), bool(has_PT11)),
        "PT15": (float(stock_PT15 or 0), bool(has_PT15)),
    }
    qty_needed = float(order_qty or 0)

    def pallet_ok(loc_stock, pkg_unit):
        if pallet_rule == "FULL_PALLET_ONLY" and pkg_unit:
            try:
                pallet_size = float(pkg_unit)
                if pallet_size > 0:
                    return loc_stock >= qty_needed and (qty_needed % pallet_size == 0)
            except (ValueError, TypeError):
                pass
        return loc_stock >= qty_needed

    start_loc = starting_location if starting_location in stock_map else "PT12"

    # --- Step 1: check starting location first ---
    start_stock, start_has_stock = stock_map.get(start_loc, (0.0, False))
    if start_has_stock and pallet_ok(start_stock, packaging_unit):
        return (start_loc, "false", "CASE_1",
                f"Order already at {start_loc} and stock is sufficient. Auto-confirm, no change.",
                "false")

    # --- Step 2: walk priority fallback list, skipping starting location ---
    priority_list = PRIORITY_RULES.get(customer_group, PRIORITY_RULES["Default"])
    auth_rules = AUTH_RULES.get(customer_group, AUTH_RULES["Default"])

    for loc in priority_list:
        if loc == start_loc:
            continue  # already checked in Step 1
        loc_stock, loc_has_stock = stock_map.get(loc, (0.0, False))
        if loc_has_stock and pallet_ok(loc_stock, packaging_unit):
            needs_auth = auth_rules.get(loc, True)
            auth_str = "true" if needs_auth else "false"
            return (loc, auth_str, "CASE_2",
                    f"Starting location {start_loc} short on stock. Reallocated to {loc}. "
                    f"Auth required: {needs_auth}. PEGA ticket to be raised.",
                    "true")

    # --- Step 3: nothing available anywhere ---
    return (None, "false", "CASE_3",
            f"No location (including starting location {start_loc}) has sufficient stock. Drop line / requeue.",
            "false")

# COMMAND ----------

df_allocated = (
    df_has_stock
    .withColumn(
        "allocation",
        allocate(
            F.col("customer_group"), F.col("starting_location"), F.col("order_quantity"),
            F.col("stock_PT12"), F.col("stock_PT11"), F.col("stock_PT15"),
            F.col("has_stock_PT12"), F.col("has_stock_PT11"), F.col("has_stock_PT15"),
            F.col("pallet_rule"), F.col("packaging_unit"),
        )
    )
    .withColumn("allocated_location", F.col("allocation.allocated_location"))
    .withColumn("requires_auth",      F.col("allocation.requires_auth") == "true")
    .withColumn("case_number",        F.col("allocation.case_number"))
    .withColumn("case_reason",        F.col("allocation.case_reason"))
    .withColumn("location_changed",   F.col("allocation.location_changed") == "true")
    .drop("allocation")
)

# COMMAND ----------

# Orders whose material has no stock record at all anywhere -> forced CASE_3
df_no_stock_final = (
    df_no_stock_at_all
    .withColumn("allocated_location", F.lit(None).cast("string"))
    .withColumn("requires_auth",      F.lit(False))
    .withColumn("case_number",        F.lit("CASE_3"))
    .withColumn("case_reason",        F.lit("Material not found in stock data. Drop line."))
    .withColumn("location_changed",   F.lit(False))
    .withColumn("stock_PT12", F.lit(0.0)).withColumn("stock_PT11", F.lit(0.0)).withColumn("stock_PT15", F.lit(0.0))
    .withColumn("has_stock_PT12", F.lit(False)).withColumn("has_stock_PT11", F.lit(False)).withColumn("has_stock_PT15", F.lit(False))
    .withColumn("available_locations_count", F.lit(0))
    .withColumn("available_locations_list", F.lit(""))
    .withColumn("total_stock", F.lit(0.0))
    .withColumn("warehouse_type", F.lit(None).cast("string"))
    .withColumn("packaging_unit", F.lit(None).cast("string"))
    .withColumn("palletization_indicator", F.lit(None).cast("string"))
    .withColumn("plant", F.lit(None).cast("string"))
)

shared_cols = [c for c in df_allocated.columns if c in df_no_stock_final.columns]
df_final_allocated = df_allocated.select(shared_cols).union(df_no_stock_final.select(shared_cols))

print("Allocation case distribution:")
df_final_allocated.groupBy("case_number").count().orderBy("case_number").show()

# COMMAND ----------

# MAGIC %md ## 5 — PEGA ticket fields & dashboard status

# COMMAND ----------

df_output = (
    df_final_allocated
    .withColumn("sap_action",
        F.when(F.col("case_number") == "CASE_1", F.lit("AUTO_CONFIRM_NO_CHANGE"))
         .when(F.col("case_number") == "CASE_2", F.lit("UPDATE_STORAGE_LOCATION"))
         .otherwise(F.lit("DROP_LINE")))
    .withColumn("pega_ticket_required", F.col("case_number") == "CASE_2")
    .withColumn("pega_ticket_type",
        F.when(F.col("case_number") == "CASE_2", F.lit("ORDER_VALIDATION")).otherwise(F.lit(None).cast("string")))
    .withColumn("approval_status",
        F.when(F.col("case_number") == "CASE_1", F.lit("AUTO_APPROVED"))
         .when((F.col("case_number") == "CASE_2") & F.col("requires_auth"), F.lit("PENDING_APPROVAL"))
         .when((F.col("case_number") == "CASE_2") & ~F.col("requires_auth"), F.lit("AUTO_APPROVED"))
         .otherwise(F.lit("NOT_APPLICABLE")))
    .withColumn("dashboard_status",
        F.when(F.col("case_number") == "CASE_1", F.lit("✅ Auto-Confirmed — No Change"))
         .when((F.col("case_number") == "CASE_2") & F.col("requires_auth"), F.lit("🔐 Pending Approval — Ticket to PEGA"))
         .when((F.col("case_number") == "CASE_2") & ~F.col("requires_auth"), F.lit("📋 Location Changed — Ticket to PEGA"))
         .otherwise(F.lit("❌ Drop Line — No Stock")))
    .withColumn("_processed_timestamp", F.lit(PROCESS_TS))
)

# COMMAND ----------

# MAGIC %md ## 6 — Combine with Silver exceptions for the dashboard table

# COMMAND ----------

df_exceptions_gold = (
    df_exceptions_silver
    .withColumn("customer_group", F.lit("N/A"))
    .withColumn("pallet_rule", F.lit("N/A"))
    .withColumn("starting_location", F.col("storage_location"))
    .withColumn("allocated_location", F.lit(None).cast("string"))
    .withColumn("requires_auth", F.lit(False))
    .withColumn("case_number", F.lit("EXCEPTION"))
    .withColumn("case_reason", F.col("exception_reason"))
    .withColumn("location_changed", F.lit(False))
    .withColumn("sap_action", F.lit("NO_ACTION"))
    .withColumn("pega_ticket_required", F.lit(False))
    .withColumn("pega_ticket_type", F.lit(None).cast("string"))
    .withColumn("approval_status", F.lit("NOT_APPLICABLE"))
    .withColumn("dashboard_status", F.lit("⚠️ Exception — No Change"))
    .withColumn("_processed_timestamp", F.lit(PROCESS_TS))
    .withColumn("stock_PT12", F.lit(None).cast("double")).withColumn("stock_PT11", F.lit(None).cast("double")).withColumn("stock_PT15", F.lit(None).cast("double"))
    .withColumn("has_stock_PT12", F.lit(None).cast("boolean")).withColumn("has_stock_PT11", F.lit(None).cast("boolean")).withColumn("has_stock_PT15", F.lit(None).cast("boolean"))
    .withColumn("available_locations_count", F.lit(None).cast("int"))
    .withColumn("available_locations_list", F.lit(None).cast("string"))
    .withColumn("total_stock", F.lit(None).cast("double"))
    .withColumn("warehouse_type", F.lit(None).cast("string"))
    .withColumn("packaging_unit", F.lit(None).cast("string"))
    .withColumn("palletization_indicator", F.lit(None).cast("string"))
    .withColumn("plant", F.lit(None).cast("string"))
)

all_output_cols = [
    "sales_order_number", "sales_order_line_item", "material_number", "material_description",
    "customer_name", "customer_group", "sold_to_party", "ship_to_party",
    "order_quantity", "required_delivery_date", "item_category", "order_type",
    "country_code", "client_code", "pallet_rule",
    "starting_location", "plant", "warehouse_type", "packaging_unit", "palletization_indicator",
    "stock_PT12", "stock_PT11", "stock_PT15",
    "has_stock_PT12", "has_stock_PT11", "has_stock_PT15",
    "available_locations_count", "available_locations_list", "total_stock",
    "allocated_location", "location_changed", "requires_auth",
    "case_number", "case_reason", "sap_action",
    "pega_ticket_required", "pega_ticket_type",
    "approval_status", "dashboard_status",
    "validation_status", "_processed_timestamp",
]

def align_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("string"))
    return df.select(cols)

df_final = align_cols(df_output, all_output_cols).union(align_cols(df_exceptions_gold, all_output_cols))

print(f"Final allocation output rows: {df_final.count()}")

# COMMAND ----------

(df_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.allocation_output"))

print(f"Written → {CATALOG}.{GOLD_SCHEMA}.allocation_output")

# COMMAND ----------

# MAGIC %md ## 7 — Data Product 1: Allocation Feasibility Output
# MAGIC
# MAGIC Only CASE_2 rows (location is changing) — one record per material/order
# MAGIC showing everything needed to evaluate feasibility before a ticket is raised.

# COMMAND ----------

df_feasibility = (
    df_final
    .filter(F.col("case_number") == "CASE_2")
    .withColumn("allocated_stock_qty",
        F.when(F.col("allocated_location") == "PT12", F.col("stock_PT12"))
         .when(F.col("allocated_location") == "PT11", F.col("stock_PT11"))
         .when(F.col("allocated_location") == "PT15", F.col("stock_PT15"))
         .otherwise(F.lit(0.0)))
    .select(
        "sales_order_number", "sales_order_line_item", "material_number", "material_description",
        F.lit("Yes").alias("allocation_feasibility_flag"),
        F.col("order_quantity").alias("required_quantity"),
        F.col("allocated_stock_qty").alias("available_quantity"),
        F.col("allocated_location").alias("warehouse_selection_recommendation"),
        F.col("allocated_location").alias("new_recommended_warehouse_location"),
        F.concat_ws(" | ", F.col("customer_group"), F.col("pallet_rule")).alias("client_specific_constraint_indicators"),
        F.when(F.col("pallet_rule") == "FULL_PALLET_ONLY", F.lit("Yes")).otherwise(F.lit("No"))
            .alias("full_pallet_requirement_flag"),
        F.col("available_locations_list").alias("allowed_restricted_storage_locations"),
        F.when(F.col("requires_auth"), F.lit("Yes")).otherwise(F.lit("No")).alias("client_authorization_needed"),
        F.lit(None).cast("string").alias("exception_indicator"),
        F.when(F.col("requires_auth"), F.lit("Yes")).otherwise(F.lit("No")).alias("approval_requirement_flag"),
        F.col("approval_status").alias("business_confirmation_status"),
        F.lit(None).cast("string").alias("reason_for_non_feasibility"),
        "_processed_timestamp",
    )
)

(df_feasibility.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.allocation_feasibility_output"))

print(f"Allocation feasibility records (CASE_2 only): {df_feasibility.count()}")
print(f"Written → {CATALOG}.{GOLD_SCHEMA}.allocation_feasibility_output")

# COMMAND ----------

# MAGIC %md ## 8 — Data Product 2: Order Validation Ticket (PEGA)
# MAGIC
# MAGIC Only CASE_2 rows where authorization is required — these are the tickets
# MAGIC that wait for UL approval before the location change is submitted to the OV team.

# COMMAND ----------

df_order_validation_ticket = (
    df_final
    .filter((F.col("case_number") == "CASE_2") & (F.col("requires_auth") == True))
    .withColumn("row_num", F.monotonically_increasing_id())
    .select(
        F.concat(F.lit("PEGA-OV-"), F.lpad(F.col("row_num").cast("string"), 5, "0")).alias("ticket_id"),
        F.lit("Order Validation").alias("ticket_type"),
        F.concat_ws("-", F.col("sales_order_number").cast("string"), F.col("sales_order_line_item").cast("string"))
            .alias("sales_order_number_line_items"),
        F.col("starting_location").alias("old_storage_location"),
        F.col("allocated_location").alias("new_storage_location"),
        "material_number",
        F.col("case_reason").alias("reason_for_change"),
        F.concat(F.lit("Warehouse_Change_"), F.col("sales_order_number").cast("string"), F.lit(".pdf"))
            .alias("attachment_file_name"),
        F.lit("WAREHOUSE_ALLOCATION_AGENT").alias("requestor_id_analyst_id"),
        F.lit("UL_APPROVAL_TEAM").alias("routing_team"),
        F.lit(PROCESS_TS).alias("ticket_creation_timestamp"),
        F.lit("Open").alias("ticket_status"),
    )
)

(df_order_validation_ticket.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.order_validation_ticket_pega"))

print(f"Order validation tickets (CASE_2, auth required only): {df_order_validation_ticket.count()}")
print(f"Written → {CATALOG}.{GOLD_SCHEMA}.order_validation_ticket_pega")

# COMMAND ----------

# MAGIC %md ## 9 — Data Product 3: Allocation Execution & Delivery Creation Output
# MAGIC
# MAGIC All 298 rows (216 valid orders + 82 exceptions) — original SAP order info,
# MAGIC plus the updated allocation info where applicable, plus an execution_status.

# COMMAND ----------

df_execution_delivery_final = (
    df_final
    .withColumn("final_allocated_quantity", F.col("order_quantity"))
    .withColumn("delivery_number", F.col("sales_order_number").cast("string"))
    .withColumn("delivery_creation_timestamp", F.lit(PROCESS_TS))
    .withColumn("warehouse_plant_executing_delivery", F.col("starting_location"))
    .withColumn("allocated_storage_location", F.col("starting_location"))
    .withColumn("sales_order_reference",
        F.concat_ws("-", F.col("sales_order_number").cast("string"), F.col("sales_order_line_item").cast("string")))
    .withColumn("execution_status",
        F.when(F.col("case_number") == "CASE_1", F.lit("Proceed to raise a ticket"))
         .when((F.col("case_number") == "CASE_2") & (F.col("requires_auth") == True), F.lit("Request for Auth"))
         .when((F.col("case_number") == "CASE_2") & (F.col("requires_auth") == False), F.lit("Change location, proceed to raise a ticket"))
         .when(F.col("case_number") == "CASE_3", F.lit("Drop Line"))
         .when(F.col("case_number") == "EXCEPTION", F.lit("Proceed to raise a ticket"))
         .otherwise(F.lit(None).cast("string")))
)

(df_execution_delivery_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.allocation_execution_delivery_output"))

print(f"Execution/delivery records (all orders + exceptions): {df_execution_delivery_final.count()}")
print(f"Written → {CATALOG}.{GOLD_SCHEMA}.allocation_execution_delivery_output")

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

print("=== Gold Layer Complete ===")
df_final.groupBy("case_number", "dashboard_status").count().orderBy("case_number").show(truncate=False)
print(f"\nAllocation Feasibility Output      : {df_feasibility.count()}")
print(f"Order Validation Ticket (PEGA)     : {df_order_validation_ticket.count()}")
print(f"Allocation Execution & Delivery    : {df_execution_delivery_final.count()}")
print(f"Timestamp                          : {PROCESS_TS}")