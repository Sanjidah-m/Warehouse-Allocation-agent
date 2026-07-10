# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Raw Ingestion
# MAGIC Reads raw SAP extracts (Sales Orders + Stock Availability) from AWS S3
# MAGIC and lands them as Delta tables with standardised column names.
# MAGIC
# MAGIC **Source files expected in S3 (csv, matching your SAP export format):**
# MAGIC - `sales_orders/` → ORDER FLOW EXTRACT
# MAGIC - `stock_availability/` → STOCK AVAILABILITY REPORT
# MAGIC
# MAGIC Column names below are renamed from the exact headers in your SAP export
# MAGIC to snake_case for consistency across the pipeline.

# COMMAND ----------

# MAGIC %md ## Config

# COMMAND ----------

S3_BUCKET         = "s3://warehouse-sm1/"
SALES_ORDER_PATH  = f"s3://warehouse-sm1/Sales-order"
STOCK_PATH        = f"s3://warehouse-sm1/stock-availability"

CATALOG = "warehouse_allocation"
SCHEMA  = "bronze"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"Catalog: {CATALOG}.{SCHEMA} ready")

# COMMAND ----------

# MAGIC %md ## Column rename maps
# MAGIC
# MAGIC Left = exact header from your SAP export file. Right = standardised name used
# MAGIC throughout Silver/Gold. Edit the LEFT side only if your real SAP header text differs.

# COMMAND ----------

SALES_COLUMN_MAP = {
    "Sales Order Number":     "sales_order_number",
    "Sales Order Line Item":  "sales_order_line_item",
    "Customer Name":          "customer_name",          # Walmart / Cencosud / TentA / DMart
    "Sold to Party":          "sold_to_party",
    "Ship to Party":          "ship_to_party",
    "Item Category":          "item_category",          # ZTAN vs ZUCC
    "Material UCC14":         "material_ucc14",
    "Material DU":            "material_du",
    "Storage Location":       "storage_location",        # order's STARTING location (from SAP)
    "Line item":              "line_item_ref",
    "Material Description":   "material_description",
    "Order Type":             "order_type",
    "Delivery Note":          "delivery_note",
    "Reason for Rejection":   "reason_for_rejection",
    "Order Quantity":         "order_quantity",
    "Required Delivery Date": "required_delivery_date",
    "Country Code":           "country_code",
    "Client Code":            "client_code",
    "Material Number":        "material_number",
    "Order Status":           "order_status",
    "Blocking Indicators":    "blocking_indicator",
}

STOCK_COLUMN_MAP = {
    "Material Number":               "material_number",
    "Plant":                         "plant",
    "Storage Location":              "storage_location",
    "Available Stock Quantity":      "available_stock_qty",
    "Allocable Stock Quantity":      "allocable_stock_qty",
    "Warehouse Type":                "warehouse_type",       # Carousel / Non Carousel (info only)
    "Packaging Unit / Pallet Unit":  "packaging_unit",
    "Palletization Indicator":       "palletization_indicator",
    "Stock Timing Timestamp":        "stock_timing_timestamp",
}

# COMMAND ----------

# MAGIC %md ## Helper — read csv from S3 natively via Spark
# MAGIC
# MAGIC Source files are CSV (SAP export dropped straight to S3), so we read them
# MAGIC directly with `spark.read.csv(...)` — no driver-side pandas hop needed.
# MAGIC `header=True` picks up the exact SAP column names, `inferSchema=True` gets
# MAGIC numeric/date typing for free, and a UTF-8 BOM on the first header (common
# MAGIC with Excel-exported CSVs) is stripped explicitly so the rename map still matches.

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime

INGESTION_TS = datetime.utcnow().isoformat()

def read_csv_from_s3(path, column_map):
    """Reads a CSV file straight from S3 with Spark, renames columns per the
    SAP header → snake_case map, and keeps only the mapped columns."""
    df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .option("encoding", "UTF-8")
             .option("multiLine", "true")
             .option("escape", '"')
             .csv(path)
    )
    # strip a UTF-8 BOM that sometimes prefixes the very first header
    df = df.toDF(*[c.lstrip("\ufeff") for c in df.columns])

    for source_col, target_col in column_map.items():
        if source_col in df.columns:
            df = df.withColumnRenamed(source_col, target_col)

    # keep only mapped columns, in case source has extra/unexpected columns
    keep_cols = [c for c in column_map.values() if c in df.columns]
    return df.select(*keep_cols)

# COMMAND ----------

# MAGIC %md ## 1 — Ingest Sales Orders

# COMMAND ----------

df_sales_raw = (
    read_csv_from_s3(SALES_ORDER_PATH, SALES_COLUMN_MAP)
    .withColumn("_ingestion_timestamp", F.lit(INGESTION_TS))
    .withColumn("_source_file", F.lit(SALES_ORDER_PATH))
)

print(f"Sales orders ingested: {df_sales_raw.count()} rows")
df_sales_raw.printSchema()
df_sales_raw.show(5, truncate=False)

# COMMAND ----------

(
    df_sales_raw.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(f"{CATALOG}.{SCHEMA}.sales_orders_raw")
)

print(f"Written → {CATALOG}.{SCHEMA}.sales_orders_raw")

# COMMAND ----------

# MAGIC %md ## 2 — Ingest Stock Availability

# COMMAND ----------

df_stock_raw = (
    read_csv_from_s3(STOCK_PATH, STOCK_COLUMN_MAP)
    .withColumn("_ingestion_timestamp", F.lit(INGESTION_TS))
    .withColumn("_source_file", F.lit(STOCK_PATH))
)

print(f"Stock records ingested: {df_stock_raw.count()} rows")

# COMMAND ----------

(
    df_stock_raw.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(f"{CATALOG}.{SCHEMA}.stock_availability_raw")
)

print(f"Written → {CATALOG}.{SCHEMA}.stock_availability_raw")

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

print("=== Bronze Ingestion Complete ===")
print(f"  sales_orders_raw      : {spark.table(f'{CATALOG}.{SCHEMA}.sales_orders_raw').count()} rows")
print(f"  stock_availability_raw: {spark.table(f'{CATALOG}.{SCHEMA}.stock_availability_raw').count()} rows")
print(f"  Timestamp: {INGESTION_TS}")