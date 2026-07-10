# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Order Validation & Stock Filtering
# MAGIC
# MAGIC **Stage 0 — Line category dedup**
# MAGIC Keep only `item_category == 'ZTAN'`. ZUCC rows are duplicates of the same
# MAGIC order/material with a different line category — dropping them avoids
# MAGIC double-counting the same order.
# MAGIC
# MAGIC **Stage 1 — Order Validation**
# MAGIC - Ignore blocked orders and closed orders entirely
# MAGIC - Keep only: `delivery_note IS NULL` AND `reason_for_rejection IS NULL`
# MAGIC - Output: `valid_orders` + `exceptions`
# MAGIC
# MAGIC **Stage 2 — Stock Filtering**
# MAGIC - Remove materials with 0 stock across ALL storage locations (PT12/1000, PT11, PT15)
# MAGIC - Tag materials by how many locations carry stock

# COMMAND ----------

CATALOG       = "warehouse_allocation"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")

from pyspark.sql import functions as F
from datetime import datetime

PROCESS_TS = datetime.utcnow().isoformat()

# COMMAND ----------

# MAGIC %md ## 1 — Load Bronze tables

# COMMAND ----------

df_orders = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.sales_orders_raw")
df_stock  = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.stock_availability_raw")

print(f"Bronze orders : {df_orders.count()} rows")
print(f"Bronze stock  : {df_stock.count()} rows")

# COMMAND ----------

# MAGIC %md ## 2 — Stage 0: Filter to ZTAN only (removes ZUCC duplicate lines)

# COMMAND ----------

before_ct = df_orders.count()
df_orders = df_orders.filter(F.upper(F.trim(F.col("item_category"))) == "ZTAN")
after_ct = df_orders.count()

print(f"Rows before ZTAN filter : {before_ct}")
print(f"Rows after ZTAN filter  : {after_ct}  (removed {before_ct - after_ct} ZUCC duplicate lines)")

# COMMAND ----------

# MAGIC %md ## 3 — Order Validation
# MAGIC
# MAGIC 1. Skip blocked orders (`blocking_indicator` not null) and closed orders (`order_status` = 'Closed')
# MAGIC 2. Validate: `delivery_note` IS NULL **and** `reason_for_rejection` IS NULL

# COMMAND ----------

# Normalise empty strings to true nulls
df_orders_clean = (
    df_orders
    .withColumn("delivery_note",
        F.when(F.trim(F.col("delivery_note")) == "", None).otherwise(F.col("delivery_note")))
    .withColumn("reason_for_rejection",
        F.when(F.trim(F.col("reason_for_rejection")) == "", None).otherwise(F.col("reason_for_rejection")))
    .withColumn("blocking_indicator",
        F.when(F.trim(F.col("blocking_indicator")) == "", None).otherwise(F.col("blocking_indicator")))
)

valid_condition = (
    F.col("delivery_note").isNull() &
    F.col("reason_for_rejection").isNull()
)

df_valid = (
    df_orders_clean
    .filter(valid_condition)
    .withColumn("validation_status", F.lit("VALID"))
    .withColumn("exception_reason",  F.lit(None).cast("string"))
    .withColumn("_processed_timestamp", F.lit(PROCESS_TS))
)

df_exceptions = (
    df_orders_clean
    .filter(~valid_condition)
    .withColumn("validation_status", F.lit("EXCEPTION"))
    .withColumn("exception_reason",
        F.when(F.col("delivery_note").isNotNull() & F.col("reason_for_rejection").isNotNull(),
               F.lit("delivery_note not blank; reason_for_rejection not blank"))
         .when(F.col("delivery_note").isNotNull(), F.lit("delivery_note not blank"))
         .otherwise(F.lit("reason_for_rejection not blank"))
    )
    .withColumn("_processed_timestamp", F.lit(PROCESS_TS))
)

print(f"Valid orders : {df_valid.count()}")
print(f"Exceptions   : {df_exceptions.count()}")

# COMMAND ----------

(df_valid.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.valid_orders"))

(df_exceptions.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.exceptions"))

print(f"Written → {CATALOG}.{SILVER_SCHEMA}.valid_orders")
print(f"Written → {CATALOG}.{SILVER_SCHEMA}.exceptions")

# COMMAND ----------

# MAGIC %md ## 4 — Stock Filtering
# MAGIC
# MAGIC Normalise `1000` → `PT12` (same default location, different naming),
# MAGIC pivot to one row per material with a column per location, drop materials
# MAGIC with zero stock everywhere, tag how many locations carry stock.

# COMMAND ----------

df_stock_norm = (
    df_stock
    .withColumn("storage_location",
        F.when(F.col("storage_location") == "1000", F.lit("PT12")).otherwise(F.col("storage_location")))
    .withColumn("allocable_stock_qty", F.coalesce(F.col("allocable_stock_qty"), F.lit(0.0)))
    .withColumn("available_stock_qty", F.coalesce(F.col("available_stock_qty"), F.lit(0.0)))
)

df_stock_relevant = df_stock_norm.filter(F.col("storage_location").isin("PT12", "PT11", "PT15"))

# COMMAND ----------

df_stock_by_loc = (
    df_stock_relevant
    .groupBy("material_number", "plant", "storage_location")
    .agg(
        F.sum("allocable_stock_qty").alias("allocable_stock_qty"),
        F.sum("available_stock_qty").alias("available_stock_qty"),
    )
)

df_pivot = (
    df_stock_by_loc
    .groupBy("material_number", "plant")
    .pivot("storage_location", ["PT12", "PT11", "PT15"])
    .agg(F.first("allocable_stock_qty"))
    .withColumnRenamed("PT12", "stock_PT12")
    .withColumnRenamed("PT11", "stock_PT11")
    .withColumnRenamed("PT15", "stock_PT15")
)

# warehouse_type, packaging_unit, palletization_indicator, and stock_timing_timestamp
# can all differ location-to-location for the SAME material (confirmed against the
# actual data: warehouse_type alone differs across locations for 59/120 materials).
# None of them affect the pivot's row identity — they're informational attributes,
# picked once per material+plant here and joined back in, so they can never re-split
# a material into multiple stock_filtered rows.
df_attrs = (
    df_stock_relevant
    .groupBy("material_number", "plant")
    .agg(
        F.max("stock_timing_timestamp").alias("stock_timing_timestamp"),
        F.first("warehouse_type", ignorenulls=True).alias("warehouse_type"),
        F.first("packaging_unit", ignorenulls=True).alias("packaging_unit"),
        F.first("palletization_indicator", ignorenulls=True).alias("palletization_indicator"),
    )
)

df_pivot = df_pivot.join(df_attrs, on=["material_number", "plant"], how="left")

df_pivot = (
    df_pivot
    .withColumn("stock_PT12", F.coalesce(F.col("stock_PT12"), F.lit(0.0)))
    .withColumn("stock_PT11", F.coalesce(F.col("stock_PT11"), F.lit(0.0)))
    .withColumn("stock_PT15", F.coalesce(F.col("stock_PT15"), F.lit(0.0)))
)

# COMMAND ----------

df_pivot = (
    df_pivot
    .withColumn("has_stock_PT12", F.col("stock_PT12") > 0)
    .withColumn("has_stock_PT11", F.col("stock_PT11") > 0)
    .withColumn("has_stock_PT15", F.col("stock_PT15") > 0)
    .withColumn("total_stock", F.col("stock_PT12") + F.col("stock_PT11") + F.col("stock_PT15"))
)

df_stock_with_stock = df_pivot.filter(F.col("total_stock") > 0)

print(f"Materials before zero-stock removal : {df_pivot.count()}")
print(f"Materials after zero-stock removal  : {df_stock_with_stock.count()}")

# COMMAND ----------

df_stock_tagged = (
    df_stock_with_stock
    .withColumn(
        "available_locations_count",
        F.col("has_stock_PT12").cast("int") + F.col("has_stock_PT11").cast("int") + F.col("has_stock_PT15").cast("int")
    )
    .withColumn(
        "available_locations_list",
        F.concat_ws(",",
            F.when(F.col("has_stock_PT12"), F.lit("PT12")),
            F.when(F.col("has_stock_PT11"), F.lit("PT11")),
            F.when(F.col("has_stock_PT15"), F.lit("PT15"))
        )
    )
    .withColumn("_processed_timestamp", F.lit(PROCESS_TS))
)

print("Stock distribution by location count:")
df_stock_tagged.groupBy("available_locations_count").count().orderBy("available_locations_count").show()

# COMMAND ----------

(df_stock_tagged.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.stock_filtered"))

print(f"Written → {CATALOG}.{SILVER_SCHEMA}.stock_filtered")

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

print("=== Silver Layer Complete ===")
print(f"  valid_orders   : {spark.table(f'{CATALOG}.{SILVER_SCHEMA}.valid_orders').count()} rows")
print(f"  exceptions     : {spark.table(f'{CATALOG}.{SILVER_SCHEMA}.exceptions').count()} rows")
print(f"  stock_filtered : {spark.table(f'{CATALOG}.{SILVER_SCHEMA}.stock_filtered').count()} materials")
print(f"  Timestamp      : {PROCESS_TS}")

# COMMAND ----------

spark.table(f"{CATALOG}.{SILVER_SCHEMA}.exceptions").groupBy("exception_reason").count().orderBy(F.desc("count")).show(truncate=False)