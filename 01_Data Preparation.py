# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/optimized-picking. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/order-picking-optimization.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to prepare the data for use in the picking optimization. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC As described in the *PK 00: Intro & Config* notebooks, our goal is to optimize orders for in-store picking given a real-world store layout.  To support this, we will make use of the [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis), made available as part of a past Kaggle competition. To prepare this data for use in our analysis, we will load it into the Databricks environment, map products to zones in a provided store layout, and set a priority value which will serve to indicate a product's preferred picking priority given its fragility (or other special handling needs).

# COMMAND ----------

# MAGIC %md ## Step 1: Data Preparation
# MAGIC 
# MAGIC The data in the Instacart dataset should be [downloaded](https://www.kaggle.com/c/instacart-market-basket-analysis) and uploaded to cloud storage. The cloud storage location should then be [mounted](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) to the Databricks file system as shown here:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=240>
# MAGIC 
# MAGIC **NOTE** The name of the mount point, file locations and database used is configurable within the *PK 00: Intro & Config* notebook. 
# MAGIC We have automated this data setup step for you in a temporary location */tmp/instacart_order_picking*.

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# MAGIC %md
# MAGIC The individual files that make up each entity in this dataset can then be presented as a queryable table as part of a database with a high-level schema as follows:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql import window as w

# COMMAND ----------

# DBTITLE 1,Reset the Database Environment
# drop & create database
_ = spark.sql('DROP DATABASE IF EXISTS {0} CASCADE'.format(config['database']))
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database']))

# COMMAND ----------

# DBTITLE 1,Set Current Database
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# DBTITLE 1,Define Table Creation Helper Functions
def read_data(file_path, schema):
  df = (
    spark
      .read
      .csv(
        file_path,
        header=True,
        schema=schema
        )
    )
  return df

def write_data(df, table_name):
   _ = (
       df
        .write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema','true')
        .saveAsTable(table_name)
       )  

# COMMAND ----------

# DBTITLE 1,Load the Data To Tables
# orders data
# ---------------------------------------------------------
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])

orders = read_data(config['orders_path'], orders_schema)
write_data( orders, '{0}.orders'.format(config['database']))
# ---------------------------------------------------------

# products
# ---------------------------------------------------------
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])

products = read_data( config['products_path'], products_schema)
write_data( products, '{0}.products'.format(config['database']))
# ---------------------------------------------------------

# order products
# ---------------------------------------------------------
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])

order_products = read_data( config['order_products_path'], order_products_schema)
write_data( order_products, '{0}.order_products'.format(config['database']))
# ---------------------------------------------------------

# departments
# ---------------------------------------------------------
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])

departments = read_data( config['departments_path'], departments_schema)
write_data( departments, '{0}.departments'.format(config['database']))
# ---------------------------------------------------------

# aisles
# ---------------------------------------------------------
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])

aisles = read_data( config['aisles_path'], aisles_schema)
write_data( aisles, '{0}.aisles'.format(config['database']))
# ---------------------------------------------------------

# COMMAND ----------

# DBTITLE 1,Present Tables in Database
display(
  spark
    .sql('SHOW TABLES IN {0}'.format(config['database']))
  )

# COMMAND ----------

# MAGIC %md ## Step 2: Map Departments to Zones
# MAGIC 
# MAGIC Next, we need to map our products to the fifteen zones found within the provided store layout.  In order to keep similar products together within a zone, we will map each of the 21 Instacart departments to zones as follows:
# MAGIC 
# MAGIC **NOTE** We have no knowledge as to where particular departments might reside within the provided store layout.  We attempted to group a few of the smaller departments (based on item purchase frequencies) into zones based on personal shopping experiences but otherwise the department-to-zone assignments should be seen as arbitrary.

# COMMAND ----------

# DBTITLE 1,Define Department-Zone Mappings
department_zones = (
  spark.createDataFrame(
      [ 
      (1,'frozen',1),
      (2,'other',15),
      (3,'bakery',2),
      (4,'produce',3),
      (5,'alcohol',4),
      (6,'international',5),
      (7,'beverages',6),
      (8,'pets',7),
      (9,'dry goods pasta',5),
      (10,'bulk',8),
      (11,'personal care',7),
      (12,'meat seafood',9),
      (13,'pantry',10),
      (14,'breakfast',11),
      (15,'canned goods',12),
      (16,'dairy eggs',13),
      (17,'household',7),
      (18,'babies',7),
      (19,'snacks',8),
      (20,'deli',14),
      (21,'missing',15)
      ],
      schema=['department_id','department','zone']
      )
  )

# COMMAND ----------

# DBTITLE 1,Write Department-Zone Mappings to Table
write_data( department_zones, '{0}.department_zones'.format(config['database']))

# COMMAND ----------

# MAGIC %md ## Step 3: Assign Product Priorities
# MAGIC 
# MAGIC When picking items in a grocery scenario, item fragility is often a key consideration.  If you pickup bread first and then something heavy like a carton of milk, you run the risk of crushing the bread. Similarly, you wouldn't want to pickup ice cream at the start of a large order as it will likely begin melting before you reach the checkout. 
# MAGIC 
# MAGIC When assigning a fragility score to a product, product weight and volume as well as special handling, *.e.g* refrigeration, requirements are often taken into consideration.  Because we do not have that information for the products in the Instacart dataset, we will assign a random value between 0.01 and 0.99 to each product to represent the *priority* with which it should be picked up. Higher priority items, *i.e.* those with scores closer to 1.00, should be picked up prior to those with lower scores. We will round this value to 2 decimal places as any level of precision below that is not likely to be of interest from a prioritization standpoint and the extra decimal places would only create recalcitrant precedence constraints during optimization:
# MAGIC 
# MAGIC **NOTE** Priority values of 1.00 and 0.00 are being reserved for the starting and ending points of a picking sequence as will be explained in the next notebook.

# COMMAND ----------

# DBTITLE 1,Assign Product Priorities
products = (
  spark
    .table('products')
    .withColumn('priority',f.round(f.rand(),2))
    .withColumn('priority',f.expr('case when priority=0.0 then 0.01 when priority=1.0 then 0.99 else priority end')) # make sure no 0.00 or 1.00 priorities
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('products')
  )

display(spark.table('products'))

# COMMAND ----------

# MAGIC %md ## Step 4: Assemble Orders for Picking
# MAGIC 
# MAGIC Lastly, we'll simplify the access to our order data through the definition of a view. This view will bring together required order details with priority and zone assignments as follows:

# COMMAND ----------

# DBTITLE 1,Define View for Order Picking
# MAGIC %sql
# MAGIC 
# MAGIC CREATE VIEW orders_for_picking
# MAGIC AS
# MAGIC   SELECT 
# MAGIC     a.order_id,
# MAGIC     a.order_dow,
# MAGIC     a.order_hour_of_day,
# MAGIC     b.product_id,
# MAGIC     b.add_to_cart_order,
# MAGIC     c.priority,
# MAGIC     d.zone
# MAGIC   FROM picking.orders a
# MAGIC   INNER JOIN picking.order_products b
# MAGIC     ON a.order_id=b.order_id
# MAGIC   INNER JOIN picking.products c
# MAGIC     ON b.product_id=c.product_id
# MAGIC   INNER JOIN picking.department_zones d
# MAGIC     ON c.department_id=d.department_id
# MAGIC   ORDER BY
# MAGIC     a.order_id,
# MAGIC     b.add_to_cart_order

# COMMAND ----------

# DBTITLE 1,Display Order Details
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM orders_for_picking
# MAGIC ORDER BY order_id, add_to_cart_order

# COMMAND ----------

# MAGIC %md ##Step 5: Explore the Dataset
# MAGIC 
# MAGIC With our data prepared, it may be helpful to examine the products and zones associated with the orders in our dataset.  The Instacart dataset contains 3.3-million orders with an average of 10 products requiring the traversal of 4 to 5 zones per order.  There's considerable variation in order sizes and the traversal required to complete each so that we should expect to see a wide range of outcomes with our optimizations:

# COMMAND ----------

# DBTITLE 1,Calculate Summary Statistics
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   COUNT(order_id) as orders,
# MAGIC   AVG(products) as avg_products_per_order,
# MAGIC   MIN(products) as min_products_per_order,
# MAGIC   MAX(products) as max_products_per_order,
# MAGIC   AVG(zones) as avg_zones_per_order,
# MAGIC   MIN(zones) as min_zones_per_order,
# MAGIC   MAX(zones) as max_zones_per_order
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     order_id,
# MAGIC     COUNT(DISTINCT zone) as zones,
# MAGIC     COUNT(DISTINCT product_id) as products
# MAGIC   FROM orders_for_picking
# MAGIC   GROUP BY order_id
# MAGIC   )

# COMMAND ----------

# MAGIC %md © 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | ortools                                  | Google OR-Tools python libraries and modules | Apache 2.0    | https://pypi.org/project/ortools/                       |
# MAGIC | tsplib95 | TSPLIB95 works with TSPLIB95 files | Apache 2.0 | https://pypi.org/project/tsplib95/ |
# MAGIC | SOP-Optimization |Solving SOP with SA, GRASP, Tabu Search algorithms |  | https://github.com/salehafzoon/SOP-optimization |
