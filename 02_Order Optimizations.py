# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/optimized-picking. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/order-picking-optimization.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to explore various optimization strategies for in-store order picking.  

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In this notebook we will apply various strategies for optimizing in-store picking.  Several of these are addressed in the paper on which we are basing much of this work, *[The Buy-Online-Pick-Up-in-Store Retailing Model: Optimization Strategies for In-Store Picking and Packing](https://www.mdpi.com/1999-4893/14/12/350/pdf)*.
# MAGIC 
# MAGIC The point of this notebook is not necessarily to say that one optimization strategy is better than another but instead to show how each can be implemented in a Databricks cluster.  The evaluation of optimization strategies against historical datasets is a common practice in optimization work and one that can be very time-consuming when a large number of (order) instances need to be considered. Using Databricks, we can perform this work in an efficient and scalable manner, allowing us time to more thoroughly evaluate new approaches before making changes to operational practices.
# MAGIC 
# MAGIC **NOTE** The time it takes to complete each of these optimizations can be considerable given we have 3.3-million historical orders to optimize in the full dataset.  You may wish to test this code against a small sample of data and then scale your cluster appropriately to achieve the level of performance you require.  Code has been inserted in the cell titled *Assemble Original Orders* to enable such a sampling. (You'll need to modify the *sample_fraction* variable to perform a fractional sample.)

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install tsplib95==0.6.0  # the SOP optimization code is very sensitive to tsplib95 library versions
# MAGIC %pip install ortools==9.3.10497

# COMMAND ----------

# DBTITLE 1,Retrieve Config
# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql import Window

import numpy as np
import pandas as pd
import os

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import tsplib95

from itertools import groupby
from operator import itemgetter

from random import random

# COMMAND ----------

# DBTITLE 1,Set Current Database
# set current database
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# MAGIC %md ##Step 1: Define Zone Traversal Times
# MAGIC 
# MAGIC A key element many of the optimizations in this notebooks is information about how long it takes to move from zone-to-zone within the store. As mentioned previously, the authors of the paper we are referencing used the layout of a real store, divided into 15 zones, to establish this information:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/picking_store_layout.png' width=600>
# MAGIC 
# MAGIC Using simple calculations documented in their paper, they estimated the number of seconds it would take an individual to move between each zone as follows:
# MAGIC 
# MAGIC |       |Zone 01|Zone 02|Zone 03|Zone 04|Zone 05|Zone 06|Zone 07|Zone 08|Zone 09|Zone 10|Zone 11|Zone 12|Zone 13|Zone 14|Zone 15|
# MAGIC | ---   | --- | ---  | ---  | ---  | ---  | ---  | ---  | ---  | ---  | ---  | ---  | ---  | ---  | ---  | ---  |
# MAGIC |Zone 01|00.00| 15.58| 39.84| 49.84| 44.20| 40.67| 37.14| 40.67| 44.20| 47.73| 51.26| 70.33| 65.96| 74.83| 77.18|
# MAGIC |Zone 02|15.58| 00.00| 24.26| 34.26| 28.62| 25.09| 21.56| 25.09| 28.62| 32.15| 35.68| 54.75| 50.39| 59.25| 61.60|
# MAGIC |Zone 03|39.84| 24.26| 00.00| 24.06| 29.07| 31.76| 28.24| 31.76| 35.29| 38.82| 42.35| 61.42| 57.06| 65.92| 68.28|
# MAGIC |Zone 04|49.84| 34.26| 24.06| 00.00| 21.91| 25.44| 28.97| 32.50| 36.02| 39.55| 43.08| 62.15| 57.79| 66.65| 69.01|
# MAGIC |Zone 05|44.20| 28.62| 29.07| 21.91| 00.00| 16.47| 20.00| 23.53| 27.06| 30.59| 34.12| 53.18| 48.82| 57.69| 60.04|
# MAGIC |Zone 06|40.67| 25.09| 31.76| 25.44| 16.47| 00.00| 16.47| 20.00| 23.53| 27.06| 30.59| 49.66| 45.29| 54.16| 56.51| 
# MAGIC |Zone 07|37.14| 21.56| 28.24| 28.97| 20.00| 16.47| 00.00| 16.47| 20.00| 23.53| 27.06| 46.13| 41.76| 50.63| 52.98|
# MAGIC |Zone 08|40.67| 25.09| 31.76| 32.50| 23.53| 20.00| 16.47| 00.00| 16.47| 20.00| 23.53| 29.66| 25.29| 34.16| 36.52|
# MAGIC |Zone 09|44.20| 28.62| 35.29| 36.02| 27.06| 23.53| 20.00| 16.47| 00.00| 16.47| 20.00| 26.13| 21.76| 30.64| 32.99|
# MAGIC |Zone 10|47.73| 32.15| 38.82| 39.55| 30.59| 27.06| 23.53| 20.00| 16.47| 00.00| 16.47| 22.60| 18.24| 27.11| 29.46|
# MAGIC |Zone 11|51.26| 35.68| 42.35| 43.08| 34.12| 30.59| 27.06| 23.53| 20.00| 16.47| 00.00| 19.07| 14.71| 23.57| 25.92|
# MAGIC |Zone 12|70.33| 54.75| 61.42| 62.15| 53.18| 49.66| 46.13| 29.66| 26.13| 22.60| 19.07| 00.00| 13.77| 22.64| 29.69|
# MAGIC |Zone 13|65.96| 50.39| 57.06| 57.79| 48.82| 45.29| 41.76| 25.29| 21.76| 18.24| 14.71| 13.77| 00.00| 17.59| 14.26|
# MAGIC |Zone 14|74.83| 59.25| 65.92| 66.65| 57.69| 54.16| 50.63| 34.16| 30.64| 27.11| 23.57| 22.64| 17.59| 00.00| 07.20|
# MAGIC |Zone 15|77.18| 61.60| 68.28| 69.01| 60.04| 56.51| 52.98| 36.52| 32.99| 29.46| 25.92| 29.69| 14.26| 07.20| 00.00|
# MAGIC 
# MAGIC To make use of these estimates, we drop them into a matrix:

# COMMAND ----------

# DBTITLE 1,Define Travel Time Matrix
travel_times = [
   [00.00, 15.58, 39.84, 49.84, 44.20, 40.67, 37.14, 40.67, 44.20, 47.73, 51.26, 70.33, 65.96, 74.83, 77.18],
   [15.58, 00.00, 24.26, 34.26, 28.62, 25.09, 21.56, 25.09, 28.62, 32.15, 35.68, 54.75, 50.39, 59.25, 61.60],
   [39.84, 24.26, 00.00, 24.06, 29.07, 31.76, 28.24, 31.76, 35.29, 38.82, 42.35, 61.42, 57.06, 65.92, 68.28],
   [49.84, 34.26, 24.06, 00.00, 21.91, 25.44, 28.97, 32.50, 36.02, 39.55, 43.08, 62.15, 57.79, 66.65, 69.01],
   [44.20, 28.62, 29.07, 21.91, 00.00, 16.47, 20.00, 23.53, 27.06, 30.59, 34.12, 53.18, 48.82, 57.69, 60.04],
   [40.67, 25.09, 31.76, 25.44, 16.47, 00.00, 16.47, 20.00, 23.53, 27.06, 30.59, 49.66, 45.29, 54.16, 56.51], 
   [37.14, 21.56, 28.24, 28.97, 20.00, 16.47, 00.00, 16.47, 20.00, 23.53, 27.06, 46.13, 41.76, 50.63, 52.98],
   [40.67, 25.09, 31.76, 32.50, 23.53, 20.00, 16.47, 00.00, 16.47, 20.00, 23.53, 29.66, 25.29, 34.16, 36.52],
   [44.20, 28.62, 35.29, 36.02, 27.06, 23.53, 20.00, 16.47, 00.00, 16.47, 20.00, 26.13, 21.76, 30.64, 32.99],
   [47.73, 32.15, 38.82, 39.55, 30.59, 27.06, 23.53, 20.00, 16.47, 00.00, 16.47, 22.60, 18.24, 27.11, 29.46],
   [51.26, 35.68, 42.35, 43.08, 34.12, 30.59, 27.06, 23.53, 20.00, 16.47, 00.00, 19.07, 14.71, 23.57, 25.92],
   [70.33, 54.75, 61.42, 62.15, 53.18, 49.66, 46.13, 29.66, 26.13, 22.60, 19.07, 00.00, 13.77, 22.64, 29.69],
   [65.96, 50.39, 57.06, 57.79, 48.82, 45.29, 41.76, 25.29, 21.76, 18.24, 14.71, 13.77, 00.00, 17.59, 14.26],
   [74.83, 59.25, 65.92, 66.65, 57.69, 54.16, 50.63, 34.16, 30.64, 27.11, 23.57, 22.64, 17.59, 00.00, 07.20],
   [77.18, 61.60, 68.28, 69.01, 60.04, 56.51, 52.98, 36.52, 32.99, 29.46, 25.92, 29.69, 14.26, 07.20, 00.00]
  ]

# COMMAND ----------

# MAGIC %md We will be making use of the *travel_times* matrix as part of several distributed calculations.  To ensure each worker in our cluster has access to a local copy of the matrix, we will replicate it to each worker in our cluster using a [broadcast method](https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables) call. The use of a broadcast variable will create a local variable on each worker node within the cluster which will have its own copy of the broadcasted value.  To access this value, code running on a worker simply needs to call the *value* property of the broadcast variable:   
# MAGIC 
# MAGIC **NOTE** This step makes a huge performance difference, especially for the SOP optimizations which perform numerous iterations on each order.

# COMMAND ----------

# DBTITLE 1,Replicate Travel Time Matrix
# replicate time-travel matrix to cluster workers
travel_times_bc = sc.broadcast(travel_times)

# COMMAND ----------

# MAGIC %md It's important to note that while the zones are labeled 1 through 15, the matrix will index them as positions 0 through 14.  You will notice in the code below several places where we adjust the 1-based zone labels to 0-based index positions.

# COMMAND ----------

# MAGIC %md ##Step 2: Assemble Original Orders
# MAGIC 
# MAGIC The Instacart dataset provides a field named *add_to_cart_order* which preserves the sequence with which items were added to an order. This is important because the referenced paper highlights that this is often the order in which items are presented to pickers.  
# MAGIC 
# MAGIC As we extract the items for an order to assemble the sequence within which they need to be picked up, we need to be careful to ensure these items are sequenced using this field.  While there are a few tricks we could use to force the Spark engine to enforce that order as the sequences are assembled, we've elected to write a custom function to explicitly enforce this behavior.
# MAGIC 
# MAGIC The function we've defined is written as a [pandas UDF](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html).  Unlike a traditional user-defined function which receives values from one row at a time, the pandas UDF receives small units of data from Spark representing values from multiple rows of our Spark dataframe. Accessing multiple values at a time in this manner provides us the opportunity to boost the performance of the code even if it adds a bit of complexity to the logic.
# MAGIC 
# MAGIC Pandas UDFs come in multiple flavors.  For our pandas UDF, we've elected to follow a Series-to-Series pattern.  In this pattern, values from one or more fields are delivered to the function as separate pandas Series, each of which preserves the sequence of values between rows in the original dataframe. This type of pandas UDF expects that for each row of data provided a single value is returned.
# MAGIC 
# MAGIC When defining a pandas UDF, we must articulate the structure of the data to be returned.  For a Series-to-Series pandas UDF, a single value is returned for each row delivered to the function.  As such, we simply need to specify the data type of that value.  
# MAGIC 
# MAGIC When expressing the schema of the returned value, we have two options.  We can attach a short-hand *decorator* to our function or define a schema at the time of function *registration*.  With the function below, we will be using a cart-order sequence to sort a set of submitted items.  When those items are products, an array of integer product IDs will be returned.  When those items are zones, an array of integer zone IDs will be returned.  And when those items are priority scores, an array of floating point value will be returned. Because we have different data types associated with different calls to the same unit of logic, we'll define a schema for two separate registrations of the same function to align with Spark's expectations:arate functions, each of which has a slightly different return type.

# COMMAND ----------

# DBTITLE 1,Define Helper Function to Sort Items Based on Cart Order
# function to sort items based on cart order
def sort_items_to_cart_order(items_to_sort_series: pd.Series, cart_order_series: pd.Series) -> pd.Series:
  
  # combine sequence and zones series to form a new dataframe
  df = pd.DataFrame({'cart_order':cart_order_series, 'items':items_to_sort_series})
  df['ordered_items'] = None # initialize with first entry to get datatypes right
  
  # for each order (row) in dataframe
  for i, row in df.iterrows():
    
    # make sure items are sorted in same order as placed in cart
    sorted_items = [z for _,z in sorted(zip(row['cart_order'], row['items']))]
    df.at[i,'ordered_items'] = sorted_items
    
  return df['ordered_items']

# register function to return arrays of integers and floats
sort_items_to_cart_order_INTEGERS = f.pandas_udf(sort_items_to_cart_order, returnType=ArrayType(IntegerType()))
sort_items_to_cart_order_FLOATS = f.pandas_udf(sort_items_to_cart_order, returnType=ArrayType(FloatType()))                        

# COMMAND ----------

# MAGIC %md Now we can assemble the sequence within which orders are fulfilled based on cart ordering.  Please note that all pickers start in zone 1 which represents the entrance to the store and end in zone 15 which represents the store's checkout counters. Each order has been capped with this zone information and priorities for these zones have been set to 1.0 and 0.0 respectively to ensure they are used as starting and ending points during priority-based sorting.  *Null* values are used to identify the lack of products to pick up at the starting and ending points:
# MAGIC 
# MAGIC **NOTE** Zones 1 and 15 represent the store entrance and exit/check-out but do contain products which may need to be picked for a given order.  As a result, you may see zones 1 and 15 represented throughout the order sequence.
# MAGIC 
# MAGIC **NOTE** Adjust the *sample_fraction* variable to indicate the fraction of the 3.3-million records against which to perform optimizations. The closer the value gets to 1.0, the longer the job will take to complete unless you expand your cluster size.

# COMMAND ----------

# DBTITLE 1,Assemble Original Orders
# set value to control fraction of orders to employ in optimization steps
sample_fraction = 0.05

orders = (
  spark
    .table('orders_for_picking')
    .groupBy('order_id')
      .agg(
        f.concat( # assemble list of products for this order
          f.array(f.lit(None)), # entrance is non-product
          f.collect_list('product_id'),
          f.array(f.lit(None))  # exist is non-product
          ).alias('products'),
        f.concat( # assemble list of zones for this order
          f.array(f.lit(1)), # always start at entrance
          f.collect_list('zone'),
          f.array(f.lit(15)) # always end at exit
          ).alias('zones'),
        f.concat( # assemble product priorities for this order
          f.array(f.lit(1.0)),  # entrance gets highest priority
          f.collect_list('priority'),
          f.array(f.lit(0.0))  # exit gets lowest priority 
          ).alias('priorities'),
        f.concat( # assemble cart orders for this order
          f.array(f.lit(0)), # entrance is always add_to_cart_order at position 0
          f.collect_list('add_to_cart_order'),
          f.array(f.size(f.collect_list('add_to_cart_order'))+1) # exit is 1 more than size of original list
          ).alias('cart_order')
        )
  .withColumn('products', sort_items_to_cart_order_INTEGERS('products','cart_order')) # ensure properly sorted based on cart order
  .withColumn('zones', sort_items_to_cart_order_INTEGERS('zones','cart_order'))
  .withColumn('priorities', sort_items_to_cart_order_FLOATS('priorities','cart_order'))
  .sample(fraction=sample_fraction)
  .repartition( sc.defaultParallelism * 20 ) # make sure these data are well distributed
  .select('order_id','products','zones','priorities')
  .cache() # cache in memory to minimize repeats of this step
  )

display(orders)

# COMMAND ----------

# MAGIC %md ##Step 3: Perform Time-Optimized Routing
# MAGIC 
# MAGIC The first algorithm we will apply will be a time-based optimization known as the Traveling Sales Person (TSP) optimization.  In this scenario, we have a fixed number of destinations (zones) we must traverse, and we simply need to minimize the time to reach them all.
# MAGIC 
# MAGIC To solve the TSP problem, we'll make use of [Google's OR-Tools](https://developers.google.com/optimization) which perform route optimization. The implementation of the TSP optimization is very well documented [here](https://developers.google.com/optimization/routing/tsp), so that we've attempted to write a function that encapsulates the documented TSP patterns with as few changes as possible.  This function, when provided a reduced travel-time matrix (for just those zones we need to traverse), will return an optimized sequence.
# MAGIC 
# MAGIC Please note that unlike the standard TSP problem, our problem employs a fixed ending point (zone 15) in addition to a fixed starting point (zone 1).  As explained earlier, these represent the entrance and exit (checkout) zones of the store:

# COMMAND ----------

# DBTITLE 1,Function for TSP Optimization (Adapted from Google OR Tools)
def perform_tsp_routing(travel_times_subset):
  
  # create a dataset to capture parameters for our problem
  # this dictionary simply provides a nice way to bundle
  # up the values we'll use throughout the solution
  def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = travel_times_subset
    data['num_vehicles'] = 1
    data['starts'] = [0] # start with zone in first position (entrance)
    data['ends'] = [len(travel_times_subset)-1] # end with zone in last position (exit)
    return data
  
  # get data for modeling exercise
  data = create_data_model()

  # define a routing index manager
  manager = pywrapcp.RoutingIndexManager(
    len(data['distance_matrix']),
    data['num_vehicles'], 
    data['starts'],
    data['ends']
    )
  
  # define routing model
  routing = pywrapcp.RoutingModel(manager)
  
  # get time to move between two zones 
  # from: https://developers.google.com/optimization/routing/tsp#dist_callback1
  def distance_callback(from_index, to_index):
    #convert indices to nodes
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]
  
  # configure model to calculate time between zones
  transit_callback_index = routing.RegisterTransitCallback(distance_callback)
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
  
  # add distance constraint
  # https://developers.google.com/optimization/routing/vrp#distance_dimension
  dimension_name = 'Distance'
  routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        2000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name
        )
  distance_dimension = routing.GetDimensionOrDie(dimension_name)
  #distance_dimension.SetGlobalSpanCostCoefficient(100)

  # define objective
  search_parameters = pywrapcp.DefaultRoutingSearchParameters()
  search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
  
  # print optimized route details
  # modified from: https://developers.google.com/optimization/routing/tsp#printer
  # modifications allow us to return a sequenced array of zones
  def print_solution(manager, routing, solution):
    """Prints solution on console."""
    #print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = [] #='Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += [manager.IndexToNode(index)] #+= ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += [manager.IndexToNode(index)] #+= ' {}\n'.format(manager.IndexToNode(index))
    #print(plan_output)
    #plan_output += 'Route distance: {}miles\n'.format(route_distance)
    return(plan_output)
  
  # have model solve for objective
  solution = routing.SolveWithParameters(search_parameters)
  if solution:
    return print_solution(manager, routing, solution)

# COMMAND ----------

# MAGIC %md Now we need to write a function to which we can provide our products and zones and receive back an optimized sequence of products.  We will write this as a pandas UDF which will receive one or more rows of data from Spark and return a value for each row delivered to it.  For each row, the function will need to *shrink* the travel-time matrix to include just those zones which need to be traversed.  This matrix will be delivered to the TSP function (defined in the previous cell) for optimization and the returned sequence will be used to identify the sequence of products to pick:
# MAGIC 
# MAGIC **NOTE** This function will return a map-type object, equivalent to a Python dictionary, that contains an optimized set of products along with the sorted zones.  The product list is used for picking while the zone list is used for estimating total traversal time for the order.

# COMMAND ----------

# DBTITLE 1,Define TSP Product Optimization Function
@f.pandas_udf(MapType(StringType(),ArrayType(IntegerType()))) # will return {'key':[val, val, ..., val]}
def get_tsp_optimized_products(products_series: pd.Series, zones_series: pd.Series) -> pd.Series:
  
  results = []
  
  # for each set of zones passed into function
  for i, zones in enumerate(zones_series):
      
    # get unique set of zones in this order
    uniq_zones = list(set(zones))
    uniq_zones.sort()
    
    # count unique set of zones in this order
    zone_count = len(uniq_zones)
    
    # get the travel times between these zones 
    travel_times_subset = []
    for x in uniq_zones:
      for y in uniq_zones:
        travel_times_subset += [travel_times_bc.value[x-1][y-1]]
        
    # convert it into a 2-D array structure
    travel_times_subset = np.array(travel_times_subset).reshape(zone_count, zone_count).tolist()
    
    # get optimized route (with zone indexes)
    raw_opt_zones = perform_tsp_routing(travel_times_subset)
    
    # translate zone indexes back to recognized zones
    opt_zones = []
    for z in raw_opt_zones:
      opt_zones += [uniq_zones[z]]
      
    # create dict with products by zone (key)
    zone_product_mapping = sorted(zip(zones, products_series.iloc[i]))
    zone_product_dict = {z:list(list(zip(*p))[1]) for z,p in groupby(zone_product_mapping, itemgetter(0))}
    
    # map products to optimized zones
    opt_products = []
    zone_products = []
    for z in opt_zones:
      opt_products += zone_product_dict[z]
        
    # capture optimized route
    results.append({
      'products': opt_products,
      'zones': opt_zones
      })
    
  return pd.Series(results)

# COMMAND ----------

# MAGIC %md It's important to note as you review the previous function that the TSP optimization (encapsulated in the *perform_tsp_routing* function) receives a travel time matrix for just the zones associated with a given order.  It returns the sequence of zones based on their position in the reduced matrix provided.  For example, if we had an order with zones \[1, 5, 8, 15\], we'd provide the function with a 4x4 travel time matrix focused on just those zones.  The output would reference each zone by index position in the provided (reduced) matrix.  For example, we might receive back something like \[0, 2, 1, 3\] which we would translate as zones \[1, 8, 5, 15\] given the structure of the matrix submitted to the optimization function. The unit of code commented as *translate zone indexes back to recognized zones* tackles this work.
# MAGIC 
# MAGIC In addition, it's important to consider that within a zone, we are picking all products for that zone.  For example, if you had an order with 2 products in zone 5, you might start with a original zone sequence of \[1, 5, 8, 5, 15\] and receive back an optimized zone sequence of \[1, 8, 5, 15\].  When the picker is sent to zone 5, both products will be picked at that time.  The unit of code commented as *map products to optimized zones* ensures products are sequenced appropriately given the condensed output of the TSP optimization function.
# MAGIC 
# MAGIC With the function now in place, we can now calculate our time-optimized product sequence as follows:

# COMMAND ----------

# DBTITLE 1,Calculate TSP-Optimized Product List
orders = (
  orders
    .withColumn('tsp_products', get_tsp_optimized_products('products','zones'))
  )

display(orders.select('order_id','products','zones','tsp_products'))

# COMMAND ----------

# MAGIC %md ##Step 4: Perform Priority Sorted Routing
# MAGIC 
# MAGIC In the TSP optimization, we ordered the picking list with the goal of minimizing travel time between zones.  While this minimizes the labor cost for the initial pick, it doesn't take into consideration differences in item fragility or the special handling needs associated with some products.  As a result, we may move quickly through a store using a TSP optimization but end up with damaged items at the end of the picking run.
# MAGIC 
# MAGIC If our goal is to minimize the potential for damage, we might prioritize instead on a metric that encapsulates the general precedence an item should receive in a picking sequence.  Such a metric may be derived using knowledge of a product's weight and volume as well as other attributes that may encourage us to pick an item at the top or bottom of an order. We've captured such a metric in our dataset as a two-decimal place score between 0.00 and 1.00.  We might order our picking list based on this *priority* metric as follows:

# COMMAND ----------

# DBTITLE 1,Define Priority Sorted Routing Function
@f.pandas_udf(MapType(StringType(),ArrayType(IntegerType())))
def get_priority_optimized_products(products_series: pd.Series, zones_series: pd.Series, priorities_series: pd.Series) -> pd.Series:
  
  # combine sequence and zones series to form a new dataframe
  df = pd.DataFrame({'priorities':priorities_series, 'products':products_series, 'zones': zones_series})
  df['results'] = None
  
  # for each order (row) in dataframe
  for i, row in df.iterrows():
    
    # make sure items are sorted by priorities
    sorted_products = [i for _,i in sorted(zip(row['priorities'], row['products']), reverse=True)]
    sorted_zones = [z for _,z in sorted(zip(row['priorities'], row['zones']), reverse=True)]
    
    df.at[i,'results'] = {'products':sorted_products, 'zones':sorted_zones}
    
  return df['results']

# COMMAND ----------

# DBTITLE 1,Calculate Priority-Sorted Product List
orders = (
  orders
    .withColumn('priority_products', get_priority_optimized_products('products','zones','priorities'))
  )

display(orders.select('order_id','products','zones','priority_products'))

# COMMAND ----------

# MAGIC %md ##Step 5: Perform SOP Routing
# MAGIC 
# MAGIC The priority-optimized product list performs a simple sort of products based on their assigned priority values.  With nearly 50,000 products with assigned priorities between 0.01 and 0.99, it's very likely multiple products in a given order may have identical priority values. In that situation, we may wish to consider the sequence within which we pick items with identical priorities in order to minimize traversal time.  Again, we are only making these adjustments for those products with identical priorities so that the overall priority sorting is preserved.
# MAGIC 
# MAGIC This type of optimization is referred to as a sequential ordering problem (SOP) optimization.  There are several algorithms for solving such problems.  We will be using a simulated annealing ant colony optimization algorithm inline with the technique employed by the authors of the paper on which we are basing this work.  To that end, we are borrowing heavily from the code provided by Saleh Afzoon with his permission in his [SOP-Optimization GitHub repository](https://github.com/salehafzoon/SOP-optimization). 
# MAGIC 
# MAGIC More specifically, we are using a modified version of the code provided in the *main.py* file in the repository's *SA* folder.  Because this code was originally written to be called from the Python command line, we've made several modifications to simplify its use within a pandas UDF:
# MAGIC 
# MAGIC **NOTE** You might want to adjust the NUM_TRIALS and NUM_ITERATIONS parameters which control how many optimization attempts are made and how many iterations occur within a single attempt, respectively.  The higher these values go, the more likely you are to arrive at an improved outcome but the longer the optimization will take.

# COMMAND ----------

# DBTITLE 1,Implement SOP Class Based on SA/main.py
from random import randrange
import random as rn
import numpy as np
import math
import copy
import time
import os
import re
import sys

class SOP:

  LINEAR = 'linear'
  LOG = 'logarithmic'
  EXP = 'exponential'

  INIT_HEURISTIC = True
  
  NUM_ITERATIONS = 1000
  NUM_TRIALS = 10
  
  dependencies = []
  EPSILON = 1e-323
  TEMP_MODE = EXP
  graph = None
  START_T = 1
  T = START_T
  ALPHA = 0.9


  class Graph(object):

    class Edge(object):
      def __init__(self, vertices, weight):
          self.vertices = vertices
          self.weight = weight

      def __str__(self):
          return str(self.vertices) + "->" + str(self.weight)
      
      def __repr__(self):
          return str(self)
      
    def __init__(self, problem):
        self.edges = []
        self.dependencies = []
        self.dimension = problem.dimension
        problemEdges = list(problem.get_edges())
        problemWeights = problem.edge_weights[1:]
        for i in range(len(problemEdges)):
            self.edges.append(self.Edge(problemEdges[i], problemWeights[i]))


  def calculateDependencies(self, problem):
    dependencies = []
    edgeWeights = problem.edge_weights[1:]

    for i in range(problem.dimension):
        dependencies.append(list())
        for j in range(self.graph.dimension):
            if(edgeWeights[(i*problem.dimension)+j] == -1):
                dependencies[i].append(j)
    return dependencies


  def fpp3exchange(self, problem, deps, solution):
    dimension = problem.dimension
    edgeWeights = problem.edge_weights[1:]

    solutions = []
    for it in range(int(dimension/2)):
        h = randrange(0, dimension-3)
        i = h + 1
        leftPath = []
        leftPathLen = randrange(1, int(dimension-i))
        leftPath.extend(solution[i:i+leftPathLen])

        i += leftPathLen
        end = False
        rightPath = []
        for j in range(i, len(solution)):

            for dep in deps[solution[j]]:
                if dep != 0 and dep in leftPath:
                    end = True
                    break

            # terminate the progress
            if end:
                break
            # add j to right path
            else:
                rightPath.append(solution[j])

        if (len(rightPath) != 0):
            sol = solution[0:h+1]
            sol.extend(rightPath)
            sol.extend(leftPath)
            sol.extend(solution[len(sol):])
            solutions.append((sol, cost_function(problem, sol)))

    solutions.sort(key=lambda x: x[1])
    if len(solutions) != 0:
        return solutions[0]
    else:
        return None

  def bpp3exchange(self, problem, deps, solution):
    dimension = problem.dimension
    edgeWeights = problem.edge_weights[1:]

    solutions = []
    for it in range(int(dimension/2)):
        h = randrange(3, dimension)
        i = h - 1
        rightPath = []
        rightPathLen = randrange(1, i+1)
        rightPath.extend(solution[i-rightPathLen+1:i+1])
        rightDeps = []

        for node in rightPath:
            rightDeps.extend(deps[node])

        i -= rightPathLen

        leftPath = []
        for j in range(i, 0, -1):

            # add j to left path
            if solution[j] not in rightDeps:
                leftPath.insert(0, solution[j])
            else:
                break

        if (len(leftPath) != 0):
            sol = solution[h:]
            sol = leftPath + sol
            sol = rightPath + sol
            sol = solution[:dimension - len(sol)] + sol
            solutions.append((sol, cost_function(problem, sol)))

    solutions.sort(key=lambda x: x[1])
    if len(solutions) != 0:
        return solutions[0]
    else:
        return None

  def random_start(self, graph, deps):
    solution = []
    dependencies = copy.deepcopy(deps)

    while(len(solution) < graph.dimension):
        for i in range(graph.dimension):
            if(self.INIT_HEURISTIC):
                src = 0
                if len(solution) != 0:
                    src = solution[-1]
                if len(solution) == 7:
                    pass
                candidates = []
                result = [i for i in range(
                    len(dependencies)) if len(dependencies[i]) == 0]
                candidates = [
                    (i, graph.edges[(src*graph.dimension) + i].weight)
                    for i in result if i not in solution]

                candidates = sorted(candidates, key= lambda tup: tup[1])
        
                solution.append(candidates[0][0])

                for dep in dependencies:
                    if(candidates[0][0] in dep):
                        dep.remove(candidates[0][0])

            else:
                if(len(dependencies[i]) == 0 and not(i in solution)):
                    solution.append(i)
                    for dep in dependencies:
                        if(i in dep):
                            dep.remove(i)

    return solution

  def cost_function(self, problem, solution):

    weight = 0
    edgeWeights = problem.edge_weights[1:]
    sol = copy.deepcopy(solution)

    while(len(sol) > 1):
        src = sol.pop(0)
        dest = sol[0]
        w = edgeWeights[(src*problem.dimension)+dest]
        weight += w

    return weight

  def acceptance_probability(self, cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        p = math.exp(- (new_cost - cost) / temperature)
        return p

  def get_neighbour(self, problem, dependencies, state, cost):
    new_states = []
    new_state1 = self.fpp3exchange(problem, dependencies, state)
    new_state2 = self.bpp3exchange(problem, dependencies, state)

    if new_state1 != None:
        new_states.append(new_state1)
    if new_state2 != None:
        new_states.append(new_state2)

    if len(new_states) != 0:
        new_states.sort(key=lambda x: x[1])
        return new_states[0]

    else:
        return (state, cost)

  def updateTemperature(self, step):
    global T
    if self.TEMP_MODE == self.LINEAR:
        return ALPHA * T
    elif self.TEMP_MODE == self.LOG:
        return self.START_T / math.log(step+2)
    elif self.TEMP_MODE == self.EXP:
        return math.exp(-self.ALPHA * step+1)*self.START_T

  def annealing(self, problem, random_start, cost_function, random_neighbour,
              acceptance, updateTemperature, maxsteps=1000, debug=True):

    global T
    EPSILON = 1e-323
    state = random_start(self.graph, self.dependencies)
    cost = self.cost_function(problem, state)
    states, costs = [state], [cost]
    for step in range(maxsteps):
        (new_state, new_cost) = self.get_neighbour(
            problem, self.dependencies, state, cost)
        if debug:
            print('step:', step, '\t T:', T, '\t new_cost:', new_cost)

        if self.acceptance_probability(cost, new_cost, self.T) > rn.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)

        T = self.updateTemperature(step)
        if T == 0.0:
            T = EPSILON

    return new_state, new_cost, states, costs
   
  def run(self, problem):

    # assemble problem graph and dependencies
    self.graph = self.Graph(problem)
    self.dependencies = self.calculateDependencies(problem)
    
    ## initialize answer (do no worse than original sort)
    
    #answer = None # original
    #last_cost = sys.maxsize # original
    
    answer = list(range(problem.dimension)) # initial answer is original sort
    last_cost = self.cost_function(problem, list(range(problem.dimension))) # get cost for this initial answer

    # take best of n runs
    for _ in range(self.NUM_TRIALS):
      
        state, cost, states, costs = self.annealing(
          problem, 
          self.random_start, 
          self.cost_function, 
          self.get_neighbour,
          self.acceptance_probability, 
          self.updateTemperature, 
          self.NUM_ITERATIONS,   
          False
          )
    
        if cost < last_cost: answer = state

    return answer

# COMMAND ----------

# MAGIC %md It's important to note in the SOP optimization logic above we've modified the initial answer given by the algorithm to be the original sort of the items associated with a product. We found that occasionally the algorithm would arrive at an answer that was slightly worse than the original sort if we didn't start with this initial answer. This would likely be resolved by increasing the number of iterations but that would add computational time to our efforts so this compromise was employed.
# MAGIC 
# MAGIC With the SOP optimization logic in place, we now need to define the problem we wish the algorithm to solve. Using the TSPLIB95 format which provides a standard way of articulating various optimization problems, we can express our SOP problem as follows: 

# COMMAND ----------

# DBTITLE 1,Define Functions to Assemble SOP Problem
def get_sop_matrix(zones):
  
  # count zones in list
  n = len(zones) 
  
  # build default matrix of zeros in n x n configuration
  sop_matrix = np.zeros(shape=(n,n)).tolist()
  
  # loop through matrix and populate values:
  # based on instructions in Section 3.4 of http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
  for i, z_row in enumerate(zones):
    for j, z_col in enumerate(zones):
      
      # if first row and last column: set to 10,000
      if i==0 and j==n-1:
        sop_matrix[i][j] = int(10000)
        
      # else if subsequent rows and first column: -1
      elif i > 0 and j == 0:
        sop_matrix[i][j] = int(-1)
   
      # else if last row and now last column: -1
      elif i == n-1 and j != n-1:
        sop_matrix[i][j] = int(-1)
            
      # otherwise 100x travel time between zones
      else:
        sop_matrix[i][j] = int(100*travel_times_bc.value[z_row-1][z_col-1])
        
  # install precedence constraints into matrix
  j = -1
  for i in range(n):
    # if column is not first:
    if j > 0:
      # if row is not last
      if i != n-1: 
        # set value at position to -1
        sop_matrix[j][i] = int(-1)
    
    # capture column index (to move diagonally through matrix starting at position 1)
    j = i
    
  return sop_matrix

def get_sop_problem(zones, sop_matrix):
  
  # format the sop matrix
  sop_matrix_str = str(sop_matrix)                    # convert to string
  sop_matrix_str = sop_matrix_str.replace(',','')     # remove commas                
  sop_matrix_str = sop_matrix_str.replace('[','')     # remove opening square brackets
  sop_matrix_str = sop_matrix_str.replace(']','\n')   # replace closing square brackets with line feeds
  sop_matrix_str = sop_matrix_str.replace('\n ','\n') # replace line feed + space with line feed
  sop_matrix_str = sop_matrix_str[:-2]                # remove trailing two line feed chars
  
  # assemble the problem in the tsplib95 format
  tsplib95_str = f'NAME: order.000.sop\n'
  tsplib95_str += f'TYPE: SOP\n'
  tsplib95_str += f'COMMENT: databricks picking solution accelerator\n'
  tsplib95_str += f'DIMENSION: {len(zones)}\n'
  tsplib95_str += f'EDGE_WEIGHT_TYPE: EXPLICIT\n'
  tsplib95_str += f'EDGE_WEIGHT_FORMAT: FULL_MATRIX\n'
  tsplib95_str += f'EDGE_WEIGHT_SECTION\n'
  tsplib95_str += f'{len(zones)}\n'
  tsplib95_str += f'{sop_matrix_str}\n'
  
  # convert the tsplib95 format into a parsed problem
  problem = tsplib95.utils.load_problem_fromstring(tsplib95_str)
  
  # return problem
  return problem

# COMMAND ----------

# MAGIC %md Please note that in the original SOP implementation referenced above, the TSPLIB95 problem was written to file before being read and parsed in the algorithm. We've modified the code to be able to pass a pre-parsed TSPLIB95 problem object to the function.  The *tsplib95* library used to support this has undergone some changes since when the SOP algorithm was first developed, and we've found the library's current problem parsing functionality is slightly inconsistent with expectations in the algorithm.  With that in mind, we've installed (at the top of this notebook) a prior version of the library to ensure compatibility.
# MAGIC 
# MAGIC With all the logic needed to define and solve a problem in place, we can now construct a pandas UDF to perform the SOP optimization on the priority-sorted picking list:

# COMMAND ----------

# DBTITLE 1,Define SOP Optimization Function
@f.pandas_udf(MapType(StringType(),ArrayType(IntegerType())))
def get_sop_optimized_products(sorted_products_series: pd.Series, sorted_zones_series: pd.Series) -> pd.Series:
   
    answers = []
    
    # instantiate SOP class
    sop = SOP()
    
    # for each row in incoming pd.Series
    for i, zones in enumerate(sorted_zones_series):

      # assemble problem matrix
      sop_matrix = get_sop_matrix(zones)

      # convert matrix into SOP problem per tsplib95 format
      problem = get_sop_problem(zones, sop_matrix)

      # default answers
      zone_answer = []
      product_answer = []
      
      # solve problem to get answer
      if problem.dimension <= 3: # skip optimization if three or less zones to consider 
        raw_answer = list(range(problem.dimension)) # i.e. [0, 1, 2]
      else:
        raw_answer = sop.run(problem)

        
      # map answer to recognized zones
      zone_answer = [zones[a] for a in raw_answer]

      # get associated products
      products = sorted_products_series.iloc[i]

      # map products to zones in answer
      product_answer = []
      zone_product_mapping = list(zip(zones, products))
      # for each zone in answer
      for z in zone_answer:
        # loop through zone-product mapping
        for i,m in enumerate(zone_product_mapping):
          # if zone matches
          if z == m[0]:
            _,product = zone_product_mapping.pop(i)
            product_answer += [product]
            break
      
      answers += [{'products':product_answer, 'zones':zone_answer}]

    return pd.Series(answers)

# COMMAND ----------

# MAGIC %md Applying the algorithm, we get results as follows:

# COMMAND ----------

# DBTITLE 1,Calculate SOP Optimized Product List
orders = orders.withColumn('sop_products', get_sop_optimized_products('priority_products.products','priority_products.zones'))

display(orders.select('order_id','products','zones','sop_products'))

# COMMAND ----------

# MAGIC %md ## Step 6: Perform Relaxed Priority Sorted Routing
# MAGIC 
# MAGIC The ability of the SOP algorithm to identify lower cost routes through the store depends on the existence of multiple items with identical priorities within the same order. While instances of matching priorities do occur, we might increase their frequency by setting a threshold below which items with similar priorities might be considered to have equal priority.
# MAGIC 
# MAGIC We might define a relaxed set of priorities as follows:

# COMMAND ----------

# DBTITLE 1,Define Function to Relaxed Priorities
@f.pandas_udf(ArrayType(FloatType()))
def get_relaxed_priorities(priorities_series: pd.Series, thresholds_series: pd.Series) -> pd.Series:
  
  # combine sequence and zones series to form a new dataframe
  df = pd.DataFrame({'priorities':priorities_series, 'threshold': thresholds_series})
  df['results'] = None
  
  # for each order (row) in dataframe
  for i, row in df.iterrows():   
    
    # get threshold value
    threshold = row['threshold']
    
    # get priorities
    priorities = row['priorities']
    
    # assign index to priority values and sort on priority (highest to lowest)
    sorted_priorities = sorted([[p,x] for x,p in enumerate(priorities)], reverse=True)
    
    # initialize priorities
    adjusted_priorities = sorted_priorities

    # adjust priorities based on threshold
    x = 2 # start with second item in list (x=0 is store entrance)
    while x < len(sorted_priorities)-1:
      # if difference between current priority and prior priority less than threshold
      # NOTE: we are rounding to minimize trivial 
      if round(adjusted_priorities[x-1][0] - sorted_priorities[x][0], 6) <= threshold:
      #  # set priority to same as prior
        adjusted_priorities[x][0] = adjusted_priorities[x-1][0]
      
      # next item
      x += 1
    
    # resort priorities in original index order
    results = [p for x,p in sorted([[x,p] for p,x in adjusted_priorities])]
    
    df.at[i,'results'] = results
    
  return df['results']

# COMMAND ----------

# MAGIC %md While we wouldn't expect a relaxed-priority sorted list to produce different results other than by chance (because we aren't employing any knowledge of traversal times to sort products with the same relaxed-priority values just yet), we might still generate a sorted list for evaluation as follows:

# COMMAND ----------

# DBTITLE 1,Calculate Relaxed Priority Sorted Product List
threshold = 0.25

orders = (
  orders
    .withColumn('relaxed_priorities', get_relaxed_priorities('priorities', f.lit(threshold)))
    .withColumn('relaxed_priority_products', get_priority_optimized_products('products','zones','relaxed_priorities'))
  )

display(orders.select('order_id','products','zones','priorities','relaxed_priorities','relaxed_priority_products'))

# COMMAND ----------

# MAGIC %md ##Step 7: Perform Relaxed SOP Routing
# MAGIC 
# MAGIC Using the relaxed priority values, we might reapproach the SOP optimization as follows:

# COMMAND ----------

# DBTITLE 1,Calculate Relaxed SOP Optimized Product List
orders = orders.withColumn('relaxed_sop_products', get_sop_optimized_products('relaxed_priority_products.products','relaxed_priority_products.zones'))

display(orders.select('order_id','products','zones','relaxed_sop_products'))

# COMMAND ----------

# MAGIC %md ##Step 8: Perform Random Sort
# MAGIC 
# MAGIC Before comparing the various optimizations, it is helpful to establish a baseline.  In many picking scenarios, pickers are handed a list of items to retrieve based either on the order in which the items were placed in the online cart or in an alphabetical or an otherwise seemingly random sort.  
# MAGIC 
# MAGIC We already have a cart-order sorted list of products from earlier in this notebook.  We might generate a random sorted list as follows:

# COMMAND ----------

# DBTITLE 1,Define Function to Perform Random Sort
@f.pandas_udf(MapType(StringType(),ArrayType(IntegerType())))
def get_random_sorted_products(products_series: pd.Series, zones_series: pd.Series) -> pd.Series:
  
  # combine item and zones series to form a new dataframe
  df = pd.DataFrame({'products':products_series, 'zones': zones_series})
  df['results'] = None
  
  # for each order (row) in dataframe
  for i, row in df.iterrows():
    
    # get products and zones
    products = row['products']
    zones = row['zones']
    
    # skip start and finish positions for sorting
    products_to_sort = products[1:-1]
    zones_to_sort = zones[1:-1]
    
    # sort sortable products|zones in random order   
    product_zones = [(p,z) for p, z in zip(products_to_sort, zones_to_sort)]
    random_sorted = [pz for _, pz in sorted([(random(), pz) for pz in product_zones])]
    
    # retrieve sorted products and zones
    sorted_products = [p for p,z in random_sorted]
    sorted_products = [products[0]] + sorted_products + [products[-1]]
    
    sorted_zones = [z for p,z in random_sorted]
    sorted_zones = [zones[0]] + sorted_zones + [zones[-1]]
    
    # set return values
    df.at[i,'results'] = {'products':sorted_products, 'zones':sorted_zones}
    
  return df['results']

# COMMAND ----------

# DBTITLE 1,Calculate Random Sorted Product List
orders = orders.withColumn('random_sorted_products', get_random_sorted_products('products','zones'))

display(orders.select('order_id','products','zones','random_sorted_products'))

# COMMAND ----------

# MAGIC %md ##Step 9: Compare Optimization Approaches
# MAGIC 
# MAGIC With several strategies for sorting the items to be picked, it might be helpful to perform some comparisons.  Without data for the cost (or even the price) of items being ordered, we don't have a way to estimate the cost associated with potentially damaged items.  Instead, we must limit ourselves to an evaluation of the picking times associated with each strategy.
# MAGIC 
# MAGIC Before jumping into such a comparison, it would be helpful to persist the optimized lists.  This will force the Databricks environment to work its way through the entire dataset and give us a quick starting point for any post-optimization analysis:

# COMMAND ----------

# DBTITLE 1,Persist Optimized Picking Data
_ = (
  orders
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('optimized_orders')
)

# COMMAND ----------

# MAGIC %md Now we can write a function to estimate the total time spent in transit between zones for a given order.  As was mentioned earlier, we are assuming the time within a zone is not optimizable and that the wasted time spent moving between zones is what we wish to reduce:

# COMMAND ----------

# DBTITLE 1,Define Function to Calculate Total Travel Time
@f.pandas_udf(FloatType())
def get_travel_seconds(ordered_zones_series: pd.Series) -> pd.Series:
  
  # combine sequence and zones series to form a new dataframe
  df = ordered_zones_series.to_frame(name='zones')
  
  # create a picking time column with a default of 0.0
  df['travel_seconds'] = 0.0
  
  # for each order
  for i, row in df.iterrows():
    
    # initialize picking time
    picking_time = 0.0
    
    # for each zone in sequence
    for j, zone in enumerate(row['zones']):
      
      zone = int(zone)
      
      # adjust zone number for travel times lookup
      zone = zone - 1
      
      # get time to travel between this and last zone and add to total picking time
      if j > 0: picking_time += travel_times_bc.value[last_zone][zone]
      
      # make this zone the new "last zone"
      last_zone = zone
    
    # update picking time for this row
    df.at[i,'travel_seconds'] = picking_time
  
  # return the calculated picking times
  return df['travel_seconds']

# COMMAND ----------

# MAGIC %md And now we can estimate the travel time associated with each optimization strategy:

# COMMAND ----------

# DBTITLE 1,Calculate Average Seconds Per Order Using Different Optimization Strategies
display(
  spark
    .table('optimized_orders')
    .withColumn('default_seconds', get_travel_seconds('zones'))
    .withColumn('random_seconds', get_travel_seconds('random_sorted_products.zones'))
    .withColumn('tsp_seconds',get_travel_seconds('tsp_products.zones'))
    .withColumn('priority_seconds', get_travel_seconds('priority_products.zones'))
    .withColumn('sop_seconds', get_travel_seconds('sop_products.zones'))
    .withColumn('relaxed_priority_seconds', get_travel_seconds('relaxed_priority_products.zones'))
    .withColumn('relaxed_sop_seconds', get_travel_seconds('relaxed_sop_products.zones'))
    .groupBy()
      .agg(
        f.avg('default_seconds').alias('default_seconds'),
        f.avg('random_seconds').alias('random_seconds'),
        f.avg('tsp_seconds').alias('tsp_seconds'),
        f.avg('priority_seconds').alias('priority_seconds'),
        f.avg('sop_seconds').alias('sop_seconds'),
        f.avg('relaxed_priority_seconds').alias('relaxed_priority_seconds'),
        f.avg('relaxed_sop_seconds').alias('relaxed_sop_seconds')
        )
  )

# COMMAND ----------

# MAGIC %md Examining the time it takes to fulfill an average order using the default vs. random sort, it appears customers may be ordering multiple items from a given category before moving on to the next.  This light-weight optimization probably follows a natural tendency to consider similar types of items as they place an order or may reflect department-oriented pages or recommendations that prompt additional purchases of items that would be assigned to the same zone.  As is expected, a randomization of products in the cart results in the worst fulfillment times so that leveraging the *knowledge* of the customer by using their default sort order is better than no optimization at all.
# MAGIC 
# MAGIC Examining the *true* optimizations, the traveling sales person (TSP) optimization provided the best fulfillment times by far.  This is to be expected as this sort order takes into consideration nothing but travel times between zones.  When we introduce the precedence constraints (provided by the *priority* scores), fulfillment times jump up considerably and are only slightly reduced by a sequential order processing (SOP) optimization.  Relaxing the precedence constraints provides the SOP optimization more space within which to identify faster paths to fulfillment though with an increased risk of product damage.  (It should be expected that increasing the threshold value for the relaxation of precedence constraints closer to 1.0 would bring the optimized sort closer to the TSP sort.)
# MAGIC 
# MAGIC **NOTE** With our dataset, the random sort and the priority-based sorting resulted in similar fulfillment times.  This is because the priority scores in this dataset are themselves randomly assigned. Priority scores generated on real-world data may provide different results.
# MAGIC 
# MAGIC For our analysis, there is no one right way to optimize the orders but instead multiple approaches available based on competing priorities of time-reduction and damage prevention.  Incorporation of damage statistics and product cost metrics might help organizations identify an ideal strategy for their needs.

# COMMAND ----------

# MAGIC %md ##Step 10: Next Steps
# MAGIC 
# MAGIC The purpose of this notebook is to help organizations explore patterns for evaluating optimizations applied against large historical datasets of orders. For organizations actively pursuing new and evolving strategies, such an analysis may be periodically re-approached as a batch operation within which new order information and new or altered algorithms would be considered.
# MAGIC 
# MAGIC That said, it's not inconceivable that organizations processing large volumes of orders may wish to apply one or more of these algorithms against incoming orders in order to optimize them before they are delivered to workers in a store.  In such a scenario, a solution would need to be able to solve multiple incoming orders in a timely manner and send optimized results to downstream applications quickly.
# MAGIC 
# MAGIC While the approaches explored in this notebook have centered on batch processing, the pandas UDFs used to perform the optimization work can be employed in a real-time scenario leveraging Databricks' [streaming capabilities](https://docs.databricks.com/spark/latest/structured-streaming/index.html).  In such a solution, newly placed orders might be streamed to a data ingest layer such as Apache Kafka, Azure IOT HUb/Event Hub, AWS Kinesis, GCP Pub-Sub or any number of cloud storage services, read in real-time using Databricks Structured Streaming, transformed using the appropriate algorithm (packaged as a pandas UDF), and then delivered to a downstream data ingest layer accessible to various operational systems.</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/picking_streaming.png' width=700>
# MAGIC   
# MAGIC The beauty of such an approach is that algorithms vetted against historical data could be rapidly deployed into a real-time optimization pipeline as the mechanics of processing real-time data in Databricks closely mirror those of processing batch data.  As the organization experiences variable workloads, the Databricks environment could be scaled up and down to keep pace with demand for the services, keeping the cost of optimizing the orders to a minimum.

# COMMAND ----------

# MAGIC %md © 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | ortools                                  | Google OR-Tools python libraries and modules | Apache 2.0    | https://pypi.org/project/ortools/                       |
# MAGIC | tsplib95 | TSPLIB95 works with TSPLIB95 files | Apache 2.0 | https://pypi.org/project/tsplib95/ |
# MAGIC | SOP-Optimization |Solving SOP with SA, GRASP, Tabu Search algorithms |  | https://github.com/salehafzoon/SOP-optimization |
