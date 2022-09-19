## Introduction

With more and more stores serving as local, rapid fulfillment centers for online orders, there is growing interest in maximizing the efficiency of workers traversing a store's aisles to pick items. High labor costs mean the more time workers spend moving from aisle to aisle, the harder it is for retailers to make buy online, pickup in-store (BOPIS) and other omnichannel fulfillment services profitable. 

The principal challenge in improving efficiency is that most stores have been purposefully laid out to be inefficient.  Having a customer move from one end of the store to the other in order to pick the items they require increases their exposure to other goods and thereby increases the chances they will leave the store having purchased more items than originally intended.  Given the store isn't paying for the customer's time, the cost of this inefficiency (outside of extreme circumstances that simply discourage a customer to shop at a given location) is negligible so that there is only upside to this configuration. But that's not the case when retailers are paying for the time of workers who lack the ability to add impulse items to someone else's order. 

In this scenario, it's critical that stores find ways to improve the speed with which a worker can pick the items associated with an order.  Barring changes in store configurations, this is best accomplished through the optimized sorting of items to be picked. Such an optimization must consider both the time it takes to move through the store and the fragility of the items being picked.  If a fragile item is damaged during picking, not only is the retailer responsible for the cost of the item, the worker may need to make a return visit to a location within the store to pick a replacement.

This topic of in-store picking optimization was recently addressed by the authors of a paper titled, *[The Buy-Online-Pick-Up-in-Store Retailing Model: Optimization Strategies for In-Store Picking and Packing](https://www.mdpi.com/1999-4893/14/12/350/pdf)*. In the paper, the authors apply various optimizations to the sequencing of items picked to fulfill a set of online orders placed with a store having the following physical layout:</p>

<img src='https://brysmiwasb.blob.core.windows.net/demos/images/picking_store_layout.png' width=600>

Their optimization strategies explored both the sequence within which items were picked and the opportunity for order bundling. Lacking the data needed to address the bundling aspects of their work, we've elected to explore the order picking problem using techniques identified in their paper.  As we don't have access to the order information used by the authors, we will use the orders in the [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis), made available as part of a Kaggle competition, as examples of orders we might wish to optimize.  This dataset is particularly interesting given the sequence in which items are placed in a shopping cart is preserved in the data, the same sequencing identified as the default sequencing approach for the real-world orders explored by the paper's authors.  In order to align the data with the store layout as presented above, we've had to arbitrarily assign products to the 15 zones in the presented store. Our results show similar benefits to optimization as those demonstrated by the paper's authors.

To move this solution accelerator beyond an academic evaluation of the benefits of optimization, we conclude our work by proposing a simple architecture within which newly arriving orders can be rapidly optimized.  We do this as a streaming scenario so that orders may be processed quickly in a manner aligned with narrowing fulfillment windows most retailers are pursuing for these kinds of orders.  We use Databricks to scale out this work so that large volumes of orders submitted across multiple stores may be tackled in a parallel manner.

These various aspects of the solution accelerator are tackled in the following notebooks:</p>

* PK 01: Data Preparation
* PK 02: Order Optimizations

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| ortools                                  | Google OR-Tools python libraries and modules | Apache 2.0    | https://pypi.org/project/ortools/                       |
| tsplib95 | TSPLIB95 works with TSPLIB95 files | Apache 2.0 | https://pypi.org/project/tsplib95/ |
| SOP-Optimization |Solving SOP with SA, GRASP, Tabu Search algorithms | | https://github.com/salehafzoon/SOP-optimization |

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
