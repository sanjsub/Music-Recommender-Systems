# -*- coding: utf-8 -*-
"""
@author: sanjs
"""

# Import command line arguments and helper functions
import sys
import getpass
from datetime import datetime
import csv

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics


'''
On Peel's HDFS, you will find the following files in `hdfs:/user/bm106/pub/MSD`:

  - `cf_train.parquet`
  - `cf_validation.parquet`
  - `cf_test.parquet`

These files are also available under `/scratch/work/public/MillionSongDataset/` if you want to access them from outside of HDFS
'''


def popularity_baseline(spark, val, test, netID):
    
    test_user = test.select('user_id_index').distinct()
    y_true = test.select('user_id_index', 'track_id_index')\
                    .groupBy('user_id_index')\
                    .agg(F.expr('collect_list(track_id_index) as true_id'))
    
    # to run SQL commands
    val.createOrReplaceTempView('val')
    y_baseline = spark.sql('SELECT track_id_index, SUM(count) FROM val \
                           GROUP BY track_id_index ORDER BY SUM(count) DESC LIMIT 500')
    print('y_baseline SQL query')
    y_baseline.show(5)
                          
    
    #y_baseline = val.select('track_id_index', 'play_count').groupBy('track_id_index')\
    #    .agg(F.sum('count')).sort(F.desc('play_count')).head(500)
    
    # import model of choice here to get recs
    #path = f'hdfs:/user/{netID}/Model_ALS_{netID}_2021-5-15_005923027030'
    #model = ALSModel.load(path)
    
    #res = model.recommendForUserSubset(val_user, 500)
    #y_pred = res.select('user_id_index','recommendations.track_id_index')
    
    #joined = y_true.join(y_pred, y_true.user_id_index == y_pred.user_id_index, "inner")
    #rdd = joined.rdd.map(lambda x: (x[1],x[3]))
    
    length = test_user.count()
    base_list = [y_baseline.select('track_id_index').rdd.flatMap(lambda x: x).collect()]*length
    test_list = test_user.select('user_id_index').rdd.flatMap(lambda x: x).collect()
    
    baseline = spark.createDataFrame(zip(test_list, base_list), schema=['user_id_index', 'track_id_index'])
    joined_base = y_true.join(baseline, y_true.user_id_index == baseline.user_id_index, "inner")
    print('y_baseline merge with y_true')
    joined_base.show(5)
    rdd_baseline = joined_base.rdd.map(lambda x: (x[1],x[3]))
    
    
    #print("Action: First element: "+str(rdd_baseline.first()))
    
    #metrics = RankingMetrics(rdd)
    #meanap = metrics.meanAveragePrecision
    #ndcg = metrics.ndcgAt(500)
    #patk = metrics.precisionAt(500)
    
    metrics_base = RankingMetrics(rdd_baseline)
    meanap_base = metrics_base.meanAveragePrecision
    ndcg_base = metrics_base.ndcgAt(500)
    patk_base = metrics_base.precisionAt(500)
    
    #print(f'map: {meanap}, ndcg: {ndcg}, PatK: {patk}')
    print(f'base map: {meanap_base}, base ndcg: {ndcg_base}, base PatK: {patk_base}')
    
    
    # for avg
    y_baseline_avg = spark.sql('SELECT track_id_index, SUM(count) / (COUNT(track_id_index) + 100) \
                               FROM val GROUP BY track_id_index ORDER BY SUM(count) / \
                                   (COUNT(track_id_index) + 100) DESC LIMIT 500')
    y_baseline_avg.show(5)
    print('y_baseline_avg SQL query')
    
    base_list_avg = [y_baseline_avg.select('track_id_index').rdd.flatMap(lambda x: x).collect()]*length
    baseline_avg = spark.createDataFrame(zip(test_list, base_list_avg), schema=['user_id_index', 'track_id_index'])
    joined_base_avg = y_true.join(baseline_avg, y_true.user_id_index == baseline_avg.user_id_index, "inner")
    joined_base_avg.show(5)
    rdd_baseline_avg = joined_base_avg.rdd.map(lambda x: (x[1],x[3]))
    print("Action avg: First element: "+str(rdd_baseline_avg.first()))
    
    metrics_base_avg = RankingMetrics(rdd_baseline_avg)
    meanap_base_avg = metrics_base_avg.meanAveragePrecision
    ndcg_base_avg = metrics_base_avg.ndcgAt(500)
    patk_base_avg = metrics_base_avg.precisionAt(500)
    print(f'base map avg: {meanap_base_avg}, base ndcg avg: {ndcg_base_avg}, base PatK avg: {patk_base_avg}')
    
    pass
    
    
def main(spark, netID):
    train_path = f'hdfs:/user/{netID}/music/processed_files/cf_train.parquet'
    #val_path = f'hdfs:/user/{netID}/music/processed_files/cf_validation.parquet'
    test_path = f'hdfs:/user/{netID}/music/processed_files/cf_test.parquet'
    
    # train = spark.read.parquet(train_path)
    val = spark.read.parquet(train_path)
    test = spark.read.parquet(test_path)
    

    popularity_baseline(spark, val, test, netID)
    


# Only enter this block if we're in main
if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.appName('extension_baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)

