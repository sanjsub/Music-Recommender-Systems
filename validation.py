import sys
import statistics

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

validation_file_path = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
model_path = "models/als_regparam_1.0_rank_10_alpha_1.0.model"




def validation(spark, validation_file_path, model_path):
	val_df = spark.read.parquet(validation_file_path)
	model = ALS.load(model_path)
	
	val_df.show()

	pred_df = ALS.transform(val_df)

	pred_df.show()

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline_model').getOrCreate()


    ## Replacing file path with test_file_path
    validation(spark, validation_file_path, model_path)
