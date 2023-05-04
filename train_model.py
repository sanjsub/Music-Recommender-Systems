import argparse, sys
from datetime import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

"""
Sample Usage

spark-submit --driver-memory=8g --executor-memory=8g train_model.py --TrainParquetPath="hdfs:/user/sa6149/music/processed_files/cf_train.parquet" --Implicit=True --UserCol="user_id_hash" --ItemCol="track_id_hash" --RatingCol="count" --MaxIter=5 --RegParam="1.0" --ColdStartStrategy="drop" --Rank=10 --NonNegative=True --Alpha="1.0" 
--Seed="42"

"""


parser=argparse.ArgumentParser()

parser.add_argument('--TrainParquetPath', help='Training DataFrame')
parser.add_argument('--Implicit', help='ALS Implicit', type=bool)
parser.add_argument('--UserCol', help='UserIDColumn Hashed')
parser.add_argument('--ItemCol', help='TrackIDColumn Hashed')
parser.add_argument('--RatingCol', help='Number of Times Played')
parser.add_argument('--MaxIter', help='Maximum Iteration ALS Parameter')
parser.add_argument('--RegParam', help='Regularization Parameter ALS Parameter')
parser.add_argument('--ColdStartStrategy', help='ColdStartStrategy ALS Parameter')
parser.add_argument('--Rank', help='Rank ALS Parameter')
parser.add_argument('--NonNegative', help='NonNegative ALS Parameter', type=bool)
parser.add_argument('--Alpha', help='Alpha ALS Parameter')
parser.add_argument('--Seed', help="Random Seed")

args = parser.parse_args()

model_save_path = "als_regparam_" + args.RegParam + "_rank_" + args.Rank + "_alpha_" + args.Alpha + ".model"


def fit_als(spark, train_parquet_path, implicit, user_col, item_col, rating_col, max_iter, reg_param, cold_start_strategy, rank, non_negative, alpha, seed, model_path):
    a = datetime.now()
    train_df = spark.read.parquet(train_parquet_path)
    b = datetime.now()
    print("Dataset Loaded")
    print("Reading Time : {}".format(b-a))
    a = datetime.now()
    als = ALS(implicitPrefs=implicit, maxIter=int(max_iter), regParam=float(reg_param), userCol=user_col, itemCol=item_col, ratingCol=rating_col, coldStartStrategy=cold_start_strategy, rank = int(rank), nonnegative = non_negative, alpha = float(alpha), seed = int(seed))
    model = als.fit(train_df) 
    als.save(model_save_path)
    b = datetime.now()
    print("Model Fit")
    print("Model Fit Time : {}".format(b-a))
     


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('fit_model').getOrCreate()

    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    ## Replacing file path with test_file_path
    fit_als(spark, train_parquet_path = args.TrainParquetPath, implicit = args.Implicit, user_col = args.UserCol, item_col = args.ItemCol, rating_col = args.RatingCol, max_iter = args.MaxIter,reg_param = args.RegParam, cold_start_strategy = args.ColdStartStrategy, rank = args.Rank, non_negative = args.NonNegative, alpha = args.Alpha, seed = args.Seed, model_path = model_save_path)


