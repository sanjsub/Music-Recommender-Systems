import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import getpass
from pyspark.sql.types import LongType

'''
## OLD VERSION OF PREP DATASET

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]


def main(spark, input_file_path, output_file_path):

    df = spark.read.parquet(input_file_path)
    df = df.withColumn("user_id_hash", F.hash(F.col("user_id")))
    df = df.withColumn("track_id_hash", F.hash(F.col("track_id")))
    df.write.mode('overwrite').parquet(output_file_path)
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('prep_dataset').getOrCreate()
    main(spark, input_file_path, output_file_path)
'''

train_path = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
val_path = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
test_path = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'
data_paths = [train_path, val_path, test_path]
file_names = ['cf_train', 'cf_validation', 'cf_test']

def deterministic_hash_map(string):
    '''
    Modified to make hash extremely unlikely, not a 1-to-1
    '''
    hashed_string_first = ''
    hashed_string_second = ''
    first_string_half = string[:9]
    second_string_half = string[9:]
    
    for char in first_string_half:
        hashed_string_first += str(ord(char))
    
    for char in second_string_half:
        hashed_string_second += str(ord(char))
    
    return int(hashed_string_first) + int(hashed_string_second)

def prep(spark, input_file_path, outfile_path, count):

    df = spark.read.parquet(input_file_path)
    df.createOrReplaceTempView('df')
    
    mapped_rdd = df.rdd.map(lambda x: (x[0], x[1], x[2], x[3], 
                                       deterministic_hash_map(x[0]),
                                       deterministic_hash_map(x[2])))

    mapped_df = mapped_rdd.toDF(['user_id', 'count', 'track_id', 
                                 '__index_level_0__', 'user_id_hash', 'track_id_hash'])

    final_df = mapped_df.withColumn("user_id_hash",F.col("user_id_hash").cast(LongType()))\
                        .withColumn("track_id_hash",F.col("track_id_hash").cast(LongType()))
    
    #mapped_df.select(F.col("user_id_hash").cast('long').alias("user_id_hash"))
    #mapped_df.select(F.col("track_id_hash").cast('long').alias("track_id_hash"))
    
    #mapped_df.write.csv(f'{count}.csv')

    #my_schema = ('user_id STRING, count INT, track_id STRING, ' 
    #             '__index_level_0__ LONG, user_id_hash LONG, track_id_hash LONG')
    #final_df = spark.read.csv(f'{count}.csv', schema=my_schema)

    
    
    print("printing dataframes")
    final_df.printSchema()
    final_df.show()
    #mapped_df.printSchema()
    #mapped_df.show()
    #df.write.mode('overwrite').parquet(output_file_path)

def main():
    
    count = 0
    netID = getpass.getuser()

    spark = SparkSession.builder.appName('prep_dataset').getOrCreate()
    
    for idx, data_path in enumerate(data_paths):
        file_name = file_names[idx]
        outfile_path = f"hdfs:/user/{netID}/music/processed_files/{file_name}.parquet"
        prep(spark, data_path, outfile_path, count)
        count += 1


if __name__ == "__main__":

    # Create the spark session object
    main()