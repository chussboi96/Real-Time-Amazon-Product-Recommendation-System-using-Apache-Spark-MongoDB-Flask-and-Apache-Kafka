from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType, DoubleType
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 pyspark-shell'

# schema for the dataset
amazon_schema = StructType([
    StructField("asin", StringType(), True),
    StructField("overall", DoubleType(), True),
    StructField("verified", BooleanType(), True),
    StructField("reviewText", StringType(), True),
    StructField("reviewerID", StringType(), True),
    StructField("summary", StringType(), True),
    StructField("unixReviewTime", IntegerType(), True),
    StructField("vote", StringType(), True)  # Adding the vote field
])

# Spark session
spark = SparkSession \
    .builder \
    .appName("Amazon Reviews Loader") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/amazon.reviews") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/amazon.reviews") \
    .getOrCreate()

print("Spark session initialized.")

# Read the JSON file into a DataFrame
file_path = r"D:\reviews.json\All_Amazon_Review.json"
print("Loading JSON file into DataFrame...")
df = spark.read.json(file_path, schema=amazon_schema)
print("JSON file loaded into DataFrame.")

# Drop missing or malformed data
df = df.dropna()

if df.count() > 0:
    print(f"Found {df.count()} records with missing or malformed data.")
    print("Sample dataset records before handling:")
    df.show(5)

# Drop duplicate values
df = df.dropDuplicates()

# Filter out any rows with "verified" value of False
# df = df.filter(df.verified == True)

# Cache the cleaned DataFrame on disk
# df = df.persist(StorageLevel.DISK_ONLY)
# df.persist(StorageLevel.DISK_ONLY) writes cache to disk instead of memory

# Check for any missing or malformed data after handling
df_err = df.filter(df["asin"].isNull() | df["overall"].isNull() | df["reviewText"].isNull() | df["reviewerID"].isNull() | df["summary"].isNull() | df["unixReviewTime"].isNull() | df["vote"].isNull())

if df_err.count() > 0:
    print(f"Found {df_err.count()} records with missing or malformed data after handling.")
    print("Sample dataset records after handling:")
    df_err.show(10)
else:
    print("No missing or malformed data found after handling.")


print("First 10 rows of dataset after cleaning:")
df.show(10)

num_partitions = 10
df = df.repartition(num_partitions)

# Create indexes
print("Writing data to MongoDB...")
df.write\
    .format("mongo")\
    .mode("overwrite")\
    .option("uri", "mongodb://127.0.0.1/amazon.reviews")\
    .option("collection", "reviews") \
    .option("spark.mongodb.output.batchSize", 500) \
    .option("spark.mongodb.output.createCollectionOptions", '{"validator": {"$jsonSchema": { "bsonType": "object", "required": ["asin", "overall", "reviewText", "reviewerID", "summary", "unixReviewTime"], "properties": { "asin": { "bsonType": "string" }, "overall": { "bsonType": "float" }, "reviewText": { "bsonType": "string" }, "reviewerID": { "bsonType": "string" }, "summary": { "bsonType": "string" }, "unixReviewTime": { "bsonType": "int" } } } }, "validationLevel": "moderate"} }')\
    .option("spark.mongodb.output.replaceDocument", "false")\
    .option("spark.mongodb.output.bulk.ordered", "false")\
    .save()

print(f"Loaded {df.count()} records from {file_path} into MongoDB.")
print("Indexes created.")

# Stop the Spark session
spark.stop()

print("Spark session stopped.")
