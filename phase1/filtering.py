from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, length
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 pyspark-shell'

spark = SparkSession.builder.appName("Filter Amazon Reviews").config(
    "spark.mongodb.input.uri", "mongodb://127.0.0.1/amazon.reviews"
).config(
    "spark.mongodb.output.uri", "mongodb://127.0.0.1/amazon.filtered_data_forCB"
).getOrCreate()

# read data from MongoDB
df = spark.read.format("com.mongodb.spark.sql.DefaultSource").option(
    "collection", "reviews"
).load()

print("Reducing data to 2013 onwards")
# Filter data to include only reviews from 2013 onwards
df = df.filter(df.unixReviewTime >= 1356998400)
# Total records count: 16906631
# Filter reviews by length (at least 5 characters)
df = df.filter(length(df.reviewText) >= 5)
# Total records count after removing short reviews: 16905527
#  104 (The number of records removed by filtering out short reviews)

print("Filtering out products with few ratings")
# Filter out products with very few ratings
min_ratings = 3
product_ratings = df.groupBy("asin").agg(count("overall").alias("num_ratings"))
filtered_products = product_ratings.filter(col("num_ratings") >= min_ratings)

# Join the filtered products with the original dataframe
filtered_df = df.join(filtered_products, "asin", "inner")

print("Filtering out users with few reviews")
# Limit the number of users based on a minimum threshold of reviews
min_user_reviews = 3
user_review_counts = filtered_df.groupBy("reviewerID").count().withColumnRenamed("count", "review_count")
filtered_users = user_review_counts.filter((user_review_counts.review_count >= min_user_reviews))

# Join the filtered users with the filtered dataframe
filtered_df = filtered_df.join(filtered_users, ["reviewerID"])

# Unique ASIN count: 770161
# Unique reveiwerIDs count: 1061371

# Select only the relevant columns
filtered_data = filtered_df.select("reviewerID", "asin", "overall", "reviewText")
print("Final df selected")

print("Writing to MongoDB...")
# Write the filtered data to MongoDB and create indexes on the "reviewerID", "asin", and "overall" fields
filtered_data.write\
    .format("mongo")\
    .mode("overwrite")\
    .option("uri", "mongodb://127.0.0.1/amazon.filtered_data_forCB")\
    .option("collection", "filtered_data_forCB")\
    .option("spark.mongodb.output.createCollectionOptions", '{"validator": {"$jsonSchema": { "bsonType": "object", "required": ["asin", "overall", "reviewerID"], "properties": { "asin": { "bsonType": "string" }, "overall": { "bsonType": "float" }, "reviewerID": { "bsonType": "string" } } } }, "validationLevel": "moderate"} }')\
    .option("spark.mongodb.output.replaceDocument", "false")\
    .option("spark.mongodb.output.bulk.ordered", "false")\
    .save()


# Stop the Spark session
spark.stop()
