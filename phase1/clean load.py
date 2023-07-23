import nltk
nltk.download('wordnet')
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType, DoubleType
import os
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from autocorrect import spell
import pandas as pd
import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import mean, round, col, concat_ws, when, udf, StringType, lit


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
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .getOrCreate()

print("Spark session initialized.")

# Read the JSON file into a DataFrame
file_path = r"D:\reviews.json\All_Amazon_Review.json"
print("Loading JSON file into DataFrame...")
df = spark.read.json(file_path, schema=amazon_schema)
print("JSON file loaded into DataFrame.")

# Drop missing or malformed data
df = df.dropna()
df = df.dropDuplicates()
# Filter out any rows with "verified" value of False
df = df.filter(df.verified == True)


def clean(df):
    df = df.withColumn("reviewText", concat_ws(" ", col("reviewText"), col("summary"))).drop("summary")
    df = df.withColumn("feedback", when(col("overall") >= 3, "good").otherwise("bad"))
    df = df.dropDuplicates(subset=['asin', 'reviewerID', 'unixReviewTime'])
    return df


def handle_vote_column(df):
    vote_mean = df.select(round(mean(col("vote")), 2).alias("vote_mean")).collect()[0]["vote_mean"]
    df = df.withColumn('vote', when(col('vote').isNull(), lit(vote_mean)).otherwise(col('vote')))
    return df


def autospell(text):
    # correct the spelling of the word.

    spells = [spell(w) for w in (nltk.word_tokenize(text))]
    return " ".join(spells)


def to_lower(text):
    # Returns converted text to lower case as in, converting "Hello" to "hello" or "HELLO" to "hello".
    return text.lower()


df = clean(df)
df = handle_vote_column(df)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")  # or any other language you need


def preprocess(text):
    text = text.lower()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


preprocess_udf = udf(preprocess, StringType())
df = df.withColumn("reviewText", preprocess_udf(col("reviewText")))

# Filter out any rows with "verified" value of False
df = df.filter(df.verified == True)

# Check for any missing or malformed data after handling
df_err = df.filter(df["asin"].isNull() | df["overall"].isNull() | df["reviewText"].isNull() | df["reviewerID"].isNull() | df["unixReviewTime"].isNull() | df["vote"].isNull())

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
    .option("spark.mongodb.output.createCollectionOptions", '{"validator": {"$jsonSchema": { "bsonType": "object", "required": ["asin", "overall", "reviewText", "reviewerID", "summary", "unixReviewTime"], "properties": { "asin": { "bsonType": "string" }, "overall": { "bsonType": "float" }, "reviewText": { "bsonType": "string" }, "reviewerID": { "bsonType": "string" }, "summary": { "bsonType": "string" }, "unixReviewTime": { "bsonType": "int" } } } }, "validationLevel": "moderate", "collation": {"locale": "en", "strength": 2}}}')\
    .option("spark.mongodb.output.replaceDocument", "false")\
    .option("spark.mongodb.output.bulk.ordered", "false")\
    .option("spark.mongodb.output.createIndexes", '[{"key": {"reviewerID": 1}, "name": "reviewerID_index"}]') \
    .save()


print(f"Loaded {df.count()} records from {file_path} into MongoDB.")
print("Indexes created.")

# Stop the Spark session
spark.stop()

print("Spark session stopped.")
