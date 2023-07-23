from kafka import KafkaConsumer
import json
from pymongo import MongoClient
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['BDA']
collection = db['recommendations']

# Create Kafka consumer
consumer = KafkaConsumer(
    'recommendation_topic',
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='latest',
     value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Continuously poll for new messages
for message in consumer:
    # Process the message
    rec = message.value
    asin = rec['asin']
    ratings = rec['overall']
    for i in range(10):
        collection.insert_one({asin[i]:ratings[i]})
        print({asin[i]:ratings[i]})
    #products.append(message.value)

#collection.insert_one(message.value)


