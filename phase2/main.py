from flask import Flask, render_template, request, redirect, url_for
from kafka import KafkaProducer
import tensorflow as tf
import pymongo
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
# producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
#                          value_serializer=lambda x:
#                          json.dumps(x).encode('utf-8'))

# Check connection to the database
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    client.server_info()
    print("Connected to MongoDB")
except pymongo.errors.ServerSelectionTimeoutError as err:
    print("Error connecting to MongoDB:", err)
db = client['BDA']
collection = db['recomendations']


data = pd.read_json('sampled_reviews.json', orient='records')
data.drop('_id', inplace=True, axis=1)
df = data[['asin', 'reviewerID', 'overall']]

# encoding UserID and ProductID to simple integers to improve computation effeciency
# Maintaing a map to get back the decoded UserID and ProductID after the calculations .

user_ids = df["reviewerID"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

product_ids = df["asin"].unique().tolist()
product2product_encoded = {x: i for i, x in enumerate(product_ids)}
product_encoded2product = {i: x for i, x in enumerate(product_ids)}

df["reviewerID"] = df["reviewerID"].map(user2user_encoded)
df["asin"] = df["asin"].map(product2product_encoded)

num_users = len(user2user_encoded)
num_product = len(product_encoded2product)
df['overall'] = df['overall'].values.astype(np.float32)

min_rating = min(df['overall'])
max_rating = max(df['overall'])

df = df.sample(frac=1, random_state=42)
x = df[["reviewerID", "asin"]].values

y = df["overall"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.7 * df.shape[0])
val_indices = int(0.9 * df.shape[0])

x_train, x_val, x_test , y_train, y_val , y_test = (
    x[:train_indices],
    x[train_indices:val_indices],
    x[val_indices : ] ,
    y[:train_indices],
    y[train_indices:val_indices],
    y[val_indices : ]
)

model = tf.keras.models.load_model("model")
EMBEDDING_SIZE = 40


class Recommender(keras.Model):
    def __init__(self, num_users, num_product, embedding_size):
        super(Recommender, self).__init__()
        self.num_users = num_users
        self.num_product = num_product
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.product_embedding = layers.Embedding(
            num_product,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.product_bias = layers.Embedding(num_product, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        product_vector = self.product_embedding(inputs[:, 1])

        user_bias = self.user_bias(inputs[:, 0])
        product_bias = self.product_bias(inputs[:, 1])

        dot_prod = tf.tensordot(user_vector, product_vector, 2)

        x = dot_prod + user_bias + product_bias

        return tf.nn.sigmoid(x)

    def getRecomendation(self, df, encoded_user, k):
        # key = list(filter(lambda x: user2user_encoded[x] == 100, user2user_encoded))[0]
        # encoded_user = user2user_encoded[key]

        all_prods = df['asin'].unique()
        prods = df[df.reviewerID == encoded_user]['asin'].values
        remainder = list(set(all_prods) - set(prods))
        n = len(remainder)
        out = np.empty((n, 2), dtype=int)
        out[:, 0] = encoded_user
        out[:, 1] = remainder[:None]
        output = self.predict(out)

        ndx = list(map(lambda x: product_encoded2product[x], remainder))
        vals = output[:, 0].tolist()

        return pd.Series(index=ndx, data=vals).sort_values(ascending=False)[:k].index


model = Recommender(num_users, num_product, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001)
)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/recommend')
def recommend():
    return render_template("recommend.html")


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    u = request.form['user_id']
    K = 10  # top k items
    top_10_prod = model.getRecomendation(df, user2user_encoded[u], K)

    # Insert recommendations into MongoDB
    recommendations = []
    for asin in top_10_prod:
        if asin not in product_encoded2product:
            continue  # Skip this product if it doesn't exist in the dictionary

        recommendation = {
            "asin": product_encoded2product[asin],
            "overall": max_rating
        }
        recommendations.append(recommendation)

    if recommendations:
        collection.insert_many(recommendations)

    # Create a new Kafka producer for each request
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                             value_serializer=lambda x: json.dumps(x).encode('utf-8'))

    # send recommendations to Kafka
    x = data[data['asin'].isin(top_10_prod)]
    x = x[['asin', 'overall']]

    producer.send('recommendation_topic', value=(x.to_dict(orient='list')))

    # Flush the messages to the Kafka broker
    producer.flush()
    # Close the producer connection
    producer.close()

    # read from Mongo and show on webpage
    rec = []
    for document in collection.find():
        rec.append((list(document.keys()), list(document.values())))

    # Retrieve recommendations from MongoDB
    recommendations = list(collection.find())
    print(recommendations)

    return render_template('recommend.html', recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)