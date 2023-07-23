from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder


os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 pyspark-shell'

# Spark session
spark = SparkSession \
    .builder \
    .appName("Amazon Reviews EDA") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/amazon.reviews") \
    .getOrCreate()

# Read data from MongoDB into a DataFrame
pandas_df = (
    spark.read.format("mongo")
    .option("uri", "mongodb://127.0.0.1/amazon.reviews")
    .load()
).toPandas()


# Display summary statistics
print(pandas_df.describe())
# Display column data types and non-null values
print(pandas_df.info())


# Define a function to preprocess the review text
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove digits
    text = re.sub('\d+', '', text)
    # Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text into words
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    text = ' '.join(words)
    return text


# Apply the preprocess_text function to the reviewText column
pandas_df['clean_review_text'] = pandas_df['reviewText'].apply(preprocess_text)

# Create a list of all the words in reviews
all_words = " ".join(pandas_df['clean_review_text']).split()
# # Create a frequency distribution of bigrams
# bigram_measures = BigramAssocMeasures()
# finder = BigramCollocationFinder.from_words(all_words)
# finder.apply_freq_filter(10) # filter out bigrams that occur less than 10 times
# bigram_freq_dist = finder.ngram_fd
# plt.rcParams['text.color'] = 'black'
# plt.rcParams['axes.labelcolor'] = 'black'
# plt.rcParams['xtick.color'] = 'black'
# plt.rcParams['ytick.color'] = 'black'
# # Get the top 25 bigrams by frequency
# top_25_bigrams = bigram_freq_dist.most_common(25)
# plt.figure(figsize=(12, 6))
# x_values = [x[0] for x in top_25_bigrams]
# y_values = [x[1] for x in top_25_bigrams]
# plt.bar(x_values, y_values)
# plt.xticks(rotation=90)
# plt.xlabel('Bigram Words')
# plt.ylabel('Frequency')
# plt.title('Top 25 Frequently Used Bigram Words')
# plt.tight_layout()
#
# for i, v in enumerate(y_values):
#     plt.text(i, v+100, str(v), ha='center', fontweight='bold', fontsize=8)
#
# plt.show()


# Calculate the length of each review
pandas_df['review_length'] = pandas_df['clean_review_text'].apply(lambda x: len(x.split()))
# Plot a histogram of the review length distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=pandas_df, x='review_length', kde=True)
plt.title("Distribution of Review Length")
plt.show()


# Plot a histogram of the helpful votes distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=pandas_df, x='vote', kde=True)
plt.title("Distribution of Helpful Votes")
plt.show()


# Create a list of all the words in reviews
all_words = " ".join(pandas_df['clean_review_text']).split()
# Create a frequency distribution of words for each rating level
freq_dist_by_rating = {}
for rating in range(1, 6):
    words = " ".join(pandas_df[pandas_df['overall'] == rating]['clean_review_text']).split()
    freq_dist_by_rating[rating] = nltk.FreqDist(words)

# Plot the frequency distribution for each rating level
plt.figure(figsize=(10, 10))
for rating in range(1, 6):
    plt.subplot(3, 2, rating)
    freq_dist_by_rating[rating].plot(20, title=f"Rating {rating}")
plt.tight_layout()
plt.show()


# Distribution of false ratings/reviews
sns.countplot(x='verified', data=pandas_df)
plt.title('Distribution of Verified Reviews')
plt.xlabel('Verified')
plt.ylabel('Count')
plt.show()


# Plot average rating distribution
avg_rating = round(pandas_df['overall'].mean(), 2)
plt.figure(figsize=(6, 6))
sns.histplot(x='overall', data=pandas_df, bins=10)
plt.axvline(x=avg_rating, color='red', linestyle='--', label='Average Rating ({})'.format(avg_rating))
plt.legend()
plt.title('Overall Rating Distribution')
plt.xlabel('Overall Rating')
plt.ylabel('Number of Reviews')
plt.show()

# top 10 most reviews products
top_10_products = pandas_df['asin'].value_counts().nlargest(10)
sns.barplot(x=top_10_products.index, y=top_10_products.values)
plt.title('Top 10 Most Reviewed Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

# Sort by overall rating and get the top and bottom 10 products
best_products = pandas_df.sort_values(by='overall', ascending=False).head(10)
worst_products = pandas_df.sort_values(by='overall', ascending=True).head(10)
# best rated
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='overall', y='asin', data=best_products)
plt.title('Top 10 Best Rated Products')
plt.xlabel('Overall Rating')
plt.ylabel('Product ID')
# worst rated
plt.subplot(1, 2, 2)
sns.barplot(x='overall', y='asin', data=worst_products)
plt.title('Top 10 Worst Rated Products')
plt.xlabel('Overall Rating')
plt.ylabel('Product ID')
plt.tight_layout()
plt.show()


# Combine all the review text into a single string
reviews_text = ' '.join(pandas_df['reviewText'])
# Create a list of stop words to exclude from the word cloud
stop_words = set(stopwords.words('english'))


# Define color function for WordCloud
def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "rgb(0, 0, 0)"


# Generate the WordCloud
wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400, color_func=black_color_func).generate(reviews_text)
# Display the WordCloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Remove punctuation and convert to lowercase
reviews_text = reviews_text.translate(str.maketrans('', '', string.punctuation))
reviews_text = reviews_text.lower()
# Tokenize the text
words = nltk.word_tokenize(reviews_text)
# Remove stop words
words = [word for word in words if word not in stop_words]
# Calculate word frequency
freqdist = nltk.FreqDist(words)
top_words = freqdist.most_common(20)
# Create a bar plot of the top 20 most common words
sns.barplot(x=[w[1] for w in top_words], y=[w[0] for w in top_words])
plt.title('Top 20 Most Common Words in Review Text')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

# Correlation matrix heatmap
corr_matrix = pandas_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


plotScatterMatrix(pandas_df, 6, 2)


# Stop the Spark session
spark.stop()
