import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def review_to_wordlist( review, remove_stopwords = False):
  review_text = BeautifulSoup(review).get_text()
  review_text = re.sub("[^A-Za-z]", " ",review_text)
  words = review_text.lower().split()
  
  if remove_stopwords:
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]

  return words

def makeFeatureVec(words, model, num_features):
  featureVec = np.zeros((num_features,), dtype="float32")
  nwords = 0.

  index2word_set = set(model.index2word)
  for word in words:
    if word in index2word_set:
      nwords = nwords + 1.
      featureVec = np.add(featureVec, model[word])

  featureVec = np.divide(featureVec, nwords)
  return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
  counter = 0.

  reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype = "float32")
  
  for review in reviews:
    if counter % 1000 == 0:
      print "Review %d Done." %(counter)

    reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
    counter = counter + 1.

  return reviewFeatureVecs

def create_bag_of_centroids( wordlist, word_centroid_map ):
  num_centroids = max( word_centroid_map.values() ) + 1

  bag_of_centroids = np.zeros( num_centroids, dtype = "float32" )

  for word in wordlist:
    if word in word_centroid_map:
      index = word_centroid_map[word]
      bag_of_centroids[index] += 1

  return bag_of_centroids


train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)

from gensim.models import Word2Vec

model = Word2Vec.load("300feature_40minwords_10context")
num_features = 300

from sklearn.cluster import KMeans
print("KMeans Clustering..")

word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit( word_vectors )

word_centroid_map = dict(zip( model.index2word, idx ))


clean_train_reviews = []
for review in train["review"]:
  clean_train_reviews.append( review_to_wordlist( review, remove_stopwords = True) )

trainCentroids = np.zeros( (train["review"].size, num_clusters), dtype = "float32" )

counter = 0
for review in clean_train_reviews:
  trainCentroids = create_bag_of_centroids( review, word_centroid_map )
  counter += 1


print "Creating centroids for test reviews.."
clean_test_reviews = []
for review in test["review"]:
  clean_test_reviews.append( review_to_wordlist( review, remove_stopwords = True ) )

testCentroids = np.zeros( (test["test"].size, num_clusters), dtype = "float32" )

counter = 0
for review in clean_test_reviews:
  testCentroids = create_bag_of_centroids( review, word_centroid_map )
  counter += 1

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a ramdom forest to labeled training data.. "
forest = forest.fit( trainCentroids, train["sentiment"] )

result = forest.predict( testCentroids )

output = pd.DataFrame( data = {"id": test["id"], "sentiment": result } )
output.to_csv("Word2Vec_Centroids.csv", index = False, quoting = 3 )

