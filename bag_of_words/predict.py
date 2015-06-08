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


train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)

from gensim.models import Word2Vec

model = Word2Vec.load("300feature_40minwords_10context")
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3


clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append( review_to_wordlist( review, remove_stopwords = True) )

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print "Creating average feature vecs for test reviews.."
clean_test_reviews = []
for review in test["review"]:
	clean_test_reviews.append( review_to_wordlist( review, remove_stopwords = True ) )

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a ramdom forest to labeled training data.. "
forest = forest.fit( trainDataVecs, train["sentiment"] )

result = forest.predict( testDataVecs )

output = pd.DataFrame( data = {"id": test["id"], "sentiment": result } )
output.to_csv("Word2Vec_AverageVectors.csv", index = False, quoting = 3 )

