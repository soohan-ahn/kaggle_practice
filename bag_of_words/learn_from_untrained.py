import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data

train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)

def review_to_wordlist( review, remove_stopwords = False):
	review_text = BeautifulSoup(review).get_text()
	review_text = re.sub("[^A-Za-z]", " ",review_text)
	words = review_text.lower().split()
	
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]

	return words

def review_to_sentences( review, tokenizer, remove_stopwords = False ):
	raw_sentences = tokenizer.tokenize(review.decode('utf-8').strip())
	sentences = []
	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0:
			sentences.append( review_to_wordlist( raw_sentence, remove_stopwords) )

	return sentences

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []

print "Parsing sentences from training set.. "
for review in train["review"]:
	sentences += review_to_sentences( review, tokenizer )

print "Parsing sentences form unlabeled set.. "
for review in unlabeled_train["review"]:
	sentences += review_to_sentences( review, tokenizer )


import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', lavel=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

from gensim.models import word2vec
print "Training model.. "
model = word2vec.Word2Vec(sentences, workers = num_workers, size = num_features, min_count = min_word_count, window = context, sample = downsampling)

model.init_sims(replace = True)

model_name = "300feature_40minwords_10context"
model.save(model_name)

