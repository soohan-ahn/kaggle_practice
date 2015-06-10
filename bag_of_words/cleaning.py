from bs4 import BeautifulSoup
import pandas as pd
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def review_to_words( raw_review ):
    review_words_only = BeautifulSoup(raw_review)
    review_words_only = re.sub("[^0-9a-zA-z]", " ", review_words_only.get_text())
    review_words_only = review_words_only.lower()
    # words_in_lower_case = only_letters_lower.split()
    
    # words_without_stopwords = [w for w in words_in_lower_case if not w in stopwords.words("english")]
    # return " ".join( words_without_stopwords )
    return review_words_only


train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter="\t", quoting=3)
total_size = train["review"].size
clean_reviews = []
for i in xrange( 0, total_size ):
    clean_reviews.append( review_to_words(train["review"][i]) )
    if (i + 1) % 1000 == 0:
        print("%d th review done.\n",i + 1)
# example1 = review_to_words(train["review"][0])
# print example1

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stopwords.words("english"), max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_reviews)
train_data_features = train_data_features.toarray()

# print train_data_features.shape

# vocab = vectorizer.get_feature_names()

# dist = np.sum(train_data_features, axis = 0)

# for tag, count in zip(vocab, dist):
#    print count, tag

print "Training the random forest..."

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit( train_data_features, train["sentiment"] )

print "Predicting the test reviews..."
test = pd.read_csv("testData.tsv", header = 0, delimiter="\t", quoting=3)
total_test_size = test["review"].size
clean_test_reviews = []
for i in xrange( 0, total_test_size ):
    clean_test_reviews.append( review_to_words(test["review"][i]) )
    if (i + 1) % 1000 == 0:
        print("%d th review done.\n",i + 1)

test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame( data = {"id":test["id"], "sentiment":result} )

output.to_csv("Bag_of_Words_model_5000.csv", index = False, quoting = 3)

