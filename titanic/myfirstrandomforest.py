""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train_df['Title'] = train_df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

train_df['Title'][train_df.Title.isin(['Jonkheer','Master','Dr'])] = 1
train_df['Title'][train_df.Title.isin(['Ms','Mlle','Miss'])] = 2
train_df['Title'][train_df.Title.isin(['Mrs','Mme'])] = 3
train_df['Title'][train_df.Title.isin(['Capt','Don','Major','Col','Sir','Rev'])] = 4
train_df['Title'][train_df.Title.isin(['Dona','Lady','the Countess'])] = 5
train_df['Title'][train_df.Title.isin(['Mr'])] = 6

train_df = pd.concat([train_df, pd.get_dummies(train_df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis = 1)


# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

#min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = preprocessing.StandardScaler()
train_df['Age_range'] = min_max_scaler.fit_transform(train_df['Age'])
train_df['Fare_range'] = min_max_scaler.fit_transform(train_df['Fare'])

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Age', 'Fare'], axis=1) 


# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

test_df['Title'] = test_df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
test_df['Title'][test_df.Title.isin(['Jonkheer','Master','Dr'])] = 1
test_df['Title'][test_df.Title.isin(['Ms','Mlle','Miss'])] = 2
test_df['Title'][test_df.Title.isin(['Mrs','Mme'])] = 3
test_df['Title'][test_df.Title.isin(['Capt','Don','Major','Col','Sir','Rev'])] = 4
test_df['Title'][test_df.Title.isin(['Dona','Lady','the Countess'])] = 5
test_df['Title'][test_df.Title.isin(['Mr'])] = 6

test_df = pd.concat([test_df, pd.get_dummies(test_df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis = 1)


# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

test_df['Age_range'] = min_max_scaler.fit_transform(test_df['Age'])

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

test_df['Fare_range'] = min_max_scaler.fit_transform(test_df['Fare'])

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Age', 'Fare'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=10000, min_samples_split = 45, max_features = 7)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Examinate the features...'
features_list = train_df.columns.values[1::]
feature_importance = forest.feature_importances_
feature_importance = (feature_importance / feature_importance.max()) * 100.0
fi_threshold = 15
important_idx = np.where(feature_importance > fi_threshold)[0]

important_features = features_list[important_idx]

sorted_idx = np.argsort(feature_importance[important_idx])[::-1]

pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1,2,2)
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align = 'center')
plt.yticks(pos, important_features[sorted_idx[::-1]])
plt.draw()
plt.show()

print train_df

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("myrandomforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
