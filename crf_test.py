#!/usr/bin/env python
#~*~ coding: utf-8 ~*~

import os
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
import pycrfsuite
import numpy as np
from sklearn.metrics import classification_report

# get comma training data
dir_path = "/home/aileen/epc-schulung/sepp_nlg_2021_data_v5/sepp_de_test/"
dataframe_documents = []
for file in os.listdir(dir_path):
  df = pd.read_csv(os.path.join(dir_path, file), sep='\t', header=None, usecols=[0,2], names=['token', 'label'])
  dataframe_documents.append(df)

docs = [list(df.to_records(index=False)) for df in dataframe_documents[:10]]

# generating POS-Tags (I don't know why this is separate from the features)
data = []
for i, doc in enumerate(docs):
    tokens = [str(t) for t, _ in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

print(data[0])

# generating features
def word2features(doc, i):
    word = str(doc[i][0])
    postag = str(doc[i][1])

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = str(doc[i-1][0])
        postag1 = str(doc[i-1][1])
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = str(doc[i+1][0])
        postag1 = str(doc[i+1][1])
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

# training the model

def extract_features(doc):i
    return [word2features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    return [label for (token, postag, label) in doc]

X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 0.1,

    'c2': 0.01,  

    'max_iterations': 200,

    'feature.possible_transitions': True
})

trainer.train('crf.model')

# Check the results
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

labels = {"Comma": 1, "No Comma": 0}

predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

print(classification_report(
    truths, predictions,
    target_names=["No Comma", "Comma"]))
