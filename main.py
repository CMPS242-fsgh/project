#!/usr/bin/env python2

import argparse
import sys
import time


import loader
from feature import CountFeature
from classifier import NaiveBayes
from multi import OneVsRest
from compute_Metrics import computeMetrics

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed

@timeit
def load_data(n, n_feature, per, vectorizer):
    YY, categories = loader.get_target(n)
    XX = vectorizer(loader.review_from_file(n))
    X, Y, Xt, Yt = loader.split_training(XX, YY, per)
    return X, Y, Xt, Yt, categories

@timeit
def run_OneVsRest(data, ensembler, classifier):
    X, Y, Xt, Yt, cats = data
    model = ensembler(classifier)
    import warnings
    model.fit(X, Y)
    Yp = model.predict(Xt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        computeMetrics(Yp, Yt, cats)

def lib_count_vectorizer(it, stop=True):
    from sklearn.feature_extraction.text import CountVectorizer
    if stop:
        stop = 'english'
    else:
        stop = None
    v = CountVectorizer(it, stop_words=stop)
    return v.fit_transform(it)

def lib_hash_vectorizer(it, stop=True):
    from sklearn.feature_extraction.text import HashingVectorizer
    if stop:
        stop = 'english'
    else:
        stop = None
    v = HashingVectorizer(it, stop_words=stop, non_negative=True, norm=None)
    return v.transform(it)

def my_dict_vectorizer(it, stop=True):
    import feature
    v = feature.CountFeature(limit = -1, use_stopwords=stop)
    return v.transform(it)


print "CMPS242 project"
print "Loading data"
# TODO: modify this:
n_samples = 60000
n_labels = 500
per = 0.6
vectorizer = my_dict_vectorizer
data = load_data(n_samples, n_labels, per, my_dict_vectorizer)

X, Y, Xt, Yt, cat = data
print type(X), type(Y)
#print "run OneVsRest with our implementation"
run_OneVsRest(data, OneVsRest, NaiveBayes)

print "run OneVsrest with sklearn"
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
print ">> Multinomial Naive Bayes"
run_OneVsRest(data, OneVsRestClassifier, MultinomialNB())
print ">> LinearSVC"
run_OneVsRest(data, OneVsRestClassifier, LinearSVC())
