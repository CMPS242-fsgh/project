import sys
import time
import warnings

import loader
from feature import CountFeature
from classifier import (NaiveBayes, LogisticRegression)
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

def load_data(n, n_feature, n_test, vectorizer):
    N = n_test + n
    YY, categories = loader.get_target(n_feature, N)
    XX = vectorizer(loader.review_from_file(N))

    X, Y, Xt, Yt = loader.split_training(XX, YY, n_test)
    return X, Y, Xt, Yt, categories

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

import feature
v = feature.CountFeature(use_stopwords=True, bigram=False)
def f(it):
    return v.transform(it)

X, Y, Xt, Yt, cats = load_data(90000, 15, 10, f)


from sklearn.feature_selection import RFE
import scipy
class OneVsRest:
    def __init__(self, classifier, cats, voc, *args, **kargs):
        self.classifier = classifier
        self._cl = []
        self._args = args
        self._kargs = kargs
        self.cats = cats
        self.voc = voc

    def fit(self, X, Y):
        self.n_labels = Y.shape[1]
        self.n_samples = Y.shape[0]
        for i in range(self.n_labels):
            print i, '/', self.n_labels, 'fitted', self.cats[i],':',
            model = self.classifier(*self._args, **self._kargs)
            #@print X.shape, Y[:,i].shape
            model.fit(X, Y[:,i])
            mat = model._mat
            minus = mat[0] - mat[1]
            minus = minus.A1
            indices = minus.argsort()[-10:]
            indices = list(reversed(indices))

            #print scipy.argmax(minus), indices[0]
            #print list(indices)
            for j in indices:
                print self.voc[j],
            print '\n'
            self._cl.append(model)

    def predict(self, Xt):
        ret = []
        for i in range(self.n_features):
            pre = self._cl[i].predict(Xt)
            ret.append(pre)
        return scipy.vstack(ret).T

from classifier import NaiveBayes
classifier = NaiveBayes

ivoc = [0]*len(v.vocab)
for k,v in v.vocab.items():
    ivoc[v] = k
model = OneVsRest(classifier, cats, ivoc)
model.fit(X, Y)
