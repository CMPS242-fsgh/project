import sys
import scipy

def data_yelp(limit=40):
    import loader
    d = loader.DataLoader()
    y = scipy.zeros(limit)
    g = d.binary_producer(limit, y, 'Restaurants')
    return g, y

def data_simple_nlp():
    docs = [
            "Chinese Beijing Chinese",
            "Chinese Chinese Shanghai",
            "Chinese Macao",
            "Tokyo Japan Chinese",
            "Chinese Chinese Chinese Tokyo Japan"
        ]
    y = [1, 1, 1, 0, 1]
    return docs, scipy.array(y)

import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed

def lib_count_vectorizer(it):
    from sklearn.feature_extraction.text import CountVectorizer
    v = CountVectorizer(it, stop_words='english')
    return v.fit_transform(it)

def lib_hash_vectorizer(it):
    from sklearn.feature_extraction.text import HashingVectorizer
    v = HashingVectorizer(it, stop_words='english', non_negative=True, norm=None)
    return v.transform(it)

def my_dict_vectorizer(it):
    import feature
    v = feature.CountFeature(limit = -1)
    return v.transform(it), v

def lib_test_naive_bayes(X, y, Xt, yt):
    print "Test naive bayes from sklearn"
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.score(Xt, yt)
    print "score", s, 'on', Xt.shape[0], 'data'

@timeit
def lib_test_logistic(X, y, Xt, yt):
    print "Test Logistic Regression from sklearn"
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='newton-cg')
    clf = model.fit(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.score(Xt, yt)
    print "score", s, 'on', Xt.shape[0], 'data'
    x = Xt[0]
    #print x.toarray()
    #print model.predict_proba(Xt)

def my_test_naive_bayes(X, y, Xt, yt):
    print "Test naive bayes from my library"
    import classifier
    model = classifier.NaiveBayes()
    model.train(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.validate(Xt, yt)
    print "score", float(s)/Xt.shape[0], 'on', Xt.shape[0], 'data'

@timeit
def my_test_logistic(X, y, Xt, yt):
    import classifier
    model = classifier.LogisticRegression()
    model.train(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.validate(Xt, yt)
    pd = model.predict_many(Xt)
    #print pd
    print "score", float(s)/Xt.shape[0], 'on', Xt.shape[0], 'data'
    x = Xt[0]
    #print model.predict(x)

def split_training(X, y, per):
    total = X.shape[0]
    train = int(total * per)
    #print train, total
    test = total - train
    return X[:train], y[:train], X[train:], y[train:]

def data_review(limit = 500):
    c = 0
    f = open('data/reviews.csv')
    for line in f:
        if c % 1000 ==0:
            print c ,'/', limit

        if c<limit:
            yield line
        else:
            break
        c += 1
    f.close()

def data_cat(limit = 500):
    c = 0
    f = open('data/cat.csv')
    for line in f:
        if c % 1000 ==0:
            print c ,'/', limit

        if c<limit:
            yield line
        else:
            break
        c += 1
    f.close()

def sp(doc):
    for word in doc.split(','):
        word = word.strip()
        #print word, word in stopwords
        word = word.lower()
        yield word


def build_dict(limit=100):
    f = open('data/label_rank.csv')
    vocab = dict()
    for line in f.readlines()[:limit]:
        line = line.strip()
        line = line.lower()
        vocab.setdefault(line, len(vocab))
    return vocab


#docs, y = data_simple_nlp()
#docs, y = data_yelp(limit = 600)
n = 1000
per = 0.8
XX,_ = my_dict_vectorizer(data_review(n))
import feature

vocab = build_dict()
v = feature.CountFeature(splitter = sp, voc = vocab)
Y = v.transform(data_cat(n))
print XX.shape, Y.shape

#y = Y[v.vocab['restaurants']]
print Y.sum(axis=0)
#print v.vocab

invis_vocab = [0]*len(v.vocab)
for k,va in v.vocab.items():
    invis_vocab[va] = k

#exit()
YY = Y.toarray()
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
X, Y, Xt, Yt = split_training(XX, YY, per)

print X.shape

print "---skearn OneVsRest + SVC---"
model = OneVsRestClassifier(LinearSVC())
model.fit(X, Y)
r = model.predict(Xt)
import compute_Metrics

compute_Metrics.computeMetrics(r, Yt)

from skmultilearn.problem_transform import (BinaryRelevance, LabelPowerset)
import sklearn.metrics

# assume data is loaded using
# and is available in X_train/X_test, y_train/y_test

# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation
print X.shape, Y.shape
classifier = LabelPowerset(classifier = MultinomialNB(), require_dense = [False, True])

# train
classifier.fit(X, Y)

# predict
predictions = classifier.predict(Xt)
predictions = predictions.toarray()
compute_Metrics.computeMetrics(predictions, Yt)


from skmultilearn.adapt import MLkNN
model = MLkNN()
model.fit(X, Y)
pre = model.predict(Xt)
compute_Metrics.computeMetrics(pre, Yt)






#y = Y.toarray()[:,v.vocab['restaurants']]
#jprint repr(X.toarray())
#X = lib_hash_vectorizer(docs)
#print y.shape
#X, y, Xt, yt = split_training(X, y, per)
#lib_test_logistic(X, y, Xt, yt)

print '--------'
from sklearn.naive_bayes import MultinomialNB
ret = []
for cat in invis_vocab:
    #print cat
    #print "Classify on [{0}]".format(cat)
    model = MultinomialNB()
    y = YY[:,v.vocab[cat]]
    #print y.shape, YY.shape
    X, y, Xt, yt = split_training(XX, y, per)
    #print y.shape
    model.fit(X, y)
    #print 'Model trained on', X.shape[0], 'data'
    s = model.score(Xt, yt)
    #print ">> score [", s, ']on', Xt.shape[0], 'data'
    pre = model.predict(Xt)
    ret.append(pre)

output = scipy.vstack(ret).T

compute_Metrics.computeMetrics(output, Yt)
#lib_test_naive_bayes(X, y, Xt, yt)
#lib_test_logistic(X, y, Xt, yt)
#my_test_naive_bayes(X, y, Xt, yt)
#my_test_logistic(X, y, Xt, yt)
#import LogisticRegression
#model = LogisticRegression.LogisticRegression(X.toarray(), y)
#model.fit()
#print model.predict(Xt)
