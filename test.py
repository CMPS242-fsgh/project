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
    return v.transform(it)

def lib_test_naive_bayes(X, y, Xt, yt):
    print "Test naive bayes from sklearn"
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.score(Xt, yt)
    print "score", s, 'on', Xt.shape[0], 'data'

def lib_test_logistic(X, y, Xt, yt):
    print "Test Logistic Regression from sklearn"
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    clf = model.fit(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.score(Xt, yt)
    print "score", s, 'on', Xt.shape[0], 'data'
    x = Xt[0]
    #print x.toarray()
    print model.predict_proba(x)

def my_test_naive_bayes(X, y, Xt, yt):
    print "Test naive bayes from my library"
    import classifier
    model = classifier.NaiveBayes()
    model.train(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.validate(Xt, yt)
    print "score", float(s)/Xt.shape[0], 'on', Xt.shape[0], 'data'

def my_test_logistic(X, y, Xt, yt):
    import classifier
    model = classifier.LogisticRegression()
    model.train(X, y)
    print 'Model trained on', X.shape[0], 'data'
    s = model.validate(Xt, y)
    print "score", float(s)/Xt.shape[0], 'on', Xt.shape[0], 'data'
    x = Xt[0]
    print model.predict(x)

def split_training(X, y, per):
    total = X.shape[0]
    train = int(total * per)
    print train, total
    test = total - train
    return X[:train], y[:train], X[train:], y[train:]

#docs, y = data_simple_nlp()
docs, y = data_yelp(limit = 5000)
per = 0.5
X = my_dict_vectorizer(docs)
#print X
#X = lib_hash_vectorizer(docs)
X, y, Xt, yt = split_training(X, y, per)
#lib_test_logistic(X, y, Xt, yt)
lib_test_naive_bayes(X, y, Xt, yt)
lib_test_logistic(X, y, Xt, yt)
my_test_naive_bayes(X, y, Xt, yt)
my_test_logistic(X, y, Xt, yt)
