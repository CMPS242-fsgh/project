import numpy as np
import scipy

class NaiveBayes:
    def __init__(self):
        self._prior = None
        self._mat = None

    def fit(self, X, y):
        y = np.matrix(y)
        #print type(y)
        #y = y.T
        p1 = y*X
        p2 = (1-y)*X
        p = np.vstack([
            np.log(p1+1) - np.log(p1.sum() + p1.shape[1]),
            np.log(p2+1) - np.log(p2.sum() + p2.shape[1])])
        pri = np.matrix([[float(y.sum())/y.shape[1]], [1 - float(y.sum())/y.shape[1] ]])
        self._prior = np.log(pri)
        self._mat = p
        #print self._prior, self._mat
        return p, pri

    def predict(self, mat):
        logp = self._mat*mat.T + self._prior
        ans = (np.sign(logp[0] - logp[1]) + 1)/2
        return ans.A1

    def validate(self, mat, real_y):
        predict_y = self.predict_many(mat)
        return (predict_y == real_y).sum()

sigmoid = lambda x: 1.0/(1+scipy.exp(-x))


from sklearn.utils.extmath import (safe_sparse_dot, log_logistic)
from sklearn.utils.fixes import expit
from sklearn.utils.optimize import newton_cg

class LogisticRegression:
    def __init__(self, lbd = 1.):
        self._w = None
        self._lbd = lbd

    def _intercept_dot(self, w, X):
        z = safe_sparse_dot(X, w[:-1])
        #print X*w[:-1]
        z = z + w[-1]
        #z = X * w[:-1] + w[-1]
        return z

    def fit(self, X, y):
        n_features = X.shape[1]
        n_data = X.shape[0]
        #self._w = np.ones(n_features + 1)
        mask = (y == 1)
        y_bin = np.ones(y.shape, dtype=np.float64)
        y_bin[~mask] = -1.
        #print y_bin

        def _loss(w):
            z = self._intercept_dot(w, X)
            #loss = scipy.dot(y, scipy.log(sigmoid(z))) + scipy.dot(1-y, scipy.log(1-sigmoid(z)))
            #print (y * scipy.logaddexp(0., -z))
            #print scipy.log(sigmoid(z))
            #print "---"
            loss = scipy.sum(y * scipy.logaddexp(0., -z) + (1-y)*scipy.logaddexp(0., z))
            return loss+self._lbd/2*scipy.dot(w,w)

        def _grad(w):
            z = self._intercept_dot(w, X)
            z = expit(z)
            grad = np.empty_like(w)
            #z = expit(y_bin * z)
            #z0 = (z - 1) * y_bin
            #print X.T.toarray()
            #print (z-y)
            #print X.T*(z-y)
            #print '---'
            grad[:n_features] = X.T * (z - y) + self._lbd*w[:-1]
            grad[-1] = scipy.sum(z - y) + self._lbd*w[-1]
            return grad

        def _hess(w):
            h = np.empty_like(w)
            z = self._intercept_dot(w, X)
            z = expit(z)
            d = z(1-z)
            R = scipy.sparse.dia_matrix((d, 0),shape=(n_data, n_data))

            h[:n_features] = scipy.dot(X.T, X*R)
            #ret[:n_features] += alpha * s[:n_features]

            # For the fit intercept case.
            if fit_intercept:
                ret[:n_features] += s[-1] * dd_intercept
                ret[-1] = dd_intercept.dot(s[:n_features])
                ret[-1] += d.sum() * s[-1]
            return ret
        #hessian_value = numpy.dot(X.T, X * S[:, numpy.newaxis])

        #opt = scipy.optimize.minimize(_loss, scipy.ones(n_features + 1), method='Powell', jac=_grad)
        #print opt['x']
        opt = scipy.optimize.minimize(_loss, scipy.ones(n_features + 1), method='Newton-CG', jac=_grad)
        #print opt['x']
        #print opt
        #print X.shape, np.hstack([X, np.ones(n_data)]).shape
        self._w = opt['x']

    def predict(self, x):
        z = self._intercept_dot(self._w, x)
        return expit(z), 1-expit(z)

    def predict_many(self, X):
        Z = expit(self._intercept_dot(self._w, X))
        return Z

    def validate(self, X, y):
        mask = self.predict_many(X) > 0.5
        Z = scipy.zeros(y.shape)
        Z[mask] = 1
        return (Z == y).sum()

if __name__ == '__main__':
    import loader
    from sklearn.feature_extraction.text import HashingVectorizer

    d = loader.DataLoader()
    g = d.alldata()
    def iter_data(n, y, cat):
        c = 0
        for business in g:
            if c % 1000 == 0:
                print c, '/', n
            if c<n:
                if cat.decode('utf-8') in business.categories:
                    y[c] = 1
                else:
                    y[c] = 0
                yield "".join(business.reviews)
            else:
                return
            c += 1
#     f = open('data/yelp.csv')
#     def iter_data(n, y, cat):
#         c = 0
#         for line in f:
#             if c % 1000 == 0:
#                 print c, '/', n
#             if c < n:
#                 b_id, categories, review =  line.split('\t')
#                 categories = categories.split(',')
#                 if cat in categories:
#                     y[c] = 1
#                 else:
#                     y[c] = 0
#                 yield review
#             else:
#                 return
#             c += 1

    import feature
    from sklearn.feature_extraction.text import CountVectorizer
    #v = feature.CountFeature()
    v = HashingVectorizer(stop_words='english', non_negative=True, norm=None)
    #v = CountVectorizer()
    #v._validate_vocabulary()
    if True:
        n = 40
        y = np.zeros(n)
        producer = d.binary_producer(n, y, 'Restaurants')

        mat = v.transform(producer())
        print 'data readed', mat.shape, y.shape
        nt = 10
        yt = np.zeros(nt)
        mt = v.transform(iter_data(nt, yt, 'Restaurants'))
    #print yt
    else:
        mat = v.transform([
            "Chinese Beijing Chinese",
            "Chinese Chinese Shanghai",
            "Chinese Macao",
            "Tokyo Japan Chinese"
        ])
        y = scipy.array([1,1,1,0])
        n = 4

        mt = v.transform([
            "Chinese Chinese Chinese Tokyo Japan"
        ])
        yt = scipy.array(np.ones(1))
        nt = 1

    print mat.shape, mt.shape

    print 'our code',
    mm = NaiveBayes()
    mm.train(mat, y)
    print float(mm.validate(mt, yt))/nt

    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    clf = model.fit(mat, y)
    print 'model trained'
    s = model.score(mt, yt)
    print s

    #print mat
    from sklearn.linear_model import LogisticRegression as LR
    m = LR()
    m.fit(mat, y)
    print m.predict(mt)
