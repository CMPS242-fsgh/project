import scipy

class OneVsRest:
    def __init__(self, classifier, *args, **kargs):
        self.classifier = classifier
        self._cl = []
        self._args = args
        self._kargs = kargs

    def fit(self, X, Y):
        self.n_features = Y.shape[1]
        self.n_samples = Y.shape[0]
        for i in range(self.n_features):
            if i % 30 == 0:
                print i, '/', self.n_features, 'fitted'
            model = self.classifier(*self._args, **self._kargs)
            #@print X.shape, Y[:,i].shape
            model.fit(X, Y[:,i])
            self._cl.append(model)

    def predict(self, Xt):
        ret = []
        for i in range(self.n_features):
            pre = self._cl[i].predict(Xt)
            ret.append(pre)
        return scipy.vstack(ret).T

class LabelPowerset:
    def __init__(self, classifier, **kargs):
        self._cl = classifier()

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.label_count = y.shape[1]
        last_id = 0
        self.unique_combinations = {}
        self.reverse_combinations = []
        self.label_count = y.shape[1]

        train_vector = []
        for labels_applied in y:
            label_string = self.encode2(labels_applied)
            if label_string not in self.unique_combinations:
                self.unique_combinations[label_string] = last_id
                self.reverse_combinations.append(labels_applied)
                last_id += 1

            train_vector.append(self.unique_combinations[label_string])

        self._cl.fit(X, train_vector)

        print "fit done"

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        # this will be an np.array of integers representing classes
        lp_prediction = self._cl.predict(X)
        from scipy import sparse
        result = sparse.lil_matrix((X.shape[0], self.label_count), dtype='float')
        r2 = []
        print scipy.matrix(self.reverse_combinations)
        for row in xrange(len(lp_prediction)):
            assignment = lp_prediction[row]
            #print self.reverse_combinations[assignment]
            #result[row, self.reverse_combinations[assignment]] = 1
            r2.append(self.reverse_combinations[assignment])
        #print r2
        #print result.toarray()
        #return result.toarray()
        return scipy.matrix(r2)

    def fit2(self, X, Y):
        comb = set()
        Ys = scipy.zeros(Y.shape[0])
        self._n = Y.shape[1]
        Ys = self.encode(Y)
        print type(Ys)
        #print Ys
        #from sklearn.utils.multiclass import unique_labels
        #exit()
        self._cl.fit(X, Ys)
        return Ys

    def predict2(self, X):
        Ys = self._cl.predict(X)
        #print Ys
        Y = self.decode(Ys)
        return scipy.matrix(Y, dtype=float)

    def encode(self, Y):
        r = []
        for feature in Y:
            r0 = 0
            for i,v in enumerate(feature):
                r0 = r0 << 1
                r0 += int(v)
            r.append(r0)
            #print 'encode', r0
        return r

    def encode2(self, feature):
        r0 = 0
        for i,v in enumerate(feature):
            r0 = r0 << 1
            r0 += int(v)
            #print 'encode', r0
        return r0


    def decode(self, Y):
        r = []
        for en in Y:
            r.append(map(lambda x: 1 if en & x else 0, [1<<i for i in range(self._n-1, -1, -1)]))
        return r

import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

class LabelPowerSetClassifier:

    def __init__(self, estimator, **args):
        self.estimator = estimator()

    def fit(self, X, y):

        # Code in the label power set
        self._n = y.shape[1]
        encoding_matrix = np.exp2(np.arange(y.shape[1])).T
        y_coded = safe_sparse_dot(y, encoding_matrix, dense_output=True)

        self.estimator.fit(X, y_coded)

    def predict(self, X):
        y_coded = self.estimator.predict(X)
        binary_code_size = self._n

        shifting_vector = 2 ** np.arange(binary_code_size)

        # Shift the binary representation of a class
        y_shifted = y_coded.reshape((-1, 1)) // shifting_vector
        y_shifted = y_shifted.astype(np.int)

        # Decode y by checking the appropriate bit
        y_decoded = np.bitwise_and(0x1, y_shifted)

        print y_coded
        print '!!!!!'
        return y_decoded

from scipy import sparse

class MLkNN:
    def __init__(self, model, k=5):
        self.k = k
        self.s = 1.0
        self._cl = model
        self.knn = None

    def fit(self, X, Y):
        N = X.shape[0]
        D = Y.shape[1]

        #print 'size', N, D
        self.knn = self._cl(self.k).fit(X)
        neighbors = self.knn.kneighbors(X, self.k, return_distance=False)

        #print 'neighbors', neighbors
        self.prior = scipy.array((self.s + Y.sum(axis=0))/(self.s * 2 + N))

        c = scipy.zeros((D, self.k + 1), dtype='i8')
        cn = scipy.zeros((D, self.k + 1), dtype='i8')

        Y = scipy.matrix(Y)
        for i in range(N):
            deltas = Y[neighbors[i],:].sum(axis=0)
            #print 'deltas', deltas
            for label in range(D):
                label = int(label)
                total = int(deltas[0, label])
                #print label, total,c.shape
                if Y[i, label] == 1:
                    c[label, total] += 1
                else:
                    cn[label, total] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_true = scipy.zeros((D, self.k + 1), dtype='float')
        cond_false = scipy.zeros((D, self.k + 1), dtype='float')

        for label in range(D):
            for neighbor in range(self.k + 1):
                cond_true[label,neighbor] = (self.s + c[label, neighbor]) / (self.s * (self.k + 1) + c_sum[label])
                cond_false[label,neighbor] = (self.s + cn[label, neighbor]) / (self.s * (self.k + 1) + cn_sum[label])

        self.train_labels = Y
        self.D = D
        self.cond_true = cond_true
        self.cond_false = cond_false
        #print self.prior

    def predict(self, X):
        result = np.zeros((X.shape[0], self.D), dtype='i8')
        neighbors = self.knn.kneighbors(X, self.k, return_distance=False)
        for i in range(X.shape[0]):
            deltas = self.train_labels[neighbors[i],].sum(axis=0)
            for label in range(self.D):
                count = int(deltas[0, label])
                p_true = self.prior[label] * self.cond_true[label,count]
                p_false = (1 - self.prior[label]) * self.cond_false[label,count]
                result[i,label] = int(p_true >= p_false)

        return result

if __name__ == "__main__":
    X = scipy.matrix([
        [1, 1],
        [1, 2],
        [-1, 3],
        [-1, -1]])
    Y = scipy.array([[0, 0],
                 [0, 0],
                 [0, 1],
                 [1, 1]])

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    model = LabelPowerset(LinearSVC)
    model.fit(X, Y)

    Xt = scipy.matrix([
        [-1, -1]
    ])
    print model.predict(Xt)
