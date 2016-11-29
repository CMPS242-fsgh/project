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
