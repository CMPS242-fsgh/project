import numpy as np

class NaiveBayes:
    def __init__(self):
        self._prior = None
        self._mat = None

    def train(self, X, y):
        y = np.matrix(y)
        p1 = y*X
        p2 = (1-y)*X
        p = np.vstack([
            np.log(p1+1) - np.log(p1.sum() + p1.shape[1]),
            np.log(p2+1) - np.log(p2.sum() + p2.shape[1])])
        pri = np.matrix([[float(y.sum())/y.shape[1]], [1 - float(y.sum())/y.shape[1] ]])
        self._prior = np.log(pri)
        self._mat = p
        return p, pri

    def predict_many(self, mat):
        logp = self._mat*mat.T + self._prior
        ans = (np.sign(logp[0] - logp[1]) + 1)/2
        return ans.A1

    def validate(self, mat, real_y):
        predict_y = self.predict_many(mat)
        return (predict_y == real_y).sum()

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


    n = 4000
    y = np.zeros(n)
    #v = HashingVectorizer(stop_words='english', non_negative=True, norm=None)
    #mat = v.transform(iter_data(n, y, 'Restaurants'))
    import feature
    stop = True
    v = feature.CountFeature(use_stopwords = stop)
    mat = v.transform(iter_data(n, y, 'Restaurants'))
    dct = v.vocab

    print 'data readed', mat.shape, y.shape
    nt = 1000
    yt = np.zeros(nt)

    v2 = feature.CountFeature(voc = dct, use_stopwords = stop)
    mt = v2.transform(iter_data(nt, yt, 'Restaurants'))
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


