from scipy import sparse
import numpy as np

from stopwords import stopwords

class CountFeature:
    def __init__(self, voc = None, use_stopwords = True, limit = 0):
        self._limit = limit
        if not voc:
            self._new_vocab = True
            self.vocab = dict()
            self.use_stopwords = use_stopwords
        else:
            self._new_vocab = False
            self.vocab = voc

    def splitter(self, doc):
        for word in doc.split():
            if self.use_stopwords and word not in stopwords:
                yield word
            else:
                yield word

    def lookup_vocabulary(self, word):
        if self._new_vocab:
            return self.vocab.setdefault(word, len(self.vocab))
        else:
            if word in self.vocab:
                return self.vocab[word]
            else:
                return -1

    def transform(self, it):
        indices = []
        ind_ptr = []
        values = []

        ind_ptr.append(0)

        for doc in it:
            feature_count = dict()
            for token in self.splitter(doc):
                idx = self.lookup_vocabulary(token)
                if idx < 0:
                    continue

                if idx in feature_count:
                    feature_count[idx] += 1
                else:
                    feature_count[idx] = 1

            indices.extend(feature_count.keys())
            values.extend(feature_count.values())
            ind_ptr.append(len(indices))

        indices = np.asarray(indices, dtype=np.intc)
        values = np.asarray(values, dtype = np.intc)
        ind_ptr = np.asarray(ind_ptr, dtype = np.intc)
        X = sparse.csr_matrix((values, indices, ind_ptr),
                              shape=(len(ind_ptr) - 1, len(self.vocab)),
                              dtype=np.intc)
        X.sort_indices()
        return X


if __name__=='__main__':
    f = CountFeature()
    dataset = [
        "Hello World Hello",
        "Good Luck"
    ]
    X = f.transform(dataset)
    print X.toarray()
