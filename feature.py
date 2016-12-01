from scipy import sparse
import numpy as np

from stopwords import stopwords

class CountFeature:
    def __init__(self, voc = None, use_stopwords = True, limit = 0, splitter = None, bigram=False):
        if limit > 0:
            raise "Not Implemented"

        self.bigram = bigram
        self._limit = limit
        self.use_stopwords = use_stopwords
        if splitter:
            self.splitter = splitter
        else:
            self.splitter = self._splitter
        if not voc:
            self._new_vocab = True
            self.vocab = dict()
        else:
            self._new_vocab = False
            self.vocab = voc

    def _splitter(self, doc):
        last_word = ''
        for word in doc.split():
            #print word, word in stopwords
            word = word.lower()
            if self.use_stopwords:
                if word not in stopwords:
                    yield word
                    if self.bigram and last_word:
                        yield last_word+' '+word
                    last_word = word
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

    def fit_transform(self, it):
        return self.transform(it)

    def transform(self, it):
        indices = []
        ind_ptr = []
        values = []

        ind_ptr.append(0)

        for doc in it:
            feature_count = dict()
            for token in self.splitter(doc):
                #print token
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
        values = np.asarray(values, dtype = np.float32)
        ind_ptr = np.asarray(ind_ptr, dtype = np.intc)
        X = sparse.csr_matrix((values, indices, ind_ptr),
                              shape=(len(ind_ptr) - 1, len(self.vocab)),
                              dtype=np.float32)
        X.sort_indices()
        #print self.vocab
        return X


if __name__=='__main__':
    f = CountFeature(use_stopwords=True, bigram=True)
    dataset = [
        "Hello World Hello a",
        "Good Luck",
        "a"
    ]
    X = f.transform(dataset)
    print X.toarray()
    print f.vocab
