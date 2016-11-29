

import json
import sqlite3
import collections

__all__ = ['DataLoader', 'get_target', 'split_training']
database = 'data/data.db'
Record = collections.namedtuple('Record', ['business_id', 'reviews', 'categories'])

class DataLoader():
    def __init__(self):
        self._conn = sqlite3.connect(database)
        self.datasize = self._conn.execute('SELECT count(business_id) FROM business;').fetchone()[0]
        print "{0} business loaded".format(self.datasize)

    def alldata(self):
        sql = 'SELECT business_id, cat FROM business;'
        #l = []
        for business_id, cat in self._conn.execute(sql):
            l = []
            sql = 'SELECT content FROM review WHERE business_id=?'
            for content in self._conn.execute(sql, (business_id,)):
                l.append(content[0])

            yield Record(business_id, l, json.loads(cat)['list'])

    def binary_producer(self, n, y, cat):
        g = self.alldata()
        def iter_d():
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

        return iter_d()

    def data_as_list(self, n):
        r = []
        for i, record in enumerate(self.alldata()):
            r.append(record)
            if i > n:
                break
        return r

    def first_100(self):
        r = []
        c = 0
        nr = 0
        for business in self.alldata():
            c += 1
            if u"Shopping" in business.categories:
                label = "Restaurant"
            else:
                label = "Not"
	    '''if(c > 70 and c <= 100):
		print(str(business.categories)+"\n")
	    '''
            r.append(Record(business.business_id, "".join(business.reviews), label))
            print(len(business.reviews))
            nr += len(business.reviews)
            #print nr
            if c>=100:
                print nr
                return r

    def count_cat(self, cat):
        sql = 'SELECT count(business_id) FROM business WHERE cat LIKE ?'
        n = self._conn.execute(sql, ("%"+cat+"%",)).fetchone()[0]
        return n

def review_from_file(limit = 500):
    c = 0
    f = open('data/reviews.csv')
    for line in f:
        if c % 1000 ==0:
            print 'reading reviews', c ,'/', limit

        if c<limit:
            yield line
        else:
            break
        c += 1
    f.close()

def cat_from_file(limit = 500):
    c = 0
    f = open('data/cat.csv')
    for line in f:
        if c<limit:
            yield line
        else:
            break
        c += 1
    f.close()

def sp_cat(doc):
    for word in doc.split(','):
        word = word.strip()
        word = word.lower()
        yield word

def build_cat_dict(limit=100):
    f = open('data/label_rank.csv')
    vocab = dict()
    for line in f.readlines()[:limit]:
        line = line.strip()
        line = line.lower()
        vocab.setdefault(line, len(vocab))
    return vocab

import feature
def get_target(limit = 500):
    vocab = build_cat_dict()
    v = feature.CountFeature(splitter = sp_cat, voc = vocab)
    Y = v.transform(cat_from_file(limit))
    all_cat = list(vocab.items())
    all_cat.sort(key=lambda x:x[1])
    return Y.toarray(), map(lambda x:x[0], all_cat)

def split_training(X, y, per):
    total = X.shape[0]
    train = int(total * per)
    #print train, total
    test = total - train
    return X[:train], y[:train], X[train:], y[train:]


if __name__ == '__main__':
    d = DataLoader()
    from pprint import pprint
    c = 0
    for i in d.alldata():
        pprint((i.business_id, i.categories))
        c = c+1

        if c > 20:
            break
