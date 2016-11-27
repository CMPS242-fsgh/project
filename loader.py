

import json
import sqlite3
import collections

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

if __name__ == '__main__':
    d = DataLoader()
    from pprint import pprint
    c = 0
    for i in d.alldata():
        pprint((i.business_id, i.categories))
        c = c+1

        if c > 20:
            break
