

import json
import sqlite3
import collections

database = 'data/data.db'
Record = collections.namedtuple('Record', ['business_id', 'reviews', 'categories'])

class DataLoader():
    def __init__(self):
        self._conn = sqlite3.connect(database)
        self._datasize = self._conn.execute('SELECT count(business_id) FROM business;').fetchone()[0]
        print "{0} business loaded".format(self._datasize)

    def alldata(self):
        sql = 'SELECT business_id, cat FROM business;'
        l = []
        for business_id, cat in self._conn.execute(sql):
            sql = 'SELECT content FROM review WHERE business_id=?'
            for content in self._conn.execute(sql, (business_id,)):
                l.append(content[0])

            yield Record(business_id, l, json.loads(cat)['list'])


if __name__ == '__main__':
    d = DataLoader()
    from pprint import pprint
    c = 0
    for i in d.alldata():
        pprint((i.business_id, i.categories))
        c = c+1

        if c > 20:
            break
