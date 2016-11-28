'''
a simple script to store yelp data into sqlite database
'''

import json
import sqlite3

import os

if os.path.exists('data/data.db'):
    y = raw_input("Delete original database?[y/n]")
    if y=='y':
        os.remove('data/data.db')
    else:
        print "not deleted, exiting..."
        exit()

conn = sqlite3.connect('data/data.db')
conn.execute('CREATE TABLE business '  +
                     '(id integer primary key,' +
                     'business_id text, '+
                     'cat text,' +
                     'count_review integer, ' +
                     'name text);')

with open('data/yelp_academic_dataset_business.json') as f:
    for i, line in enumerate(f.readlines()):
        obj = json.loads(line)
        conn.execute("INSERT INTO business VALUES (?, ?, ?, ?, ?);", 
            (i, obj['business_id'], json.dumps({'list': obj['categories']}), obj['review_count'],obj['name'])
        )
        #break
#exit()
conn.execute('CREATE TABLE review (review_id text, business_id text, content text, stars integer);')
with open('data/yelp_academic_dataset_review.json') as f:
    for line in f.readlines():
        obj = json.loads(line)
        #r = conn.execute('SELECT id FROM business WHERE business_id=?', obj['business_id'])
        #id, = r.fetchall()
        conn.execute("INSERT INTO review VALUES (?, ?, ?, ?);",
            (obj['review_id'], obj['business_id'], obj['text'], obj['stars'])
        )

conn.execute("CREATE INDEX b_id_index ON review(business_id);")

conn.commit()


print conn.execute('SELECT count_review FROM business WHERE business_id=?', ('QoDa50dc7g62xciFygXB9w',)).fetchall()
print conn.execute('SELECT content FROM review WHERE business_id=?', ('QoDa50dc7g62xciFygXB9w',)).fetchall()

conn.close()
