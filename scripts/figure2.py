
# coding: utf-8

# In[70]:

import json
f = open('data/yelp_academic_dataset_review.json')


# In[71]:

m = dict()         # m for bussiness
rv = list()        # rv for review
for line in f.readlines():
    obj = json.loads(line)
    r_id = obj['review_id']
    b_id = obj['business_id']
    if m.has_key(b_id):
        m[b_id] += 1
    else:
        m[b_id] = 1
    rv.append(len(obj['text'].split(' ')))



h = m.values()

import matplotlib.pyplot as plt
#plt.hist(rv, bins=range(0, 300, 20))
#plt.xlim(0, 300)
#plt.xlabel("Number of Words of the Review")
#plt.ylabel("Number of Reviews")
#plt.show()

plt.hist(h, bins=range(0, 500, 5))
plt.xlim(0, 500)
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Businesses")
plt.show()


