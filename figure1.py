
# coding: utf-8

# In[12]:

import json
f = open('data/yelp_academic_dataset_business.json')
m = {}
cc = {}
for line in f.readlines():
    obj = json.loads(line)
    m[obj['business_id']] = obj['categories']
    for c in obj['categories']:
        if cc.has_key(c):
            cc[c] += 1
        else:
            cc[c] = 2


# In[13]:

import matplotlib.pyplot as plt



# In[15]:

import operator
sorted_cc = sorted(cc.items(), key=operator.itemgetter(1), reverse=True)


# In[16]:

scc = sorted_cc[:30]
plt.bar(range(len(scc)), map(lambda x: x[1], scc), align='center')
plt.xticks(range(len(scc)), map(lambda x: x[0], scc), rotation='vertical')
plt.show()


