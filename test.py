
import loader

d = loader.DataLoader()
from pprint import pprint
c = 0
for i in d.alldata():
    pprint((i.business_id, i.categories))
    c = c+1

    if c > 20:
        break
