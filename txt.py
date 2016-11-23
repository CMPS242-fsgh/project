import loader

d = loader.DataLoader()
f = open('data/yelp.csv', 'w')
c = 0
for business in d.alldata():
    c += 1
    f.write(business.business_id)
    f.write('\t')
    f.write(','.join(business.categories).encode('utf-8'))
    f.write('\t')
    f.write(''.join(business.reviews).replace('\t', ' ').replace('\n', ' ').encode('utf-8'))
    f.write('\n')
    if c % 1000 == 0:
        print c, '/', 85901
