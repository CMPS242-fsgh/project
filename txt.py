import loader

d = loader.DataLoader()
review = open('data/reviews.csv', 'w')
categories = open('data/cat.csv', 'w')

c = 0
dcat = {}

for business in d.alldata():
    c += 1
    for cat in business.categories:
        if cat in dcat:
            dcat[cat] = dcat[cat] + 1
        else:
            dcat[cat] = 1

    categories.write(','.join(business.categories).encode('utf-8'))
    categories.write('\n')

    review.write(''.join(business.reviews).replace('\t', ' ').replace('\n', ' ').encode('utf-8'))
    review.write('\n')
    if c % 1000 == 0:
        print c, '/', 85901

f = open('data/label_rank.csv', 'w')
lst = [(v,k) for k,v in dcat.items()]
for _, cat in sorted(lst, reverse=True):
    f.write(cat.strip())
    f.write('\n')

f.close()
