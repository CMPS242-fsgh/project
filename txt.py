import loader

d = loader.DataLoader()
review = open('data/reviews.csv', 'w')
categories = open('data/cat.csv', 'w')

c = 0
for business in d.alldata():
    c += 1
    categories.write(','.join(business.categories).encode('utf-8'))
    categories.write('\n')

    review.write(''.join(business.reviews).replace('\t', ' ').replace('\n', ' ').encode('utf-8'))
    review.write('\n')
    if c % 1000 == 0:
        print c, '/', 85901
