import loader, sys
from pprint import pprint

#out_life = sys.argv[1]
fp = open("yelp_data.pizza", "w")

if __name__ == '__main__':
	d = loader.DataLoader()
	result = d.first_100()
	#count = 0
	for i in result:
		#count += 1
		categories = i.categories
		reviews = i.reviews.replace("\n", "")
		#if(count >70):
		#		print(str(categories)+"\n")
		#pprint((i.business_id, i.categories))
		fp.write(str(categories)+"\t"+reviews.encode('utf-8')+"\n")
fp.close()
