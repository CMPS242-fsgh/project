import loader, sys
from pprint import pprint

out_life = sys.argv[1]
fp = open(out_file, "w")

if __name__ == '__main__':
	d = loader.DataLoader()
	result = d.first_100()
	for i in result:
		categories = i.categories
		reviews = i.reviews.replace("\n", "")
		#pprint((i.business_id, i.categories))
		fp.write(str(categories)+"\t"+reviews.encode('utf-8')+"\n")

fp.close()
