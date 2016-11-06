
import loader                    # the code should be in the same folder of loader.py
                                 # before using the loader, run preprocess.py to generate the database

d = loader.DataLoader()          # I just hard coded the database path, it should be at 'data/data.db'

c = 0
for restaurant in d.alldata():   # Here is the tricky part:
                                 #    alldata() does not return a list, it returns an iterable
                                 #    it only returns one restaurant's information at each step
                                 #    you can add your calculation code here

                                 #    in my test, it DOES take large amount of RAM, but will leave you some minimal RAM for system
                                 #    if we use big list, the whole system will be doomed....(believe me)
                                 #    it is very slow to enumerate all data point, I think we need work on small subset before we move on
    print(restaurant.business_id, restaurant.categories)
    print(restaurant.reviews)
    c = c+1
    #a = restaurant.business_id
    if c > 20:
        break
