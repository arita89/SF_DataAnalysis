from __init__ import *


state,city = "CA", "San Francisco"
SFlistings = []
#per page only 20 results, so repeat query over 57 pages
numpages = 58
for i in tqdm(range(1,numpages)):
    page_listings = get_listings(state,city, headers,page = i, items = 20)
    try:
        SFlistings.extend(page_listings)
    except Exception as e:
        print (e)
    time.sleep(1)

# store as pandas dataframe 
df = pd.DataFrame(SFlistings)
df.to_pickle("../data/df_SFlistings.pkl")