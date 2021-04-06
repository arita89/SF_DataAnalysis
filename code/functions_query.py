from modules import *

# NOT USED CLEANUP
def getPage(url):
	''' returns a soup object that contains all the information 
	of a certain webpage'''
	result = requests.get(url)
	content = result.content
	return BeautifulSoup(content, features = "lxml")

# NOT USED CLEANUP
def getRoomClasses(soupPage):
	''' This function returns all the listings that can 
	be found on the page in a list.'''
	rooms = soupPage.findAll("div", {"class": "_8ssblpx"})
	result = []
	for room in rooms:
		result.append(room)
	return result


def get_city_overview(state,city, headers):
  url = f"https://mashvisor-api.p.rapidapi.com/trends/summary/{state}/{city}"
  print (f"{url=}")
  #querystring = {"page":"1","items":"10","state":state}
  
  response = requests.request("GET", url, headers=headers)#, params=querystring)
  print (response.status_code)
  print (response.text)
  if(200 == response.status_code):
    data = json.loads(response.text)
    
    if(data["status"] == "success"):
      return data["content"]
  return None

def get_listings(state,city, headers,page = 1, items = 10):
    url = "https://mashvisor-api.p.rapidapi.com/airbnb-property/active-listings"
    #print (f"{url=}")
    querystring = {"state":state,"city":city,"page":page,"items":items}
        
    response = requests.request("GET", url, headers=headers , params=querystring)
    #print (response.status_code)
    #print (response.text)
    if(200 == response.status_code):
        data = json.loads(response.text)
                 
        if(data["status"] == "success"):
            return data["content"]["properties"]
    else:
        return None