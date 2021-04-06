from modules import *

# foliumtoken
# this is so easy to get that there is no need to hide
mytoken = "pk.eyJ1IjoiYXJpdGFvODkiLCJhIjoiY2tsZ3pmZTV0MWg5MjJwcDczZHV1anQ5cCJ9.aPiO2IJde1Ipk_NGTtehLw"

def refreshSanFrancisco(zoom_start=12):
    bandwMap = folium.Map([37.7749, -122.4194],
               tiles="Stamen Toner",
               API_key=mytoken,zoom_start=zoom_start)

    return bandwMap 

def addGridtoMap(m,
                 # up to down
                latmin = 37.81,
                latmax = 37.70,
                 
                # left to right
                lonmin = -122.52,
                lonmax = -122.35,
                
                color = "red",
                gridDim = 50 #[50x50]
                 
                
                 
                ):
    
    lat_interval = np.linspace(latmin,latmax,gridDim)
    lon_interval = np.linspace(lonmin,lonmax,gridDim)

    #print (f"\n{lat_interval=}")
    #print (f"\n{lon_interval=}")

    for lat in lat_interval:
        #print (f"{[[lat, lonmin],[lat, lonmax]]}")
        folium.PolyLine([[lat, lonmin],[lat, lonmax]], weight=0.5, color=color).add_to(m)

    for lon in lon_interval:
        folium.PolyLine([[latmin, lon],[latmax, lon]], weight=0.5, color=color).add_to(m)

    return m

def distance(Xi,Yi,Xc,Yc):
    return math.sqrt((Xi-Xc)**2+(Yi-Yc)**2)

def pointInPolygons(plat,plong,polygons_coord):
    point = Point(plong,plat)
    for district,polygon in polygons_coord.items():
        if (point.within(polygon) or polygon.touches(point) ):
            return district
    return "NA"

