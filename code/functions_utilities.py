from modules import *

def listToString(s,str1= "\n"):  
    return (str1.join(s))
        
def clean_string(mystring, lowercase = True):
    """
    replaces " " with "_" and " - " with "-"
    makes strings lowercase
    """
    mystring = mystring.replace(" - ", "-")
    mystring = mystring.replace(" ", "_")
    mystring = mystring.replace("as_of", "-") 
    mystring.translate(str.maketrans('', '', string.punctuation))
    
    if lowercase == True:
        mystring = mystring.lower()
    return mystring 

def cleanline(line):
    
    line = line.replace('\r', '')
    line = line.replace("\\", '_')
    line = line.replace("/", '_')
    line.strip('\n')
    
    return line 

def removePunctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def mytimestamp(sep= "-"):
    now = datetime.datetime.now()
    time_obj = now.strftime("%Hh%Mm%Ss")+sep
    return time_obj

def get_date(sep= "-"):
    date_obj= datetime.date.today()
    date_obj = date_obj.strftime("%Y%m%d")+sep
    return date_obj

def create_folder_indir_with_date(dir, title = "ImgStream",printstat = True):
    folder = f'{dir}/{title}'
    if not os.path.exists(folder):
        if printstat == True:
            print ( f"{folder} folder created")
        os.makedirs(folder)
    else:
        if printstat == True:
            print(f"{folder} folder exists already")
    return folder

def buildGif (folder,title,filenames,fps=55,startfromframe = 0):
    """
    titleGif = buildGif (folder,title,filenames,fps=55)
    folder = folder where are the current images to put togheter 
    title = name of gif
    filenames = list of names of images
    """
    
    titleGif = folder +"/" +title+'.gif'
    with imageio.get_writer(titleGif, mode='I',fps = fps) as writer:
        for filename in filenames[startfromframe:]:
            #print (filename)
            image = imageio.imread(folder +"/"+filename)
            writer.append_data(image)
            if filename == filenames[-1]:
                for i in range(10):
                    writer.append_data(image)
    
    return titleGif

def date2string(date, delta= None, formatting = '%d/%m/%Y'):
    """
    from np.datetime to str in the requested formatting
    if want to add days, input delta
    """
    if delta is None: 
        return pd.to_datetime(str(date)).strftime(formatting)
    else:
        return (pd.to_datetime(str(date))+datetime.timedelta(days=delta)).strftime(formatting) 

def makeListSameLength(listOfLists):
    allLengths = [len(thislist) for thislist in listOfLists]
    maxL = max(allLengths)
    return [extendListToN(list(thislist),maxL) for thislist in listOfLists]
def setLabels(listPoints, labels):
    # set labels straight
    llp = len(listPoints)
    ll = len(labels)
    diff = llp-ll
    if diff > 0:
        extendedlabels = [ f"fun_{i}" for i in range(ll, llp)]
        labels.extend(extendedlabels)
    return labels
def maxListOfLists(listOfList):
    return max([max(thislist) for thislist in listOfList])

def minListOfLists(listOfList):
    return min([min(thislist) for thislist in listOfList])

def createGIF(listPoints,
              folder,
              figsize=(20, 40), marker = "-",colors = None,
              delete_tempFiles = True,
              title = "gif", titlepos = 0,
              hide_xaxis = False, hide_legend = False,
              gridstate = False, fps = 10,
              xlabel = "", ylabel = "",labels = [],
              xlim = (None,1), ylim = (None,1),
              xticklabels = (None,0,14), yticklabels = (None,0,14)
             ):
    
    
    """
    listPoints[Y,Z...]
    delete_tempFiles: if true deletes the images
    
    title: title on top of graph
    titlepos: if negative, inside the graph. if positive outside
    fps:speed of frames in gif
    labels:custom labels for legend 
    xlim,ylim:limits of the plot
    

    """
    numPoints = len(listPoints[0])
    numLists = len(listPoints)
    
    ## create folder to save images and gif
    #folder = create_folder_indir_with_date(get_date(), title = "Images2GIF")
    
    ## if the list of points have different legths, need to be adjusted
    # the shorter lists are extended with their latest value
    if len({len(i) for i in listPoints}) != 1:
        print ("given lists of points have different lengths, adjusted")
        listPoints = makeListSameLength(listPoints)
        
    ## create x-axis
    X = np.linspace(0, len(listPoints[0]), len(listPoints[0]))
    
    ## set labels
    # if fewer labels are given than needed additional generic ones are added
    labels = setLabels(listPoints, labels)

    ## create images and store them 
    filenames = []
    #for iteration point
    for i in range(0,numPoints):
        ##plot allpoints from different series
        for j in range(0,numLists):
            plt.figure(figsize=figsize);
            plt.close()
            plt.plot(X[:i], listPoints[j][:i], 
                     marker, 
                     color=colors[j],
                     label=labels[j]);
            plt.xlabel = "Time"
            plt.ylabel = f"{ylabel}"
        
        ## set the boundaries of the plot
        if ylim[0]is None: 
            plt.ylim(minListOfLists(listPoints)-1, maxListOfLists(listPoints)*1.1)
        else:
            plt.ylim(ylim[0], ylim[1])
            
        if xlim[0]is not None:
            plt.ylim(xlim[0], xlim[1])
        
         
        plt.title(f"{title}", y=1.0, pad=titlepos)
        #plt.xlabel("iterations")
        #plt.xticks(X[:i],labels[:i], rotation = 90,fontsize=10)
        
        plt.xticks([X[i] for i in range(0,len(X[:i]),int(len(X[:i])/5)+1)],
                   [labels[i] for i in range(0,len(labels[:i]),int(len(labels[:i])/5)+1)], 
                   rotation = 90,fontsize=10)

        
        #plt.xlabel = "Time"
        #plt.ylabel = f"{ylabel}"
        plt.grid(gridstate)
        
        ## create legend 
        if hide_legend == False: 
            lgd = plt.legend(loc='upper right', 
                             #bbox_to_anchor=(1.05, 1.05),
                             ncol=1, 
                             fancybox=True, 
                             shadow=True)

        # create file name and append it to a list
        filename = f'{title}{i}.png'
        filenames.append(filename)

        # save frame
        plt.savefig(folder +"/"+filename)
        plt.close()
    
    ## build gif
    titleGif = buildGif(folder,title,filenames,fps)
    print (f"GIF built in  {titleGif}")
    
    ##Remove files
    if delete_tempFiles == True: 
        for filename in set(filenames):
            os.remove(folder +"/"+filename)

    return titleGif
