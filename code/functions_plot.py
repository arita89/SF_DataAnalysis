from functions_utilities import *

def getPerc(fulllist,onlylist):
    num = len([c for c in fulllist if c in onlylist])
    return str(round(num/len(fulllist),2)*100)

def getIntAndExc(df1_list,
                df2_list):
    shared = list(set(df1_list).intersection(df2_list))
    df1only = list(np.setdiff1d(df1_list,df2_list))
    df2only = list(np.setdiff1d(df2_list,df1_list))
    return shared,df1only,df2only 

def Venn2Plot(df1_list,
              df2_list,
              calculatedpercentage = ["","",""],
              cols = sns.color_palette(['orange', 'darkblue']),
              fontsize = 12,
              figsize = (20,10)
            ):

    shared,df1only,df2only = getIntAndExc(df1_list,df2_list)

    s1,s2 = set(df1_list), set(df2_list)
    labels = [f"DF1 {calculatedpercentage[0]}\n"+listToString(df1only), 
              f"DF2 {calculatedpercentage[1]}\n"+listToString(df2only),
              f"COMMON {calculatedpercentage[2]}\n"+listToString(shared)]
    
    plt.figure(figsize = figsize)
    v = mv.venn2_unweighted([s1, s2], ["",""], set_colors=cols)

    def label_by_id(label, ID):
        #label = v.get_label_by_id(ID)
        #label.set_x(label.get_position()[0] + 0.1)

        num = v.get_label_by_id(ID).get_text() 
        v.get_label_by_id(ID).set_text(label)
        v.get_label_by_id(ID).set_fontsize(fontsize)

    for label, ID in zip(labels, ["10", "01","11"]):
        label_by_id(label, ID)

    plt.show()

def Venn2PlotMod(df1_list,
              df2_list,
              calculatedpercentage = ["","",""],
              cols = sns.color_palette(['orange', 'darkblue']),
              fontsize = 20,
              txtfontsize = 20,
              figsize = (10,8)
            ):
    shared,df1only,df2only = getIntAndExc(df1_list,df2_list)

    s1,s2 = set(df1_list), set(df2_list)
    labels = [f"crimes uniquely\n in DF1:\n+{str(calculatedpercentage[0])} of DF1 total ", 
              f"crimes uniquely\n in DF2:\n+{calculatedpercentage[1]} of DF2 total ",
              f"Common crimes: {calculatedpercentage[2]}\n"+listToString(shared)]
    
    fig = plt.figure(figsize = figsize)
    v = mv.venn2_unweighted([s1, s2], ["",""], set_colors=cols)

    def label_by_id(label, ID):
        num = v.get_label_by_id(ID).get_text() 
        v.get_label_by_id(ID).set_text(label)
        v.get_label_by_id(ID).set_fontsize(fontsize)

    for label, ID in zip(labels, ["10", "01","11"]):
        label_by_id(label, ID)
        
    fig.text(0.275, 0.5, "DF1\n"+listToString(df1only), ha='right', va='center', fontsize=txtfontsize,
             bbox=dict(facecolor='none', edgecolor=cols[0], boxstyle='round,pad=0.5'))
    fig.text(0.745, 0.5, "DF2\n"+listToString(df2only), ha='left', va='center', fontsize=txtfontsize,
             bbox=dict(facecolor='none', edgecolor=cols[1], boxstyle='round,pad=0.5'))
    

    plt.show()
    
def hBarPlot(calculatedpercentage,calculatedpercentage_after,figsize= (10,5)):

    mydict = {'DF1': [float(calculatedpercentage[0][:-2]), float(calculatedpercentage_after [0][:-2])], 
            'DF2': [float(calculatedpercentage[1][:-2]), float(calculatedpercentage_after [1][:-2])], 
            'Common': [100-float(calculatedpercentage[2][:-2]),100-float(calculatedpercentage_after [2][:-2])]
        }
    c = []
    v = []        
    for key, val in mydict.items():
        c.append(key)
        v.append(val)
    v = np.array(v)
    plt.figure(figsize = figsize)
    plt.title ("Percentage of discarded data, before and after the mapping")
    plt.barh(range(len(c)), v[:,0],color="lightblue",label='before',alpha = 0.5)
    plt.barh(range(len(c)), v[:,1],color="green",label='after',alpha = 0.5)
    plt.ylabel("Data frames")
    plt.xlabel("% of discarded data")
    plt.yticks(range(len(c)), c)
    plt.legend(loc='best')
    plt.show()


def weatherOverview(df, #df
                    x = 'Date',
                    xlabel = None,
                    plotcol=None, #list of names of columns to be plotted
                    kind = 'bar',
                    rows= 9, columns =1,
                    figsize=(20, 35), fontsize=12,legend=False):
    #check size
    l = len(plotcol)
    if not rows*columns >= l:
        print (f"{rows} rows arent enough, will use more")
        rows = int(1+l/columns)

    # check plot col
    if plotcol is None:
        plotcol = list(df.select_dtypes(include=['float64','int64']).columns)

    # axes coordinates
    # one ax per plot               
    coordinates = [(i,j) for i in range(rows) for j in range(columns)]
    # color map
    mycmap = list(sns.color_palette("Paired",len(plotcol )).as_hex())
    #init figure
    fig, axes = plt.subplots(rows,columns,squeeze= False)

    for i,col in enumerate(plotcol):
        ## axes
        ii,jj = coordinates[i][0],coordinates[i][1]
        ax = axes[ii][jj]
        firstAx = (i == 0)
        lastAx = (i == len(plotcol)-1)
        if firstAx:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            
        # only first and last visible
        ax.axes.xaxis.set_visible(firstAx or lastAx)
        
        # title in the graph
        hv = max(list(df[col])) #highest value
        lv = min(list(df[col])) #lowest value
        ax.set_ylim([0.8*lv,1.2*hv ])
        ax.set_title("%s"%col, y=1, pad=-18,fontsize=fontsize+5)
        
        # labels 
        if xlabel is None: 
            ax.set_xlabel(str(x))
        else:
            ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        
        #plot
        df.plot(x=x, y=[col],kind = kind, ax = ax, color = mycmap[i],
                        figsize=figsize, fontsize=fontsize,legend=legend)
            
        #grid
        ax.grid(True)
    plt.minorticks_on()   
    plt.show()


def dark_Histo(df, bins,
                    figsize=(30,20), facecolor='#1d1135',
                    alpha=0.7, zorder=2, rwidth=0.9, colors='#5643fd',
                    ticks = "index",ticks_rotation = 90, ticks_c='w', ticks_fontsize=13,
                    numOnTopOfCol = True, numOnTopOfCol_rotation = 90, numOnTopOfCol_c='w', numOnTopOfCol_fontsize=13,
                    topping = 5,
                    xlabel = "bins", xlabel_c ='#13ca91', xlabel_size=20,
                    ylabel = "count", ylabel_c ='#13ca91', ylabel_size=20,
                    title = "histogram",title_fontweight ="bold",title_c='w', title_fontsize=25,
                    yaxis_c ='#5643fd' , xaxis_c ='#1d1135'
                    ):

    # https://towardsdatascience.com/histograms-with-pythons-matplotlib-b8b768da9305
    fig, ax = plt.subplots(1,figsize=figsize, facecolor=facecolor)
    ax.set_facecolor(facecolor)


    n, bins, patches =plt.hist(df.index, weights=df.counts,bins=bins, 
                               alpha=alpha, zorder=zorder, rwidth=rwidth, color=colors);

    ##grid
    plt.grid(axis='y', color=yaxis_c, lw = 0.5, alpha=alpha)
    plt.grid(axis='x', color=xaxis_c, lw = 0.5)
    ax.grid(True)

    ##title and axis labels
    plt.title(title,fontsize = title_fontsize, fontweight =title_fontweight, c=title_c);
    plt.xlabel(f'\n{xlabel}', c=xlabel_c, fontsize=xlabel_size)
    plt.ylabel(ylabel, c=ylabel_c, fontsize=ylabel_size);
    #print (f"{bins=}")
    #print (f"{n=}")

    ## x ticks
    locs = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])];
    if ticks == "index":
        labels = [ idx+1 for idx, value in enumerate(bins[:-1])] ;
    else:
        labels = df[str(ticks)]
    plt.xticks(locs,labels,rotation = ticks_rotation, c=ticks_c, fontsize=ticks_fontsize); # Set text labels.

    ## Hide the right and top spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))

    # plto on top of columns
    if numOnTopOfCol == True:
        for idx, value in enumerate(n):
            if value > 0:
                plt.text(locs[idx], value+topping, int(value), ha='center', fontsize = numOnTopOfCol_fontsize, c=numOnTopOfCol_c, rotation = numOnTopOfCol_rotation)
    
    return fig

def horHist(df, bins,
            figsize=(30,20), facecolor='#1d1135',
            alpha=0.7, zorder=4, rwidth=0.9, colors='#5643fd',
            ticks = "index",ticks_rotation = 90, ticks_c='white', ticks_fontsize=20,
            numOnTopOfCol = False, numOnTopOfCol_rotation = 90, numOnTopOfCol_c='w', numOnTopOfCol_fontsize=13,
            topping = 5,
            xlabel = "count", xlabel_c ='#13ca91', xlabel_size=20,
            ylabel = "bins", ylabel_c ='#13ca91', ylabel_size=20,
            title = "histogram",title_c='w', title_fontsize=22,
            xaxis_c ='#5643fd' , yaxis_c ='#1d1135',
            orientation='horizontal',
            addColorbar = False,
            palette = "icefire_r",
            ):

    # https://towardsdatascience.com/histograms-with-pythons-matplotlib-b8b768da9305
    fig, ax = plt.subplots(1,figsize=figsize, facecolor=facecolor)
    ax.set_facecolor(facecolor)
    cm_r =sns.color_palette(palette, as_cmap=True)

    n, bins, patches =plt.hist(df.index, weights=df.counts, bins=bins, 
                               orientation=orientation,
                               alpha=alpha, zorder=zorder, rwidth=rwidth);#, color=colors);
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ##grid
    plt.grid(axis='y', color=yaxis_c, lw = 0.5)
    plt.grid(axis='x', color=xaxis_c, lw = 0.5, alpha=alpha)
    ax.grid(True)

    ##title and axis labels
    plt.title(title,fontsize = title_fontsize, c=title_c);
    plt.xlabel(f'\n{xlabel}', c=xlabel_c, fontsize=xlabel_size)
    plt.ylabel(ylabel, c=ylabel_c, fontsize=ylabel_size);


    ## y ticks
    locs = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
    if ticks == "index":
        labels = [ idx+1 for idx, value in enumerate(bins[:-1])] 
    else:
        labels = [f"{e}: {int(n[i])}" for i,e in enumerate(df[str(ticks)])]
    ax.tick_params(colors='white', size=15,labelsize=ticks_fontsize)
    ax.set_yticklabels(labels)
    ax.set_yticks(locs)
    
    ## Hide the right and top spines
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_position(('outward', 10))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    if orientation =="vertical":
        poscax = "bottom"
    else: 
        poscax = "left"
    
    
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm_r(c))
    
    #norm = mpl.colors.Normalize(vmin=n.min(), vmax=n.max())
    if addColorbar == True:
        cax = divider.append_axes(poscax, size="2%", pad=0.05)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm_r,
                                        #norm=norm,
                                        ticks=[],
                                        orientation="vertical")
    return fig

def plotDataFrameColumns(df,kind = "bar", rows= 11, columns = 2, 
                         figsize=(20, 40), 
                         mycmap = None,
                         palette = "Paired", 
                         title = None, titlepos = -14,
                         hide_xaxis = False, hide_legend = False,
                         xlabel = "", ylabel = "",
                         xlim = (None,None), ylim = (None,None),
                         xticklabels = (None,0,14), yticklabels = (None,0,14),
                         saveGraphs = True, coordinates = [],
                         gridstate = False,
                        ):
    
    if mycmap is None: 
        mycmap = list(sns.color_palette(palette,len(df.columns)).as_hex())

    
    fig, axes = plt.subplots(rows,columns) 
    #tuple coordinates for the plots 
    if len(coordinates)!= rows*columns:
        #print (f"you gave {len(coordinates)} coordinates but {rows*columns} are needed, fixed with progressive coordinates")
        coordinates = [(i,j) for i in range(rows) for j in range(columns)]
    for i,column in enumerate(df.columns):

        ## axes
        ii,jj = coordinates[i][0],coordinates[i][1]
        ax = axes[ii][jj]
        # labels 
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ## create the plot and the title
        if titlepos < 0 :

            #make space for the title enlarging the area of the graph
            m = max(list(df[column]))
            ax.set_ylim([0,1.2*m ])
            ax.set_title("%s"%column, y=1.0, pad=-14)

            df[column].plot(kind=kind, figsize=figsize, ax = ax, color = mycmap[i])

            # use a list to name the ticks
            if xticklabels[0] is not None:
                rotationx =  xticklabels[1]
                fontsizex = xticklabels[2]
                #ax.locator_params(nbins=len(xticklabels[0]), axis='x')
                ax.set_xticks(xticklabels[0])
                ax.set_xticklabels(xticklabels[0], rotation = rotationx,fontsize=fontsizex)
            if yticklabels[0] is not None:
                rotationy =  yticklabels[1]
                fontsizey = yticklabels[2]
                ax.set_yticklabels(yticklabels[0], rotation = rotationy,fontsize=fontsizey)                

        ## to have the title on top 
        else: 
            df[column].plot(kind=kind, title ="%s"%column, figsize=(15, 20), ax = ax, color = mycmap[i])


        ## override axis limits if given
        if xlim[0] is not None and xlim[1] is not None:
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim[0] is not None and ylim[1] is not None:
            ax.set_xlim([ylim[0],ylim[1]])
            if titlepos == "inside_graph" and ylim[1] < 1.2*m:
                print (f"the given {ylim[1]=} is to small to show the title properly")

        ## showing axes or not
        if hide_xaxis == True: 
            ax.axes.xaxis.set_visible(False)
            if i >= len(df.columns)-2:
                ax.axes.xaxis.set_visible(True)         

        if hide_legend == False: 
            ax.legend()

        ax.grid(gridstate)

    plt.show()


def ploty_boxplot(dfT, title ="", palette = "Paired",
                  colors = None, 
                  h = 600, w = 1000,
                  x_dtick = 25, x_tickfont = 10,x_tickangle = -90,
                  y_dtick = 25, y_tickfont = 10,y_tickangle = 0,
                  showlegend = False
                 ):
        x_data = dfT.columns.tolist()
        y_data = [dfT[col] for col in x_data]
        if colors is None:
            colors = list(sns.color_palette(palette,len(x_data)).as_hex())

        fig = go.Figure(layout=go.Layout(height=h, width=w))
        for xd, yd, cls in zip(x_data, y_data, colors):
                fig.add_trace(go.Box(
                                        y=yd,
                                        name=xd,
                                        #boxpoints='all',
                                        #jitter=0.5,
                                        whiskerwidth=0.2,
                                        marker_color =cls,
                                        marker_size=2,
                                        line_width=1)
                                    )

        fig.update_layout( title=title,
                            yaxis=dict(
                                        autorange=True,
                                        showgrid=True,
                                        zeroline=True,
                                        dtick=y_dtick,
                                        tickfont = dict(size=y_tickfont)
                                    ),
                            xaxis=dict(
                                        tickangle = x_tickangle,
                                        tickfont = dict(size=x_tickfont)
                                    ),
                            margin=dict(
                                        l=40,
                                        r=30,
                                        b=80,
                                        t=100,
                                    ),
                            showlegend=showlegend
                        )

        fig.show()

def gifCrimes(df_fc,focus_crimes,mycmap,folder,delete_tempFiles = True,fps = 5,startfromframe=1):

    for i,crime in enumerate(focus_crimes[:]):
        # select crimes
        #selected_crimes = df_fc.category.unique()
        #mask = df_fc.category.isin(selected_crimes)
        plt.figure(i)
        title = f"{crime}"
        fig, ax = plt.subplots(figsize=(30,10))
        ax.set_xlim([0,10**4])
        ax.set_ylim([0,10**4])
        mask = df_fc.category == crime

        # grab list of lat and long
        X,Y = list(df_fc[mask].longitude.astype(float)),list(df_fc[mask].latitude.astype(float))

        #110x110 sections (how does this change the result?)
        n = 110

        # n equidistant ticks between max and min lat and long
        xedges = np.linspace(min(X),max(X),n)
        yedges = np.linspace(min(Y),max(Y),n)

        # create the 2d grid
        grid, xedges, yedges = np.histogram2d(X,Y, bins=(xedges, yedges))

        # we have a grid over SF
        # for each square of the grid we have a number of crimes happening
        # there will be a total of 110*110 numbers = 12100
        # we take the unique numbers out of that, these are the unique-counts of occurrences
        # and count how many time each repeats
        unique, counts = np.unique(grid, return_counts=True)
        
        #crime color
        crimecol = mycmap[i]
        
        #plot
        filenames = []
        for r in range(11,len(unique),1):
            filename = cleanline(f"{title}-{r}")
            
            plt.loglog(unique[:r],counts[:r],color = crimecol, label = crime)
            plt.title(f"LogLog {title} distribution ",fontsize = 23)
            plt.savefig(f"{folder}/{filename}")#, bbox_extra_artists=(lgd,), bbox_inches='tight')
            filenames.append(filename+".png")
            plt.close(fig)   
            
        
        ## build gif
        titleGif = buildGif(folder,filename,filenames,fps = fps,startfromframe=startfromframe)
        print (f"GIF built in  {titleGif}")
        
        ##Remove files
        if delete_tempFiles == True: 
            for filename in set(filenames):
                os.remove(folder +"/"+filename)

def staticPlotsCrimes(df_fc,focus_crimes,mycmap):
    fig,axes = plt.subplots(len(focus_crimes),1,squeeze= False,figsize=(10,100))

    for i,crime in enumerate(focus_crimes):
        # select crimes
        mask = df_fc.category == crime
        
        ax = axes[i][0]

        # grab list of lat and long
        X,Y = list(df_fc[mask].longitude.astype(float)),list(df_fc[mask].latitude.astype(float))

        #110x110 sections (how does this change the result?)
        n = 110

        # n equidistant ticks between max and min lat and long
        xedges = np.linspace(min(X),max(X),n)
        yedges = np.linspace(min(Y),max(Y),n)

        # create the 2d grid
        grid, xedges, yedges = np.histogram2d(X,Y, bins=(xedges, yedges))

        # we have a grid over SF
        # for each square of the grid we have a number of crimes happening
        # there will be a total of 110*110 numbers = 12100
        # we take the unique numbers out of that, these are the unique-counts of occurrences
        # and count how many time each repeats
        unique, counts = np.unique(grid, return_counts=True)

        #plot
        ax.set_title(f"{crime} distribution ")

        ax.loglog(unique,counts,color = mycmap[i])
        #ax.legend()

    plt.show

# per crime 
def plotCrimeCountSpacialDistribution(df_fc,
                                      crime,
                                      mycmap = None,
                                      mycolDic = None,
                                      gif_statement = True,
                                      delete_tempFiles_statement = True, # delete figures after making the gif
                                      fps = 5, #speed in frame per second
                                      startfromframe=1 # exclude first frame
                                      ):
    if (mycmap is None) and (mycolDic is None):
        mycmap = sns.color_palette("Paired", len(focus_crimes).as_hex())
        
    folder = create_folder_indir_with_date("../figures-Images2GIF/crimeMonthYear", title = cleanline(crime))
    
    for year in list(df_fc.year.unique()):
        for month in list(df_fc.month.unique()):

            # mask by category, year and month
            mask = (df_fc.category == crime) & (df_fc.year <= year) & (df_fc.month <= month)

            # grab list of lat and long
            X,Y = list(df_fc[mask].longitude.astype(float)),list(df_fc[mask].latitude.astype(float))

            #110x110 sections (how does this change the result?)
            n = 110

            # n equidistant ticks between max and min lat and long
            xedges = np.linspace(min(X),max(X),n)
            yedges = np.linspace(min(Y),max(Y),n)

            # create the 2d grid
            grid, xedges, yedges = np.histogram2d(X,Y, bins=(xedges, yedges))

            # we have a grid over SF
            # for each square of the grid we have a number of crimes happening
            # there will be a total of 110*110 numbers = 12100
            # we take the unique numbers out of that, these are the unique-counts of occurrences
            # and count how many time each repeats
            unique, num_loc = np.unique(grid, return_counts=True)


            #crime color
            if mycmap is not None: 
                crimecol = mycmap[i]
            elif mycolDic is not None:
                crimecol = mycolDic[crime]


            #plot
            title = cleanline(f"{crime}")
            fig, axs = plt.subplots(1,figsize=(25,10), squeeze = True)

            pad = "0"*(2-len(str(month)))
            filename = cleanline(f"{title}-{year}-{pad}{month}")
            axs.plot(unique,num_loc,color = crimecol, label = crime)
            axs.set_yscale("log")
            #axs.set_ylim([0.1,10**4])
            #axs.set_xlim([0,unique[-1]*1.1])
            #axs.set_xscale("log")
            axs.set_xlabel(f'x occurrences of crime: {title}', fontsize=20)
            axs.set_xticks(ticks = [u for i,u in enumerate(unique) if u%8 == 0])
            axs.set_xticklabels(labels = [int(u) for i,u in enumerate(unique) if u%8 == 0], rotation=90)
            axs.set_ylabel('In how many location does an x amount of crime occur? ', fontsize=20)
            axs.set_title(f"Log distribution of number of locations per x occurences of crime:{title}",fontsize = 20)

            ##save figure and name to create the gifs
            plt.savefig(f"{folder}/{filename}")
            plt.close(fig) 
    
    # retrieve all png from folder
    filenames = glob.glob(f'{folder}/*.png')
    filenames = sorted([f.split('/')[-1] for f in filenames])
    
    ## build gif
    if gif_statement == True: 
        titleGif = buildGif(folder,title,sorted(filenames),fps = fps,startfromframe=startfromframe)
        print (f"GIF built in  {titleGif}")
    else:
        titleGif = "nope!"
        
    ##Remove files
    if delete_tempFiles_statement == True: 
        for filename in set(filenames):
            os.remove(folder +"/"+filename)
            
    return titleGif