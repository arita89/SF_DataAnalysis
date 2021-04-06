from modules import *
from tokens import *



def overview(df):
    # check for Nan values
    checkNan = df.isnull().values.any()
    idx, idy = np.where(df.isna())
    nanPos = None
    print (f"There are Nan values: {checkNan}" )
    if checkNan:
        nanPos = np.column_stack([df.index[idx], df.columns[idy]])
        print (nanPos)
    print ()
    # statistics
    print ("STATISTICS")
    print (df.describe())
    print()
    #series types
    print ("SERIE TYPES")
    print (df.dtypes)
    return nanPos

def resetColumns(df,frompos = -1,insertAt=1):
    columnNames = df.columns.tolist()
    columnNames.insert(insertAt,columnNames.pop(frompos))
    df = df[columnNames]
    return df

def addDate(df, columnWithDate,insertAt=0):
    # adding colum with Date in format datetime
    df['Date'] = pd.to_datetime(df[columnWithDate])
    #sort columns
    df = resetColumns(df,frompos = -1,insertAt=insertAt)
    return df

def addTime(df, columnWithTime,insertAt=0):
    # adding colum with Date in format datetime
    df['Time'] = pd.to_datetime(df[columnWithTime])

    df['minuteOfDay'] = df['Time'].dt.minute
    df = resetColumns(df,frompos = -1,insertAt=insertAt)

    df['hourOfDay'] = df['Time'].dt.hour
    df = resetColumns(df,frompos = -1,insertAt=insertAt)

    return df


def formattingDF(df,insertAt = 1):

    if not 'numDayOfWeek' in df.columns:        
        df['numDayOfWeek']= df["dayofweek"].replace(daytoInt)
        df = resetColumns(df)
    
    if not'hourOfWeek' in df.columns:
         #create a hourOfWeek column
        df['hourOfWeek'] = df["numDayOfWeek"]*24+df["hourOfDay"]
        df = resetColumns(df)

    if not 'month' in df.columns:
        #create a month column
        df['month'] = df['date'].dt.month
        df = resetColumns(df)

    if not'year' in df.columns:
        #create a year column
        df['year'] = df['date'].dt.year
        df = resetColumns(df)
    
    #print (df.head(5))
    return df

def maskDfbyDateAndTime(df,start_date, end_date,start_h,end_h):
    mask = (df['Date'] >= start_date) & (df['Date'] < end_date) & (df['hourOfDay'] >= start_h) & (df['hourOfDay'] < end_h) 
    df_masked = df.loc[mask].sort_values(by=['date',"hourOfDay","minuteOfDay"], inplace=False, ascending=True)
    return df_masked

def maskDfbyDate(df,start_date, end_date,date = "Date"):
    mask = (df[date] >= start_date) & (df[date] < end_date)
    df_masked = df.loc[mask].sort_values(by=[date,"hourOfDay"], inplace=False, ascending=True)
    return df_masked

def maskDfbyCat(df,listcat):
    mask = df['category'].isin(listcat)
    df_masked = df.loc[mask].sort_values(by=['date',"hourOfDay"], inplace=False, ascending=True)
    return df_masked

def maskDfbyDateCat(df,crimes,startdate,enddate,date="date"):
    df1 = maskDfbyCat(df,crimes)
    return  maskDfbyDate (df1,startdate,enddate,date)

def makeColLowerCase(df,listCol):
    for col in listCol:
        df[col] = df[col].str.lower()
    return df