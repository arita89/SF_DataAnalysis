from __init__ import *

Socrataclient = Socrata("data.sfgov.org",
                        SocrataToken,
                        username=myemail,
                        password=mypsw)


## scrap 2003-mid2018 ##
results_20032018 = Socrataclient.get("tmnf-yvry", limit=2129525)
df_results_20032018 = pd.DataFrame.from_records(results_20032018)
## first cleaning 
# selecting only columns that might be of use, in this case from 0 to 13
df_results_20032018 = df_results_20032018[df_results_20032018.columns[0:14]]
# add columns "Date" ,"hourOfDay" and "minuteOfDay"
df_results_20032018 = addTime(df_results_20032018, 'time')
df_results_20032018 = addDate(df_results_20032018, 'date')
df_results_20032018 = df_results_20032018[df_results_20032018.columns[:-1]] # dropping redundant column
#sorting 
df_results_20032018.sort_values(by=['Date',"hourOfDay","minuteOfDay"], inplace=True, ascending=True)
#reset index
df_results_20032018.reset_index(inplace=True,drop=True)
#store
df_results_20032018.to_pickle("../data/df_clean_results_20032018_raw.pkl")


##scrap 2018-2020
results_20182020 = client.get("wg3w-h783", limit=417114)
df_results_20182020 = pd.DataFrame.from_records(results_20182020)
## first cleaning 
df_results_20182020 = addTime(df_results_20182020, 'incident_time')
df_results_20182020 = addDate(df_results_20182020, 'date')
# selecting only columns that might be of use
df_results_20182020 = df_results_20182020[['Date','hourOfDay','minuteOfDay','incident_year',
                                           'incident_day_of_week','report_datetime', 'row_id',
                                           'incident_id', 'incident_number', 'cad_number', 'report_type_code',
                                           'report_type_description', 'incident_code', 'incident_category',
                                           'incident_subcategory', 'incident_description', 'resolution',
                                           'intersection', 'cnn', 'police_district', 'analysis_neighborhood',
                                           'supervisor_district', 'latitude', 'longitude'
                                          ]]
#sorting 
df_results_20182020.sort_values(by=['Date',"hourOfDay","minuteOfDay"], inplace=True, ascending=True)
#reset index
df_results_20182020.reset_index(inplace=True,drop=True)
#store
df_results_20182020.to_pickle("../data/df_clean_results_20182020_raw.pkl")