import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
import pandas as pd
import numpy as np

def data_import():
    EU_countries=['Austria','Belgium','Bulgaria','Croatia','Czechia','Denmark','Estonia','Finland','France',
                'Germany','Greece','Hungary','Italy','Latvia','Lithuania','Luxembourg',
                'Netherlands','Norway','Poland','Portugal','Romania','Serbia',
                'Slovakia','Slovenia','Spain','Sweden','Switzerland']

    eu_dfs=[]
    for current_country in EU_countries:
        current_country_df=pd.read_csv('../data/data_import/'+current_country+'.csv',sep=',')
        current_country_df=current_country_df.set_index('Datetime (UTC)')
        current_country_df=current_country_df.drop(['Country','ISO3 Code','Datetime (Local)'],axis=1)
        current_country_df.columns=[current_country+' '+current_country_df.columns[0].split(' ')[1]]
        eu_dfs.append(current_country_df)
        
    eu_df=eu_dfs[0].join(eu_dfs[1],on='Datetime (UTC)')
    for eu_df_country in eu_dfs[2:]:
        eu_df=eu_df.join(eu_df_country,on='Datetime (UTC)')
    
    eu_df=eu_df.iloc[-5000:,:]
    eu_df.index=np.array([datetime.strptime(eu_df.index[date_idx].split(':')[0],'%Y-%m-%d %H') for date_idx in range(0,len(eu_df.index))])
    eu_df['date']=eu_df.index

    #Define Subsets:
    subsets=['Train_Set']*2000
    subsets.extend(['Test_Set1']*500)
    subsets.extend(['Test_Set2']*500)
    subsets.extend(['Test_Set3']*500)
    subsets.extend(['Test_Set4']*500)
    subsets.extend(['Test_Set5']*500)
    subsets.extend(['Test_Set6']*500)
    eu_df['Subset']=np.array(subsets)

    return eu_df, subsets