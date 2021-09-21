import wbgapi as wb
import pandas as pd
import numpy as np
from iso3166 import countries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

#get indicators list for world bank
indicators = pd.read_csv('WDISeries.csv')
indicators_list = list(indicators['Series Code']

#read in additional data
df = pd.read_excel("domestic_energy_policy.xlsx")
df = df[df['Energy Policy']=='Y'][['Country', 'Policy Name', 'Nuclear Language', 'Section on Nuclear']]
df.fillna('N', inplace=True)

#convert country names to codes
def country_code(c):
    try:
        return countries.get(c).alpha3
    except:
        return c

df['Country'] = df['Country'].apply(country_code)
extra_countries = {'Bolivia': 'BOL', 'Chinese Taipei': np.nan, "Cote D'Ivoire": 'CIV',
                   'Curacao': 'CUW', 'Czech Republic': 'CZE', 'Iran': 'IRN', 'Laos': 'LAO', 
                   'Korea': 'KOR',  'Republic of Moldova': 'MDA','Russia': 'RUS',
                   'Slovak Republic': 'SVK','St. Lucia': 'LCA',
                   'United Kingdom': 'GBR', 'United States': 'USA', 'Venezuela': 'VEN',
                  'CZECH REPUBLIC': 'CZE', 'RUSSIA': 'RUS', 'UNITED KINGDOM': 'GBR'}
df['Country'].replace(extra_countries, inplace=True)
df.dropna(inplace=True)

op_df = pd.read_excel("domestic_energy_policy.xlsx", sheet_name = 'Operational')
op_df['Country'] = op_df['Country'].apply(country_code)
op_df = op_df.iloc[:32, :]
op_df['Country'].replace(extra_countries, inplace=True)
full_df = df.merge(op_df, on='Country', how='outer')
full_df = full_df[['Country', 'Policy Name', 'Nuclear Language', 'Section on Nuclear','total net elec capacity (MW)']]
full_df.columns = ['Country', 'Policy Name', 'Nuclear Language', 'Section on Nuclear', 'Operational Capacity(MW)']

cons_df = pd.read_excel("domestic_energy_policy.xlsx", sheet_name = 'Construction')
cons_df['Country'] = cons_df['Country'].apply(country_code)
cons_df['Country'].replace(extra_countries, inplace=True)
full_df = full_df.merge(cons_df, on='Country', how='outer')
full_df = full_df[['Country', 'Policy Name', 'Nuclear Language', 'Section on Nuclear',
       'Operational Capacity(MW)',
       'total net elec capacity (MW)']]
full_df.columns = ['Country', 'Policy Name', 'Nuclear Language', 'Section on Nuclear',
       'Operational Capacity (MW)', 'Under Construction Capacity (MW)']

shut_df = pd.read_excel("domestic_energy_policy.xlsx", sheet_name = 'Shutdown')
shut_df['Country'] = shut_df['Country'].apply(country_code)
shut_df['Country'].replace(extra_countries, inplace=True)
full_df = full_df.merge(shut_df, on='Country', how='outer')
full_df = full_df[['Country', 'Policy Name', 'Nuclear Language', 'Section on Nuclear',
       'Operational Capacity (MW)', 'Under Construction Capacity (MW)',
      'total net elec capacity (MW)']]
full_df.columns = ['Country', 'Policy Name', 'Nuclear Language', 'Section on Nuclear',
       'Operational Capacity (MW)', 'Under Construction Capacity (MW)','Shutdown Capacity (MW)']
full_df['Nuclear Language'][full_df['Country']=='SVN']='N'
full_df['Section on Nuclear'][full_df['Country']=='SVN']='N'
full_df.fillna(0, inplace=True)

#function to add a column to df from world bank api
def add_column(indicator, name, df):
    col = wb.data.DataFrame([indicator], columns = 'series' )
    col.reset_index(inplace=True)
    col['time'] = col['time'].apply(lambda x: int(x.lstrip('YR')))
    yrs = col.dropna().groupby('economy')['time'].max().reset_index()
    final = col.merge(yrs, on=['economy', 'time'])[['economy', indicator]]
    final.columns = ['Country', name]
    
    return df.merge(final, on='Country', how='outer')

for indic, name in indicators_list:
    full_df = add_column(indic, name, full_df)
