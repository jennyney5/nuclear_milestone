# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:53:33 2021

@author: melan
"""

import pandas as pd


def clean(input_file, input_file_long):
  
    # Read in input files
    df = pd.read_csv(input_file,index_col=False)
    df_long = pd.read_csv(input_file_long,index_col=False)
    df = df.iloc[:,1:]
    df_long = df_long.iloc[:,1:]

    # Only use certain columns
    df = df[['Country', 'Nuclear Language', 'Section on Nuclear',
       'Operational Capacity (MW)', 'Under Construction Capacity (MW)',
       'Shutdown Capacity (MW)', 'Poverty Headcount Ratio % Pop', 
       'Electric power consump kwh percap',
       'Population Growth %annual', 'Total Population', 
        'Total Reserves', 'Gross Debt Position %GDP',
       'Export Goods Services %GDP', 'Import Goods Services %GDP',
       'Energy Intensity Level of Primary Energy mj per $',
       'Investment in energy with private participation']]

    
    long_cols = df_long[['Country', 'Current account balance (BoP, current US$)', 'GDP (current US$)', 'GDP growth (annual %)', 
                 'Present value of external debt (current US$)', 'CO2 intensity (kg per kg of oil equivalent energy use)',
                 'CO2 emissions (metric tons per capita)',  'Electricity production from renewable sources, excluding hydroelectric (kWh)',
                 'Access to electricity (% of population)', 'Public private partnerships investment in energy (current US$)',
                 'Energy imports, net (% of energy use)','Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
                 'Population density (people per sq. km of land area)',    'Manufacturing, value added (% of GDP)',
                 'Industry, value added (% of GDP)','Services, value added (% of GDP)','Urban population (% of total population)',]]
    
    # drop duplicates and prepare for merge
    df = df.drop_duplicates()
    long_cols = long_cols.drop_duplicates()
    df = df.merge(long_cols,how='left', left_on = 'Country',right_on = 'Country')

    # Create target column
    df['Nuclear'] = 0
    df['Nuclear'][(
        (df['Operational Capacity (MW)'] > 0 )|
        (df['Under Construction Capacity (MW)'] > 0 )|
        (df['Shutdown Capacity (MW)'] > 0)
        )]=1

    df_long['Nuclear'] = 0
    df_long['Nuclear'][(
        (df_long['Operational Capacity (MW)'] > 0 )|
        (df_long['Under Construction Capacity (MW)'] > 0 )|
        (df_long['Shutdown Capacity (MW)'] > 0)
        )]=1

    # Move target columns to the end of the df
    target_cols = ['Nuclear Language','Section on Nuclear', 'Operational Capacity (MW)','Under Construction Capacity (MW)', 'Shutdown Capacity (MW)', 'Nuclear']
    new_columns = df.columns.drop(target_cols).tolist() + target_cols
    new_columns_long = df_long.columns.drop(target_cols).tolist() + target_cols
    df = df[new_columns]
    df_long = df_long[new_columns_long]
    
    # drop countries that are not in both datasets
    country = list(df['Country'])
    country_long = list(df_long['Country'])
    dif = set(country_long).difference(set(country))
    df_long = df_long[~df_long['Country'].isin(dif)]
    
    # remove columns where null values are > 5%
    count_country = len(df)
    
    null_cols = pd.DataFrame(data=df.isnull().sum(axis = 0)/count_country*100)
    cols_remove = list(null_cols[null_cols[0]>5].index)    
    cols_keep = df.columns.drop(cols_remove).tolist()
    df = df[cols_keep]
    df = df.sort_values('Country', ascending=True).reset_index(drop=True)
    
    null_cols_long = pd.DataFrame(data=df_long.isnull().sum(axis = 0)/count_country*100)
    cols_remove_long = list(null_cols_long[null_cols_long[0]>5].index)
    cols_keep_long = df_long.columns.drop(cols_remove_long).tolist()
    df_long = df_long[cols_keep_long]
    df_long = df_long.sort_values('Country', ascending=True).reset_index(drop=True)
    
    return df, df_long


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file', help='the raw datafile of nuclear indicators (CSV)')
    parser.add_argument(
        'input_file_long', help='the long datafile for all indicators (CSV)')
    parser.add_argument(
        'output_file', help='the cleaned datafile (CSV)')
    parser.add_argument(
        'output_file_long', help='the cleaned long datafile (CSV)')
    args = parser.parse_args()

    clean,clean_long = clean(args.input_file, args.input_file_long)
    clean.to_csv(args.output_file, index=False)
    clean_long.to_csv(args.output_file_long, index=False)