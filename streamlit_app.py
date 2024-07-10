import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
import plotly.express as px
import requests
import json

# path to data csvs
url_owid = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/hadcrut-surface-temperature-anomaly.csv'
url_git = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/Towid-co2-data.csv'
#url_merge = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/merge3.csv'
url_kaggle = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/merge.csv'
url_merge = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/merge.csv'
url_owid_continents = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/continents-according-to-our-world-in-data.csv'

def load_original_data(url):
    
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data from GitHub.")
        return None
    
owid_df = load_original_data(url_owid)
git_df = load_original_data(url_git)
kaggle_df = load_original_data(url_kaggle)
merge_df = load_original_data(url_merge)
merge_df = merge_df.drop(merge_df.columns[0], axis=1)

st.title("Climate Change Data")
st.sidebar.title("Table of contents")
pages=["Introduction", "Exploration", "DataVizualization", "Modelling", "Prediction" , "Conclusion"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
  st.write("### Introduction")

elif page == pages[1] : 
  st.header("Data Exploration")

  st.markdown(
    """We analyzed 3 publicly available data sets:"""
  )
  st.write("### OWID Dataset")
  st.markdown(
    """The OurWorldInData Dataset contains the Surface temperature anomaly per country and year.
    While the data frame contains no missing values, not all countries' temperature deviations for all years are available. Nonetheless, it complements GitData, so we decided to include it.
    Period: 1850-2017"""
  )
  st.write("Dataset")
  st.dataframe(owid_df.head(10))
  st.write("Dataset shape")
  st.write(owid_df.shape)
  st.write("Dataset describtion")
  st.dataframe(owid_df.describe())

  st.write("Percentage of NAs per Column: 0")

  st.write("### GitData Set")
  st.markdown(
    """Data on Co2 and other greenhouse gas emissions (co2, ghg, n2o, ch4) as well as gpd, population by country and year.
      This dataset was identified as highly relevant as it contained information on many relevant variables such as gross domestic product (GDP), population numbers, energy consumption levels, and greenhouse gas emissions. 
      Greenhouse gas emissions are divided into emissions in tonnes by CO2, methane-, total greenhouse gases-, and nitrious oxide and are given in absolute numbers and scaled by population.<br> 
      Moreover, contributions to CO2 emissions are subdivided into industries such as cement, consumption, energy, coal, gas, oil, trade, other industries, and land use change. While the variables in the data set were judged highly relevant, the completeness varies from ~14% of values missing for population values to over 91% for other industries. Moreover, different countries had different numbers of years of data available. For
      1
      instance, for Germany, data is available from 1792 to 2022, whereas for Serbia, only data from 1850 onwards are included. Thus, there is an additional source of missingness in the data that does not show up in the data set as NaNs.
      Period: 1750-2022"""
    )
  st.write("Dataset")
  st.dataframe(git_df.head(10))
  st.write("Dataset shape")
  st.write(git_df.shape)
  st.write("Dataset describtion")
  st.dataframe(git_df.describe())

  # Calculate the percentage of NAs per column
  na_percentages = git_df.isna().mean() * 100
  na_percentages_df = na_percentages.reset_index()
  na_percentages_df.columns = ['Column', 'Percentage of NAs']

  expand = st.expander("Percentage of NAs per Column", icon=":material/info:")
  expand.write("Percentage of NAs per Column")
  expand.write(na_percentages_df)




  st.write("### Kaggle Dataset")
  st.markdown(
    """The Kaggle data set contains temperature change per month, year and country.
        This data set was only used for visualizations and not for the modeling part. This was done because the monthly resolution did not fit the yearly GitData resolution and had limited usability as it only dates back to 1961.
        Period: 1961-2020"""
    )
  st.write("Dataset")
  st.dataframe(kaggle_df.head(10))
  st.write("Dataset shape")
  st.write(kaggle_df.shape)
  st.write("Dataset describtion")
  st.dataframe(kaggle_df.describe())

  st.write("Percentage of NAs per Column: 0")


  st.write("### Data Merge")
  st.markdown(
    """A large proportion of data contained missing values. To circumvent the missingness problem according to different start and end records, we used data from 1900 to 2017, which also captured the largest increase in CO2 and temperature. Further, we decided to exclude variables for modeling with a hard cut-off of 80% of missingness, as these were judged to contain too little information to be attributed reliably. Moreover, rows containing three missing variables or more were excluded from further analysis.
15
In conclusion, we decided to merge the GitData and OWID data sets containing data from the period from 1900 to 2017. The target variable in our analysis is temperature anomaly, and features include:
CO2, GDP, Population, Year, Country, Continent, Primaryenergyconsumption, Temperaturechangefromgreenhousegases, Temperature change from CO2, Temperature change from Methane, Temperature change from Nitric-Oxide
    """
    )
  st.write("Dataset")
  st.dataframe(merge_df.head(10))
  st.write("Dataset shape")
  st.write(merge_df.shape)
  

elif page == pages[2] : 
  st.header("Data Visualization")

  df_aggregated = merge_df.groupby(['year'])['temp_anomaly'].mean().reset_index()
  fig = px.line(df_aggregated, x='year', y='temp_anomaly', title='Average World Surface temperature anomaly 1900 - 2016')
  st.plotly_chart(fig)

  df_aggregated = merge_df.groupby(['year', 'continent'])['temp_anomaly'].mean().reset_index()
  fig = px.line(df_aggregated, x='year', y='temp_anomaly', color='continent', title='Surface temperature anomaly by continent 1900 - 2016')
  st.plotly_chart(fig)

  #st.write("#### Boxplot temp , continents")

  fig = px.box(merge_df, x='continent', y='temp_anomaly', title='Boxplot of Temperature Anomany by Continents')
  st.plotly_chart(fig)


  #st.write("#### c20 pie")
  co2_categories = git_df[['cement_co2','oil_co2','coal_co2','consumption_co2', 'flaring_co2','gas_co2','other_industry_co2', 'trade_co2']].sum()
  fig = px.pie(co2_categories, names=co2_categories.index, values=co2_categories.values, title='Distribution of CO2 Gas Emissions')
  st.plotly_chart(fig)

  #st.write("#### c20 time")

  df_aggregated = merge_df.groupby(['year', 'continent'])['co2'].sum().reset_index()
  fig = px.line(df_aggregated, x='year', y='co2', color='continent', title='CO2 Emmissions by Continent')
  st.plotly_chart(fig)

  st.write("##### Correlation Heatmap")
  
  # Calculate the correlation matrix
  merge_df_num = merge_df.drop(['country', 'continent'], axis=1) 
  corr_matrix = merge_df_num.corr()

  # Create the heatmap using plotly.express
  fig = px.imshow(corr_matrix, 
                labels=dict(color="Correlation"), 
                x=corr_matrix.columns, 
                y=corr_matrix.columns,
                color_continuous_scale='Viridis')

  st.plotly_chart(fig)

  st.write("#### Wordmap")
  file_path = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/countries.geo.json'
  response = requests.get(file_path)
  if response.status_code == 200:
    counties = response.json()
  #with open(file_path, 'r') as file:
    #counties = json.load(file)

  df_1900_2000 = owid_df.loc[(owid_df['Year'] <= 2000) & (owid_df['Year'] >= 1900)]
  fig = px.choropleth_mapbox(df_1900_2000, geojson=counties, locations='Code', color='Surface temperature anomaly',
                           color_continuous_scale="icefire",
                           range_color=(-3, +3),
                           mapbox_style="carto-positron",
                           zoom=0.5,
                           opacity=0.5,
                           labels={'unemp':'unemployment rate'}
                          )
  fig.update_layout(title="World Surface temperature anomaly 1900 -2000")
  st.plotly_chart(fig)



  
  df_2000 = owid_df.loc[(owid_df['Year'] >= 2000)]
  fig = px.choropleth_mapbox(df_2000, geojson=counties, locations='Code', color='Surface temperature anomaly',
                           color_continuous_scale="icefire",
                           range_color=(-3, +3),
                           mapbox_style="carto-positron",
                           zoom=0.5,
                           opacity=0.5,
                           labels={'unemp':'unemployment rate'}
                          )
  fig.update_layout(title="World Surface temperature anomaly 2000 and later")
  st.plotly_chart(fig)



elif page == pages[3] : 
  st.write("### Modeling")

elif page == pages[4] : 
  st.write("### Prediction")

elif page == pages[5] : 
  st.write("### Conclusion")