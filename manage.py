#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# Standard library imports
import io
#from __future__ import unicode_literals


#Django default imports
from django.shortcuts import render , redirect
from django.views.generic import View
from django.http import HttpRequest , HttpResponse ,request
from django.template.loader import get_template
from django.template.response import TemplateResponse
from django.template.loader import render_to_string

# Third party imports
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns 
import base64
#import geopandas as gpd

def main():

    def run_visualizations():


        #earthquake2072_effect_on_tourism
        data=pd.read_csv("assets/earthquake2072_effect_on_tourism.csv",header=0)
        data.pivot(index='Subsector', columns='Disaster effect', values='value( NPR Million)').plot(kind='bar')
        mpl.pyplot.savefig('AnalysisEngine/static/img/id1.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #tourist arrivals by age group
        data=pd.read_csv("assets/tourist arrivals by age group.csv",header=1,index_col=0)      
        data.iloc[1].plot.bar()
        mpl.pyplot.savefig('AnalysisEngine/static/img/id2.png',dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #Economic_indicators_of_hotels.csv
        data=pd.read_csv("assets/Economic_indicators_of_hotels.csv",header=0)       
        df1=data[['Economic Indicators','Fiscal Year','val']]
        heatmap1_data = pd.pivot_table(df1, values='val',index=['Economic Indicators'],columns='Fiscal Year')
        sns_plot=sns.heatmap(heatmap1_data, cmap="YlGnBu")
        mpl.pyplot.savefig('AnalysisEngine/static/img/id3.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #tourist_arrivals_purpose_newlook
        data=pd.read_csv("assets/tourist_arrivals_purpose_newlook.csv",header=0 )
        data.pivot(index='year', columns='purposes', values='Arrivals').plot(kind='bar')
        mpl.pyplot.savefig('AnalysisEngine/static/img/id4.png')
        mpl.pyplot.clf()

        mpl.pyplot.axhline(0, color='k')
        #Data to plot for piechart1(tourist arrivals purpose of visits)
        labels = 'holiday pleasure', 'trekking and mountaineering', 'business', 'pilgrimage','official','conference','others'
        sizes = [489451,66490,24322,82830,21310,12801,55797] # of latest year 2017 in tourist arrivals by purpose
        colors = ['gold', 'green', 'lightcoral', 'lightskyblue','blue','red','purple']
        patches, texts = mpl.pyplot.pie(sizes, colors=colors, shadow=True, startangle=90)
        mpl.pyplot.legend(patches, labels, loc="best")
        mpl.pyplot.axis('equal')
        mpl.pyplot.tight_layout()
    
        mpl.pyplot.savefig('AnalysisEngine/static/img/id44.png')
        mpl.pyplot.clf()

        #No_tourist_industries_guides
        data=pd.read_csv("assets/No_tourist_industries_guides.csv")
        df1=data[['Industries/guides','year','numbers']]
        heatmap1_data = pd.pivot_table(df1, values='numbers',index=['Industries/guides'],columns='year')
        sns.heatmap(heatmap1_data, cmap="YlGnBu")
        mpl.pyplot.savefig('AnalysisEngine/static/img/id5.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #No. of tourists destinations distribution map


        # data=pd.read_csv("C:\\Users\\Administrator\\Desktop\\TAwithUIModified-master\\TAwithUIModified-master\\assests\\nepal-district.csv")


        # df2=data[['District','Zones','Development Regions','Tourist places']]
        
        # #read shape file

        # fp="NepalMaps-master\\baselayers\\NPL_adm\\NPL_adm3.shp"

        # map_df = gpd.read_file(fp)

        # #joining file

        # merged = map_df.set_index('NAME_3').join(df2.set_index('District'))

        # variable= 'Tourist places' #plotting data 

        # vmin, vmax = 1, 15  #data min - max values

        # fig, ax = mpl.pyplot.subplots(1, figsize=(15, 7)) #number of figure and size axis

        # #plotting map

        # merged.plot(column = variable, cmap='Blues', linewidth = 0.8,ax=ax, edgecolor = '0.8')

        # ax.axis('off')
        # ax.set_title('Tourist Attraction Places in Nepal', fontdict={'fontsize':'25', 'fontweight':'3'})

        # # Create colorbar as a legend

        # sml = mpl.pyplot.cm.ScalarMappable(cmap='Blues', norm=mpl.pyplot.Normalize(vmin=vmin, vmax=vmax))

        # # empty array for the data range

        # sml._A = []

        # # add the colorbar to the figure

        # cbar = fig.colorbar(sml)

        # #storing plots in bytes
        
        # mpl.pyplot.savefig('AnalysisEngine/static/img/id6.png', dpi=600,bbox_inches='tight')
    
        # mpl.pyplot.clf()

        # #Gross foreign exchange earning from tourism
        # data=pd.read_csv("C:\\Users\\Administrator\\Desktop\\TAwithUIModified-master\\TAwithUIModified-master\\assests\\gross foreign exchange earning from tourism.csv",header=0,index_col=0)
        # data.plot()
       
        # mpl.pyplot.savefig('AnalysisEngine/static/img/pid1.png',dpi=600,bbox_inches='tight')
        
        # mpl.pyplot.clf()




    run_visualizations()
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TourismAnalytics.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
