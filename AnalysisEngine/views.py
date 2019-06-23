

# Standard library imports
from __future__ import unicode_literals
import io

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
import geopandas as gpd

# Local application imports
from .models import Analysis
from .utils import render_to_pdf

# Create your views here.

def analysislist(request):

     all_notifications_list=Analysis.objects.order_by('created_at')[:20]
     context = {
     'all_notifications_list':all_notifications_list
     }

     return render(request,'AnalysisEngine/analysis_list.html',context)


	



def PDFF(request,id,*args, **kwargs):
    
    template = get_template('pdf_format.html')

    all_details=Analysis.objects.get(id=id)
    title=all_details.title
    #print(id)
    #print(all_details)
    response=detailview(request,id)
    html_table=response.context_data['html_table']
    image_base64=response.context_data['image_base64']
    #image_base64g=response.context_data['image_base64g']
    #print(html_table)
   
    context = {
    'all_details': all_details ,
    'html_table': html_table ,
    'image_base64': image_base64 ,
    
     
    } 
    html = template.render(context)
    pdf = render_to_pdf('pdf_format.html', context)
    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = title+".pdf"
        content =" inline; filename=%s "%(filename)
        download = request.GET.get("download")
        if download:
                content = "attachment; filename=%s" %(filename)
        response['Content-Disposition'] = content
        return response

    return HttpResponse("Not found")

    return redirect(analysislist)

   


def detailview(request,id):

    if (id==1):
        #data collecting...converting dataset to html....
        data=pd.read_csv("assets/earthquake2072_effect_on_tourism.csv",header=0)
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        #data plotting/visualizing........
        data.pivot(index='Subsector', columns='Disaster effect', values='value( NPR Million)').plot(kind='bar')
        
        #data.plot()

        #storing plots in bytes
        f = io.BytesIO()
        mpl.pyplot.savefig(f, format="png", dpi=600,bbox_inches='tight')
        image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        f.close()
        mpl.pyplot.clf()
        # getting details of id
        all_details=Analysis.objects.get(id=id)

         #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'image_base64':image_base64 ,
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)


    elif (id==2):
        data=pd.read_csv("assets/tourist arrivals by age group.csv",header=1,index_col=0)
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        #data plotting/visualizing........
        
        data.iloc[1].plot.bar()
        

        #storing plots in bytes
        f = io.BytesIO()
        mpl.pyplot.savefig(f, format="png", dpi=600,bbox_inches='tight')
        
        image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        f.close()
        mpl.pyplot.clf()
        # getting details of id
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'image_base64':image_base64 ,
       
        }

        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)


    elif (id==3):
        data=pd.read_csv("assets/Economic_indicators_of_hotels.csv",header=0)
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        #data plotting/visualizing........
        
        #data.plot()
        df1=data[['Economic Indicators','Fiscal Year','val']]
        
        heatmap1_data = pd.pivot_table(df1, values='val',index=['Economic Indicators'],columns='Fiscal Year')
        sns_plot=sns.heatmap(heatmap1_data, cmap="YlGnBu")
        #fig = sns_plot.get_figure()
        
        #storing plots in bytes
        f = io.BytesIO()
        #fig.savefig(f, format="png", dpi=600,bbox_inches='tight')
        mpl.pyplot.savefig(f, format="png", dpi=800,bbox_inches='tight')
        image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        f.close()
        mpl.pyplot.clf()
        # getting details of id
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'image_base64':image_base64 ,
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)
    elif (id==4):
        data=pd.read_csv("assets/tourist_arrivals_purpose_newlook.csv",header=0 )
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        #data plotting/visualizing........
        
        data.pivot(index='year', columns='purposes', values='Arrivals').plot(kind='bar')

        f = io.BytesIO()
        mpl.pyplot.savefig(f, format="png", dpi=600,bbox_inches='tight')
        image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        f.close()
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
        mpl.pyplot.show()
        """
        Now the redirect into the  BytesIO object >>>
        """
        
        g = io.BytesIO()          
        mpl.pyplot.savefig(g, format="png", facecolor=(0.95,0.95,0.95))
        mpl.pyplot.clf()
        image_base64g= base64.b64encode(g.getvalue()).decode('utf-8').replace('\n', '')
        g.close()


        # getting details of id
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'image_base64':image_base64 ,
        'image_base64g':image_base64g,
       
        }

        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)
    elif (id==5):
        data=pd.read_csv("assets/No_tourist_industries_guides.csv")
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        #data plotting/visualizing........
        
        #data.plot()
        df1=data[['Industries/guides','year','numbers']]
        #print(df1)
        heatmap1_data = pd.pivot_table(df1, values='numbers',index=['Industries/guides'],columns='year')
        sns.heatmap(heatmap1_data, cmap="YlGnBu")

        #storing plots in bytes
        f = io.BytesIO()
        mpl.pyplot.savefig(f, format="png", dpi=600,bbox_inches='tight')
        image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        f.close()
        mpl.pyplot.clf()
        # getting details of id
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'image_base64':image_base64 ,
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)

    elif (id==6):
        data=pd.read_csv("F:\\ANACONDAA\\input\\nepal-district.csv")
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        #data plotting/visualizing........
        
        #Attribute Filtering
        df2=data[['District','Zones','Development Regions','Tourist places']]
        
        #read shape file

        fp="F:\\ANACONDAA\\input\\NepalMaps-master\\NepalMaps-master\\baselayers\\NPL_adm\\NPL_adm3.shp"

        map_df = gpd.read_file(fp)

        #joining file

        merged = map_df.set_index('NAME_3').join(df2.set_index('District'))

        variable= 'Tourist places' #plotting data 

        vmin, vmax = 1, 15  #data min - max values

        fig, ax = mpl.pyplot.subplots(1, figsize=(15, 7)) #number of figure and size axis

        #plotting map

        merged.plot(column = variable, cmap='Blues', linewidth = 0.8,ax=ax, edgecolor = '0.8')

        ax.axis('off')
        ax.set_title('Tourist Attraction Places in Nepal', fontdict={'fontsize':'25', 'fontweight':'3'})

        # Create colorbar as a legend

        sml = mpl.pyplot.cm.ScalarMappable(cmap='Blues', norm=mpl.pyplot.Normalize(vmin=vmin, vmax=vmax))

        # empty array for the data range

        sml._A = []

        # add the colorbar to the figure

        cbar = fig.colorbar(sml)

        #storing plots in bytes
        f = io.BytesIO()
        mpl.pyplot.savefig(f, format="png", dpi=600,bbox_inches='tight')
        image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        f.close()
        mpl.pyplot.clf()
        # getting details of id
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'image_base64':image_base64 ,
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)


    else:
       pass   