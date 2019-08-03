

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
# import matplotlib as mpl
# mpl.use("Agg")
# import numpy as np
import pandas as pd
# import seaborn as sns 
# import base64
# import geopandas as gpd

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

def listpicview(request,id):

    if id == 1:
       

        return render(request,'AnalysisEngine/analysis_pic.html',{'id':1,})
    if id == 2:
       
        return render(request,'AnalysisEngine/analysis_pic.html',{'id':2,})
    if id == 3:
        
        return render(request,'AnalysisEngine/analysis_pic.html',{'id':3,})
    if id == 4:
       
        return render(request,'AnalysisEngine/analysis_pic.html',{'id':4,})

    if id == 5:
        
        return render(request,'AnalysisEngine/analysis_pic.html',{'id':5,})

    if id == 6:
        
        return render(request,'AnalysisEngine/analysis_pic.html',{'id':6,})

	



def PDFF(request,id,*args, **kwargs):
    
    template = get_template('pdf_format.html')

    all_details=Analysis.objects.get(id=id)
    title=all_details.title
    #print(id)
    #print(all_details)
    response=detailview(request,id)
    html_table=response.context_data['html_table']
    #image_base64=response.context_data['image_base64']
    #image_base64g=response.context_data['image_base64g']
    #print(html_table)
    id=id
    context = {
    'all_details': all_details ,
    'html_table': html_table ,
    'id':id,
    
     
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
       
        all_details=Analysis.objects.get(id=id)

         #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        # 'image_base64':image_base64 ,
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)


    elif (id==2):
        data=pd.read_csv("assets/tourist arrivals by age group.csv",header=1,index_col=0)
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
      
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        # 'image_base64':image_base64 ,
       
        }

        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)


    elif (id==3):
        data=pd.read_csv("assets/Economic_indicators_of_hotels.csv",header=0)
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
       
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        # 'image_base64':image_base64 ,
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)
    elif (id==4):
        data=pd.read_csv("assets/tourist_arrivals_purpose_newlook.csv",header=0 )
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        


        # getting details of ids
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        # 'image_base64':image_base64 ,
        # 'image_base64g':image_base64g,
       
        }

        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)
    elif (id==5):
        data=pd.read_csv("assets/No_tourist_industries_guides.csv")
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        # 'image_base64':image_base64 ,
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)

    elif (id==6):

        data=pd.read_csv("assets/nepal-district.csv")
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        
        # getting details of id
        all_details=Analysis.objects.get(id=id)

        #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        
       
        }
        return TemplateResponse(request,'AnalysisEngine/analysis_detail.html',context)

    else:
       pass