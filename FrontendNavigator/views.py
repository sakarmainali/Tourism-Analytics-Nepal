#from __future__ import unicode_literals
from django.shortcuts import render , redirect
from django.views.generic import View
from .models import Notifications
from django.http import HttpRequest , HttpResponse ,request
# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib import pylab
# from pylab import *
# import numpy as np
# import pandas as pd
# import seaborn as sns 
# import io
# from io import *
# import base64
# import PIL, PIL.Image
# from pandas.plotting import bootstrap_plot
# from matplotlib import style
# from pandas.plotting import scatter_matrix
# style.use('ggplot')
# from django.http import FileResponse
# from .utils import render_to_pdf
# from django.template.loader import get_template

# from django.template.response import TemplateResponse
# from django.template.loader import render_to_string
# from django.utils.text import slugify
# Create your views here.


def indexview(request):
    all_notifications_list= Notifications.objects.order_by('created_at')[:10]
    context = {
       'all_notifications_list':all_notifications_list #this is dummy
    }
    return render(request,'FrontendNavigator/index.html',context)



 



