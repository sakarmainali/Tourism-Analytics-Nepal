#from __future__ import unicode_literals
from django.shortcuts import render , redirect
from django.views.generic import View
from .models import Notifications
from django.http import HttpRequest , HttpResponse ,request


def indexview(request):
    all_notifications_list= Notifications.objects.order_by('created_at')[:10]
    context = {
       'all_notifications_list':all_notifications_list #this is dummy
    }
    return render(request,'FrontendNavigator/index.html',context)



 



