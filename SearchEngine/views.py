from django.shortcuts import render
from django.shortcuts import redirect
from .models import Search
from AnalysisEngine.models import Analysis
from PredictionEngine.models import Predictions

def searchView(request):
    query=request.GET.get('query')

    # w = (Q(summary__icontains=query)|Q(body__icontains=query))
    w=Analysis.objects.filter(body__icontains=query)
    q=Predictions.objects.filter(body__icontains=query)
    print(w)
    context={
        'y':w,
        'z':q,
    }
    return render(request,'searchtest.html',context)

