from django.shortcuts import render
from rest_framework import status
#from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Datasets2
from .serializers import DatasetSerializer



class DataJASON(APIView):

    def get(self,request):
        dataset=Datasets2.objects.all()
        serializer=DatasetSerializer(dataset,many=True)
        return Response(serializer.data)


    def post(self):
        pass



