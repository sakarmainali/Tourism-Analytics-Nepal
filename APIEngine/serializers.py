from rest_framework import serializers
from .models import Datasets2



class DatasetSerializer(serializers.ModelSerializer):

    class Meta:
        model=Datasets2
        fields='__all__'