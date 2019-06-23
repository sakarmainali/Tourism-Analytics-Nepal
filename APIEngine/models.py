from django.db import models
from datetime import datetime
# Create your models here.

class Datasets(models.Model):
    summary=models.TextField(max_length=800)
    title=models.CharField(max_length = 200)
    body=models.FileField(upload_to='assets/uploads/')
    created_at=models.DateTimeField(default=datetime.now,blank=True)
    def __str__(self):
        return self.title[:100]

class Datasets2(models.Model):
    summary=models.TextField(max_length=800)
    title=models.CharField(max_length = 200)
    body=models.TextField(max_length=99000)
    created_at=models.DateTimeField(default=datetime.now,blank=True)
    def __str__(self):
        return self.title[:100]