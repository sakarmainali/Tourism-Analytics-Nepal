from django.db import models
from datetime import datetime
# Create your models here.

class Notifications(models.Model):
    summary=models.TextField(max_length=800)
    title=models.CharField(max_length = 100)
    body=models.TextField(max_length= 90000)
    created_at=models.DateTimeField(default=datetime.now,blank=True)
    def __str__(self):
        return self.title[:50]


