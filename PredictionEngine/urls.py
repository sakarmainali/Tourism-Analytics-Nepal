from django.urls import path
from django.conf.urls import url , include
from . import views 

urlpatterns = [
    
    path('lists/', views.predict_view ,name='predict_list'),
    path('detail/<int:id>/', views.predict_detail ,name='predict_detail'),
    path('download/pdf/<int:id>/', views.PDFF ,name='ppdf_view'),
    path('lists/pic/<int:id>/', views.listpicview ,name='ppic'),

]