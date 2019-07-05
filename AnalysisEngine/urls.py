from django.urls import path
from django.conf.urls import url , include
from . import views 


urlpatterns = [
    path('lists/', views.analysislist ,name='topic_list'),
    path('download/pdf/<int:id>/', views.PDFF ,name='pdf_view'),
    path('lists/details/<int:id>/', views.detailview ,name='detail'),
    path('lists/pic/<int:id>/', views.listpicview ,name='pic'),

]