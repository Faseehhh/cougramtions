from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("Recommend", views.Recommend, name="recommend"),
    path("pdf/id=<int:id>/", views.pdf, name="pdf"),
    path("result", views.pdf, name="result"),
    
   
]