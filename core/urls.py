

from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views


app_name = 'image_classification'

urlpatterns = [
    # two paths: with or without given image
    path('', views.index, name='index'),
    path('getpdf', views.GetPDF, name='getpdf')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)