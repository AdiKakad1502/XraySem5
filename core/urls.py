
from django.contrib.auth import views as auth_views
from django.urls import path, re_path
from django.conf import settings
from django.conf.urls.static import static
from .forms import ImageUploadForm
from . import views

app_name = 'image_classification'

urlpatterns = [
    # two paths: with or without given image
    path('', views.MainPage.as_view(), name='index'),
    path('getpdf', views.MakePDF.as_view(), name='getpdf'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)