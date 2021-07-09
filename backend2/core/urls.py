from django.urls import path
from . import views

urlpatterns = [
	path('', views.reuse, name='reuse'),
]