from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [ 
	path('', views.home, name='home'),
	# path('reuse/', views.reuse, name='reuse'),
	path('empty/', views.empty, name='empty'),
	path('profile/', views.profile, name='profile'),
	path('test/', views.test, name='test'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)