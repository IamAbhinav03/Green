from django.urls import path
from .views import login_request, logout_request,  register_request

urlpatterns = [ 
    # path('accounts/signup/', home.SignUpView.as_view(), name='signup'),s
    path('accounts/signup/',  register_request, name='consumer_signup'),
    path('accounts/login', login_request, name='login'),
    path('accounts/logout', logout_request, name='logout'),
    # path('accounts/signup/staff/', staff.TeacherSignUpView.as_view(), name='teacher_signup'),
]