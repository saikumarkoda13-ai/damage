from django.urls import path
from . import views

urlpatterns = [
    path('register/',      views.api_register,      name='api_register'),
    path('login/',         views.api_login,          name='api_login'),
    path('predict/',       views.api_predict,        name='api_predict'),
    path('admin-login/',   views.api_admin_login,    name='api_admin_login'),
    path('users/',         views.api_users,          name='api_users'),
    path('activate-user/', views.api_activate_user,  name='api_activate_user'),
]
