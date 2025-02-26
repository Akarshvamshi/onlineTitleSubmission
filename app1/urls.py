from django.urls import path
from . import views
from .views import output, add_title

urlpatterns = [
    path("", views.homepage, name="homepage"),
    path("output/", output, name="output"),

    path("add_title/", add_title, name="add_title"),
]
