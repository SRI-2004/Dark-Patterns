from django.urls import path
from .views import main, feedback, get_detection_results

urlpatterns = [
    path('main/', main, name='main'),
    path('feedback/', feedback, name='feedback'),
    path('results/', get_detection_results, name='results')
]
