from django.urls import path
from .views import BookingListView

urlpatterns = [
    path('', BookingListView.as_view(), name='booking-list'),
]
