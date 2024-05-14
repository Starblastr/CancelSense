from django.utils import timezone

from django.shortcuts import render
from django.views.generic import ListView
from .models import Booking
from django.http import HttpResponse
# Create your views here.
from django.contrib.auth.mixins import LoginRequiredMixin
from faker import Faker

class BookingListView(ListView):
    model = Booking
    template_name = 'cancellationPredictor/booking_list.html'
    context_object_name = 'bookings'

    def get_queryset(self):
        today = timezone.localdate()
        today_arrivals = Booking.objects.filter(scheduled_arrival=today)

        fake = Faker()  # Create a Faker generator instance
        for booking in today_arrivals:
            booking.guest_name = fake.name()

        return today_arrivals

def home(request):
    return HttpResponse("Welcome to the Home Page!")