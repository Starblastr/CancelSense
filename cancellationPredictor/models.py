from django.db import models

# Create your models here.
from django.db import models

class Booking(models.Model):
    month_of_booking = models.IntegerField()
    day_of_booking = models.IntegerField()
    weekday_of_booking = models.IntegerField()
    month_of_arrival = models.IntegerField()
    day_of_arrival = models.IntegerField()
    weekday_of_arrival = models.IntegerField()
    num_of_nights = models.IntegerField()
    is_repeated_guest = models.BooleanField()
    previous_stays = models.IntegerField()
    previous_cancellations = models.IntegerField()
    booking_to_arrival_duration = models.IntegerField()
    rate = models.DecimalField(max_digits=10, decimal_places=2)  # Assuming 'Rate' needs to handle decimals
    override = models.DecimalField(max_digits=10, decimal_places=2)  # Assuming 'Override' is a monetary value
    probability = models.DecimalField(max_digits=10, decimal_places=6)  # Storing probabilities with precision
    prediction = models.BooleanField()  # Assuming prediction is a binary outcome (True/False)
    guest_name = models.TextField()  # Adjust max_length as needed
    date_booking_made = models.DateField()
    scheduled_arrival = models.DateField()

    def __str__(self):
        return self.guest_name
