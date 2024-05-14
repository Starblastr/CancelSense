from django.core.management.base import BaseCommand, CommandError
import pickle
import pandas as pd
from cancellationPredictor.ml_models.bw_cancellations_module import *
from cancellationPredictor.models import Booking
from django.conf import settings
import os

model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'model')



print("Current working directory:", os.getcwd())
folder_paths = [os.getcwd() + '/2021-2022',
                os.getcwd() + '/2022-2023',
                os.getcwd() + '/2023-2024',
                os.getcwd() + '/2024-2025', ]
print(folder_paths)


class Command(BaseCommand):
    help = 'Imports data from a DataFrame into the Booking model'

    def handle(self, *args, **options):
        # Assuming model.predicted_outputs() is available and returns a DataFrame
        # Adjust import as necessary
        # Instantiate your model and get the DataFrame


        model = bw_cancellations_model('model')
        model.load_and_clean_data(folder_paths=folder_paths)

        df = model.predicted_outputs()

        # Iterate over the DataFrame's rows
        for index, row in df.iterrows():
            Booking.objects.update_or_create(
                guest_name=row['Guest Name'],
                defaults={
                    'month_of_booking': row['month_of_booking'],
                    'day_of_booking': row['day_of_booking'],
                    'weekday_of_booking': row['weekday_of_booking'],
                    'month_of_arrival': row['month_of_arrival'],
                    'day_of_arrival': row['day_of_arrival'],
                    'weekday_of_arrival': row['weekday_of_arrival'],
                    'num_of_nights': row['num_of_nights'],
                    'is_repeated_guest': row['is_repeated_guest'],
                    'previous_stays': row['previous_stays'],
                    'previous_cancellations': row['previous_cancellations'],
                    'booking_to_arrival_duration': row['booking_to_arrival_duration'],
                    'rate': row['Rate'],
                    'override': row['Override'],
                    'probability': row['Probability'],
                    'prediction': row['Prediction'],
                    'date_booking_made': pd.to_datetime(row['date_booking_made']).date(),
                    'scheduled_arrival': pd.to_datetime(row['scheduled_arrival']).date(),
                }
            )

        self.stdout.write(self.style.SUCCESS('Successfully imported data into the Booking model'))