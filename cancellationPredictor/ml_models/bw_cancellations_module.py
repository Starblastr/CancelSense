import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


def loadXlsFiles(folder_paths):
    dataframes = []
    for folder_path in folder_paths:
        files = os.listdir(folder_path)
        excel_files = [f for f in files if f.endswith('.xls')]
        # Loop through the list of Excel files
        for file in excel_files:
            file_path = os.path.join(folder_path, file)  # Full path to the file
            df = pd.read_excel(file_path)  # Read the Excel file into a DataFrame
            dataframes.append(df)  # Append the DataFrame to the list
    return dataframes


class bw_cancellations_model():

    def __init__(self, model_file):
        # read the 'model' and 'scaler' files which were saved
        with open('model', 'rb') as model_file:
            self.reg = pickle.load(model_file)
            self.data = None

    # take a data file (*.csv) and preprocess it in the same way as in the lectures
    def load_and_clean_data(self, data_file=None, folder_paths=None):
        if folder_paths != None:
            try:
                dataframes = loadXlsFiles(folder_paths)
            except Exception as e:
                # Print the actual error message
                print(f"An error occurred: {e}")

        else:
            dataframes = [pd.read_csv(data_file)]



        # Remove operator summary table and its entries that are uninentionally blended with guest data during pd.read_excel
        # As well as observations containing entirely null column values

        for i in range(len(dataframes)):
            # Remove rows where 'Guest Name' column equals 'Operator Summary'
            dataframes[i] = dataframes[i][dataframes[i]['Guest Name'] != 'Operator Summary']

        for dataframe in dataframes:
            dataframe = dataframe.dropna(how='all', inplace=True)

        # For loop to reliably remove 'operator summary' table data from each data frame
        for i in range(len(dataframes)):
            # Find the index of the first occurrence of 'Admin' and 'Totals'
            start_index = dataframes[i][dataframes[i]['Guest Name'] == 'Admin'].index.min()
            end_index = dataframes[i][dataframes[i]['Guest Name'] == 'Totals'].index.min()

            # Drop rows from start_index to end_index, inclusive
            if pd.notna(start_index) and pd.notna(end_index) and start_index <= end_index:
                dataframes[i].drop(dataframes[i].loc[start_index:end_index].index, inplace=True)
        df = pd.concat(dataframes)
        df.reset_index(drop=True, inplace=True)
        import hashlib

        # Function to hash text using SHA-256
        def hash_name(name):
            # Convert the name to a byte string, then hash it
            hash_object = hashlib.sha256(name.encode())
            # Return the hexadecimal representation of the digest
            return hash_object.hexdigest()

        # Hashing names in the DataFrame
        df['Guest Name'] = df['Guest Name'].apply(hash_name)

        # Create a copy of the raw data
        # Create a series countaining non-null percentages to iterate over
        data1 = df.copy()
        non_null_pct = (data1.count() / data1.shape[0] * 100)

        # for loop to drop rows containing non-null values for each column with a non null count percentage of 60% or higher.
        for column in non_null_pct.index:
            if non_null_pct[column] > 50:
                data1.dropna(subset=column, inplace=True)

        high_nullcount_columns = [column for column in non_null_pct.index if non_null_pct[column] < 60]
        high_nullcount_df = data1[high_nullcount_columns]

        # Set null values to 0 for all high null count columns
        clean_data1 = data1
        clean_data1[high_nullcount_columns] = data1[high_nullcount_columns].fillna(0)

        # Now I will drop all the unwanted rows and columns from the dataset

        guest_only_data = data1
        guest_only_data.drop(columns=['PKG', 'Grp ID', 'Unnamed: 14'], inplace=True)

        guest_only_data = guest_only_data.copy()  # This line prevents the 'SettingCopyWithWarning' output

        guest_only_data['Made'] = pd.to_datetime(guest_only_data['Made'], format='%m/%d/%y')
        guest_only_data['Arrive'] = pd.to_datetime(guest_only_data['Arrive'], format='%m/%d/%y')

        guest_only_data = guest_only_data.copy()
        guest_only_data['Nts'] = pd.to_numeric(guest_only_data['Nts'], errors='coerce')

        guest_only_data = guest_only_data.copy()

        guest_only_data['Rate'] = pd.to_numeric(guest_only_data['Rate'], errors='coerce')
        guest_only_data['Override'] = pd.to_numeric(guest_only_data['Override'], errors='coerce')

        guest_only_data.Status = guest_only_data.Status.map({'CXL': 1, 'OUT': 0, 'GTD': 0, 'NS': 0, 'IN': 0, 'HLD': 0})
        guest_only_data.Status = guest_only_data['Status'].astype(bool)
        guest_only_data.Status = guest_only_data.Status.map({True: 1, False: 0})
        guest_only_data.rename(columns={'Status': 'is_canceled', 'Arrive': 'scheduled_arrival', 'Nts': 'num_of_nights',
                                        'Made': 'date_booking_made',
                                        'Disc': 'Discount'}, inplace=True)
        guest_only_data = guest_only_data.copy()

        guest_counts = guest_only_data['Guest Name'].value_counts()

        # to generate the list of names of repeat bookers a list comprehension is the cleanest approach
        # this list comprehension exploits the reverse lookup quality of Series.value_counts() objects

        duplicate_guest_names = [name for name in guest_only_data['Guest Name'] if guest_counts[name] > 1]

        repeat_guest_df = guest_only_data[guest_only_data['Guest Name'].isin(duplicate_guest_names)]

        repeat_guest = repeat_guest_df['Guest Name'].unique()

        guest_only_data['is_repeated_guest'] = guest_only_data['Guest Name'].isin(duplicate_guest_names)

        guest_only_data['is_repeated_guest'] = guest_only_data['is_repeated_guest'].map({True: 1, False: 0})

        repeat_guest_df = repeat_guest_df.sort_values(by='date_booking_made')
        repeat_guest_df = repeat_guest_df.sort_values(by=['Guest Name', 'date_booking_made'])

        # Get the number of all arrival dates
        num_arrival_dates = guest_only_data.scheduled_arrival.shape[0]

        # Calculate the time difference and store it in a new column called 'booking_to_arrival_duration'
        guest_only_data['booking_to_arrival_duration'] = guest_only_data['scheduled_arrival'] - guest_only_data[
            'date_booking_made']

        # Now guest_only_data contains a new column 'booking_to_arrival_duration' with the time difference
        # Convert timedelta values to integers representing the number of days
        guest_only_data['booking_to_arrival_duration'] = guest_only_data['booking_to_arrival_duration'].dt.days

        # Create month_of_arrival column
        arrival_months = []
        for i in range(num_arrival_dates):
            # Use iloc to access the scheduled_arrival by position
            arrival_months.append(guest_only_data['scheduled_arrival'].iloc[i].month)
        guest_only_data['month_of_arrival'] = pd.Series(arrival_months, index=guest_only_data.index)

        # Create day_of_arrival column

        arrival_days = []
        for i in range(num_arrival_dates):
            # Use iloc to access the scheduled_arrival by position
            arrival_days.append(guest_only_data['scheduled_arrival'].iloc[i].day)

        # Creating the day_of_arrival Series directly with the DataFrame's index
        guest_only_data['day_of_arrival'] = pd.Series(arrival_days, index=guest_only_data.index)

        # Create day_of_booking column
        booking_days = []
        for i in range(num_arrival_dates):
            # Use iloc to access date_booking_made by position
            booking_days.append(guest_only_data['date_booking_made'].iloc[i].day)

        # Creating the day_of_booking Series directly with the DataFrame's index
        guest_only_data['day_of_booking'] = pd.Series(booking_days, index=guest_only_data.index)

        # Create weekday_of_arrival column
        arrival_weekdays = []
        for i in range(num_arrival_dates):
            # Use iloc to access scheduled_arrival by position
            arrival_weekdays.append(guest_only_data['scheduled_arrival'].iloc[i].weekday())

        # Creating the weekday_of_arrival Series directly with the DataFrame's index
        guest_only_data['weekday_of_arrival'] = pd.Series(arrival_weekdays, index=guest_only_data.index)

        # Create weekday_of_booking column
        booking_weekdays = []
        for i in range(num_arrival_dates):
            # Use iloc to access date_booking_made by position
            booking_weekdays.append(guest_only_data['date_booking_made'].iloc[i].weekday())

        # Creating the weekday_of_booking Series directly with the DataFrame's index
        guest_only_data['weekday_of_booking'] = pd.Series(booking_weekdays, index=guest_only_data.index)

        # Create month_of_booking column
        booking_months = []
        for i in range(num_arrival_dates):
            # Use iloc to access date_booking_made by position
            booking_months.append(guest_only_data['date_booking_made'].iloc[i].weekday())

        # Creating the weekday_of_booking Series directly with the DataFrame's index
        guest_only_data['month_of_booking'] = pd.Series(booking_months, index=guest_only_data.index)

        df_list = []
        for guest_name in repeat_guest:
            filtered_df = repeat_guest_df[
                (repeat_guest_df['Guest Name'] == guest_name) & (repeat_guest_df['is_canceled'] == 0)]

            # Reset the counter and list for each guest
            previous_stays_list = [0] * len(filtered_df)
            num_of_previous_stays = 0

            # Iterate over the DataFrame rows
            for idx in range(1, len(filtered_df)):
                num_of_previous_stays += 1  # Increment counter for previous cancellations
                previous_stays_list[idx] = num_of_previous_stays

            # Assign the series after the loop
            filtered_df['previous_stays'] = pd.Series(previous_stays_list, dtype=int, index=filtered_df.index)
            df_list.append(filtered_df)

        updated_df = pd.concat(df_list)
        guest_only_data = pd.concat([guest_only_data, updated_df['previous_stays']], axis=1)

        guest_only_data.fillna(0, inplace=True)
        guest_only_data['previous_stays'] = guest_only_data['previous_stays'].astype(int)
        guest_only_data = guest_only_data.sort_values(by=['Guest Name', 'date_booking_made'])

        df_list = []
        for guest_name in repeat_guest:
            filtered_df = repeat_guest_df[
                (repeat_guest_df['Guest Name'] == guest_name) & (repeat_guest_df['is_canceled'] == 1)]

            # Reset the counter and list for each guest
            previous_cancels_list = [0] * len(filtered_df)
            num_of_previous_cancels = 0

            # Iterate over the DataFrame rows
            for idx in range(1, len(filtered_df)):
                num_of_previous_cancels += 1  # Increment counter for previous cancellations
                previous_cancels_list[idx] = num_of_previous_cancels

            # Assign the series after the loop with specified dtype
            filtered_df['previous_cancellations'] = pd.Series(previous_cancels_list, dtype=int, index=filtered_df.index)
            df_list.append(filtered_df)

        # Concatenate all DataFrames collected in the list
        updated_df = pd.concat(df_list)

        # Assuming 'guest_only_data' is a DataFrame defined elsewhere and ready to be used here
        guest_only_data = pd.concat([guest_only_data, updated_df[['previous_cancellations']]], axis=1)

        # Fill NA values with zero and ensure integer type for the 'previous_cancellations' column
        guest_only_data.fillna(0, inplace=True)
        guest_only_data['previous_cancellations'] = guest_only_data['previous_cancellations'].astype(int)

        # Sorting by 'Guest Name' and 'date_booking_made' assuming these columns exist
        guest_only_data = guest_only_data.sort_values(by=['Guest Name', 'date_booking_made'])

        columns_reordered = ['Guest Name', 'date_booking_made', 'month_of_booking', 'day_of_booking',
                             'weekday_of_booking',
                             'scheduled_arrival', 'month_of_arrival', 'day_of_arrival', 'weekday_of_arrival',
                             'num_of_nights', 'is_repeated_guest', 'previous_stays', 'previous_cancellations',
                             'booking_to_arrival_duration', 'Rate', 'Discount', 'Override', 'Company', 'Type', 'Clerk',
                             'Conf #',
                             'is_canceled']

        final_df = guest_only_data[columns_reordered]

        from datetime import datetime
        # Get today's date
        today_date = datetime.today()

        # Format today's date as "YYYY-MM-DD"
        formatted_date = today_date.strftime("%Y-%m-%d")

        todays_arrivals = final_df[(final_df['scheduled_arrival'] == formatted_date) & (final_df['is_canceled'] == 0)]

        self.todays_arrivals = todays_arrivals.copy()

        final_df = final_df.drop(todays_arrivals.index)

        data_with_all_features = final_df.copy()

        todays_arrivals.drop(columns='Guest Name', inplace=True)
        todays_arrivals.drop(columns='Conf #', inplace=True)
        todays_arrivals.drop(columns='date_booking_made', inplace=True)
        todays_arrivals.drop(columns='scheduled_arrival', inplace=True)
        todays_arrivals.drop(columns='Discount', inplace=True)
        todays_arrivals.drop(columns='Company', inplace=True)
        todays_arrivals.drop(columns='Clerk', inplace=True)
        todays_arrivals.drop(columns='Type', inplace=True)
        todays_arrivals.drop(columns='is_canceled', inplace=True)

        self.data = todays_arrivals.copy()
        self.preprocessed_data = todays_arrivals.copy()

    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1].astype(float).round(4)

            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            self.preprocessed_data['Guest Name'] = self.todays_arrivals['Guest Name']
            self.preprocessed_data['scheduled_arrival'] = self.todays_arrivals['scheduled_arrival']
            self.preprocessed_data['date_booking_made'] = self.todays_arrivals['date_booking_made']

            return self.preprocessed_data

