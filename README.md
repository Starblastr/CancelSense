Description
This repository contains a demo version of a web application developed to forecast, mitigate, and prevent profit loss resulting from unforeseen cancellations. The application utilizes machine learning techniques, specifically a bespoke logistic regression model developed with scikit-learn, to predict booking cancellations for a specific hotel.

Problem Statement
The application aims to address the challenge of profit loss caused by cancellations in the hospitality industry. By accurately predicting cancellations, the hotel can take proactive measures to mitigate the impact, such as optimizing staffing levels and offering incentives to prevent cancellations.

Features
Machine Learning Model: Incorporates a logistic regression model to predict booking cancellations.
Employee Authentication: Requires employee authentication to access the application.
Data Anonymization: Names in the database are hashed using SHA256 to maintain guest anonymity.
Data Update: The database is updated daily via a batch file and Task Scheduler to ensure the most current bookings and probabilities of cancellation are reflected.
Technologies Used
Django: Web framework for developing the application.
scikit-learn: Machine learning library used to build the logistic regression model.
Python Hashlib: Utilized to hash names in the database for anonymity.
Faker Library: Used to transform hashed names back into human-readable names for demo purposes.

Explore the dashboard to view booking predictions and other relevant information.
Disclaimer
This is a demo version of the application and does not contain actual guest data. All names and information displayed are generated for demonstration purposes only.

Future Improvements
Incorporate real-time data updates for more accurate predictions.
Expand the model to predict cancellations for different types of establishments.
Implement more advanced machine learning techniques for improved accuracy.
Contributing
We welcome contributions to enhance the functionality and features of the application. Please submit a pull request with your proposed changes.

License
