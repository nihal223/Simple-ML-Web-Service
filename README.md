# Simple-ML-Web-Service
:pushpin: A simple REST API for a Machine Learning Model in Flask 

### Overview:
It is a simple project to get an understanding of turning your machine learning models into a REST API

### Project Layout
* Data - Contains the Training and Test Data
* titanic_ml.py - Code to build the model
* MLWebAPI.py - Code to build the API for the model built in the above code

### Description:
The Model is constructed using data from the titanic dataset which basically has data about if a person survived or
not based on some features. titanic.py preprocesses the data and uses Logistic regression to build the model and pickle it.
MLWebAPI.py preprocesses the POST request data and makes a prediction using the built model.

### How to run
```
python3 titanic_ml.py
```
```
python3 MLWebAPI.py
```
Then use a tool like PostMan to test the API. 
