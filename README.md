# Real-Estate-Price-Prediction

The Real Estate House Prediction Project involves the development of a machine learning model to predict the prices of residential homes based on various features.

Project Description:

1.Developed a machine learning model for predicting real estate housing prices based on various features.  
2.Performed data cleaning, outlier detection and removal, feature engineering, and dimensionality reduction on the Bangalore home prices dataset from Kaggle.  
3.Tuned the model's hyperparameters using GridSearchCV and K-fold cross-validation techniques to improve its accuracy.  
4.Integrated the trained model with a Python Flask server and served HTTP requests from a website.  
5.Created a user-friendly website in HTML, CSS, and JavaScript that allowed users to input property details and obtain accurate price predictions.  
  
First, Data Preprocessing- data loading and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, and hyperparameter tuning using GridSearchCV and K-fold cross-validation is performed on the Bangalore home prices dataset.  
Next, a machine learning Linear Regression model using Scikit-Learn is trained and created using the processed dataset.  
The next step involves developing a Python Flask server that utilizes the saved model to serve HTTP requests, followed by the development of a user-friendly website in HTML, CSS, and JavaScript.  
The website allows users to input information such as house area, number of bedrooms, number of bathrooms and location which calls the Flask server to retrieve the predicted price and display on the website.  
