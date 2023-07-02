import keras
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    data = pd.read_csv(r'D:\DataEditTen.csv')
    # Preprocess the data
    data = data.dropna()  # Remove missing values
    data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})  # Convert gender to numerical
    data = pd.get_dummies(data, columns=['family'])  # One-hot encode family variable
    X = data.drop('diabetesOrN', axis=1)  # Extract features
    y = data['diabetesOrN']  # Extract target variable
    scaler = StandardScaler()  # Initialize feature scaler
    X_scaled = scaler.fit_transform(X)  # Scale features
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Build the ANN model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    # Use the model to predict the probability of diabetes for a healthy person
    new_data = pd.DataFrame({'Age': [25], 'Gender': [1], 'Height ': [152], 'Weight': [55],
                             'BMI': [23.8], 'WC': [34], 'Excersice': [1], 'food ': [1],
                             'pressure ': [0], 'family_0': [0], 'family_1': [0], 'family_2': [0], 'smoke': [0]})

    # Reorder the columns of new_data to match the order of X
    new_data = new_data[X.columns]

    new_data_scaled = scaler.transform(new_data)  # Scale the new data using the same scaler as before
    prob = model.predict(new_data_scaled)[0][0]  # Predict the probability of diabetes


    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val9 = float(request.GET['n9'])
    val10 = float(request.GET['n10'])
    val11 = float(request.GET['n11'])
    val12 = float(request.GET['n12'])
    val13 = float(request.GET['n13'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13]])

    result1 = ""

    return render(request, 'predict.html', {"result2":prob})