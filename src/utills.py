import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(models, X_train, y_train, X_test, y_test):
    """
    Evaluates the performance of multiple models and returns their accuracy scores and the trained models.

    Parameters:
    models (list): List of tuples where each tuple contains the model name and the model instance.
                   Example: [("Logistic Regression", logistic_model), ("Random Forest", rf_model)]
    X_train (DataFrame): Training features
    y_train (Series): Training labels
    X_test (DataFrame): Test features
    y_test (Series): Test labels

    Returns:
    dict: A dictionary where keys are model names and values are their accuracy scores on the test set.
    dict: A dictionary where keys are model names and values are the trained model instances.
    """
    report = {}
    best_model = None
    best_model_accuracy = 0.0
    logging.info("Evaluating models started")

    for model_name, model_instance in models:
        model = model_instance

        # Train the model
        model.fit(X_train, y_train)
        logging.info(f"Trained model: {model_name}")

        # Predict testing data
        y_test_pred = model.predict(X_test)

        # Get accuracy score for the test data
        test_model_score = accuracy_score(y_test, y_test_pred)
        
        # Store the model name and accuracy score in the report dictionary
        report[model_name] = test_model_score
        logging.info(f"Model: {model_name}, Test Accuracy: {test_model_score}")

         # Check if this model has better accuracy than current best model
        if test_model_score > best_model_accuracy:
            best_model_accuracy = test_model_score
            best_model = model
            logging.info(f"New best model found: {model_name}, Accuracy: {best_model_accuracy}")

    if best_model_accuracy < 0.6:
        raise CustomException("No best model found with accuracy less than 0.6")
    
    return best_model_accuracy,best_model

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

    