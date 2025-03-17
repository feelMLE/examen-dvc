import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score
import os
import logging

def main():
    #evaluate the trained model and save y_prediction and metriques.

    logger = logging.getLogger(__name__)
    logger.info("Evaluate the train model is running...")
    
    # Load Normalized X_train_scaled and y_train
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    y_test = np.ravel(y_test)

    # Load the Train model
    model = load_model("models/rf_lr_model.pkl")
    
    #Predictions
    y_prediction = model.predict(X_test)

    # Calculer les métriques
    metrics = evaluate(y_test, y_prediction)

    # Sauvegarder les prédictions
    save_predictions(y_prediction, output_filepath="data/y_prediction.csv")

    # Sauvegarder les métriques
    save_metrics(metrics, output_filepath="metrics/scores.json")

    logger.info("Evaluate the train model completed successfully.")

def load_model(filepath):
    #Load model format .pkl.
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


def evaluate(y_test, y_prediction):
    #metriques
    mse = mean_squared_error(y_test, y_prediction)
    rmse= np.sqrt(mse)
    r2 = r2_score(y_test, y_prediction)

    # Retourner les métriques dans un dictionnaire
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "r2_score": r2
    }
    return metrics

def save_predictions(y_prediction, output_filepath):
    #Save prédictions in data CSV format
    predictions_df = pd.DataFrame(y_prediction, columns=["Prediction"])
    predictions_df.to_csv(output_filepath, index=False)

def save_metrics(metrics, output_filepath):
    #Save métriques file JSON
    with open(output_filepath, 'w') as file:
        json.dump(metrics, file, indent=4)

if __name__ == "__main__":
    main()
