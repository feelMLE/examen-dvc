
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import logging
import os

def main():
    #Fit model RandomForestRegressor with best params from GridSearch and save.
    logger = logging.getLogger(__name__)
    logger.info("Fit the modèle RandomForest is running...")
    
    # Load Normalized X_train_scaled and y_train
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    #Load best params from models
    best_params = load_best_params("models/best_params.pkl")

    # Fit the model RandomForestRegressor
    model = train_random_forest(X_train, y_train, best_params)

    # Save the trained models in models
    save_model(model, output_filepath="models/rf_lr_model.pkl")
    logger.info("Model RandomForestRegressor fit and saved completed.")

def load_best_params(filepath):
    #lod best_params.pkl from models
    with open(filepath, 'rb') as file:
        best_params = pickle.load(file)
    return best_params

def train_random_forest(X_train, y_train, best_params):
    #Train RandomForestRegressor with best params
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train.values.ravel())  # Utiliser .values.ravel() si y_train est un DataFrame.
    return model

def save_model(model, output_filepath):
    #Save the the train_model fichier .pkl
    #os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    main()
