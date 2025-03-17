import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import logging
import os
import numpy as np
def main():
    """Gridsearch to identify best params for RandomForest and save best params."""
    logger = logging.getLogger(__name__)
    logger.info("GridSearch RandomForest running...")
    
    # Load Normalized X_train_scaled and y_train
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_train = np.ravel(y_train)

    # Perform gridsearch
    best_params = perform_grid_search(X_train, y_train)

    # Save Best params
    save_best_params(best_params, output_filepath="models/best_params.pkl")
    logger.info("GridSearch completed successfully.")

def perform_grid_search(X_train, y_train):
    # Model regression RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)

    # Params
    
    param_grid = {
        'n_estimators': [20, 50],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

    # Fit train data
    grid_search.fit(X_train, y_train)

    # Return best params
    return grid_search.best_params_

def save_best_params(best_params, output_filepath):
    # Save file format best_params.pkl
    #os.makedirs(os.path.dirname(output_filepath), exist_ok=True)    
    with open(output_filepath, 'wb') as file:
        pickle.dump(best_params, file)

if __name__ == "__main__":
    main()
