

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from check_structure import check_existing_file, check_existing_folder
import logging
from pathlib import Path
import os

def main():
    """ Runs data processing scripts to Normalize data from  data/processed 
         to saved X_train_scaled and X_test_scaled in data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Normalize final data set from data/processed')
    process_data()

def process_data():
    #DVC not authorize same input and ouput directory:scaled 
    output_filepath = "data/processed"
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    #Normalize
    X_train_scaled, X_test_scaled = normalize(X_train, X_test)

    # Convert numpy arrays to pandas DataFrames and keep the columnn naming
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled_df, X_test_scaled_df, output_filepath)

def normalize(X_train, X_test):
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    #if check_existing_folder(output_folderpath):
    #   os.makedirs(output_folderpath)
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()