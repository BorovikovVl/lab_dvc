#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап 1: Очистка табличного набора данных"""
import pandas as pd
import yaml
import sys
import os

sys.path.append(os.getcwd())
from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clean_data(path2data, output_path):
    logger = get_logger('DATA_CLEANING')
    logger.info('Starting data cleaning')
    
    df = pd.read_csv(path2data)
    logger.info(f'Original dataset shape: {df.shape}')
    
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    initial_count = len(df)
    
    # Удаление аномалий
    df = df[df.Distance <= 1e6]
    df = df[df['Engine_capacity(cm3)'] >= 200]
    df = df[df['Engine_capacity(cm3)'] <= 5000]
    df = df[df['Price(euro)'] >= 101]
    df = df[df['Price(euro)'] <= 1e5]
    df = df[df.Year >= 1971]
    
    df = df.reset_index(drop=True)
    
    # Ordinal Encoding категориальных признаков
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder()
    df[cat_columns] = encoder.fit_transform(df[cat_columns])
    
    logger.info(f'Cleaned dataset shape: {df.shape}')
    logger.info(f'Removed {initial_count - len(df)} rows')
    
    df.to_csv(output_path, index=False)
    logger.info(f'Saved cleaned data to {output_path}')
    return df

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    clean_data(
        config['data_load']['dataset_csv'],
        config['data_cleaning']['cleaned_data_path']
    )
