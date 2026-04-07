#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап 2: Генерация новых признаков"""
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

def generate_features(input_path, output_path):
    logger = get_logger('FEATURE_ENGINEERING')
    logger.info('Starting feature generation')
    
    df = pd.read_csv(input_path)
    logger.info(f'Input dataset shape: {df.shape}')
    
    # Генерация новых признаков
    df['Distance_by_year'] = df['Distance'] / (2024 - df['Year'] + 1)
    df['age'] = 2024 - df['Year']
    df['engine_per_age'] = df['Engine_capacity(cm3)'] / (df['age'] + 1)
    
    mean_engine_cap = df.groupby('Style')['Engine_capacity(cm3)'].mean()
    df['eng_cap_diff'] = df.apply(
        lambda row: abs(row['Engine_capacity(cm3)'] - mean_engine_cap[row['Style']]), 
        axis=1
    )
    
    max_engine_cap = df.groupby('Style')['Engine_capacity(cm3)'].max()
    df['eng_cap_diff_max'] = df.apply(
        lambda row: abs(row['Engine_capacity(cm3)'] - max_engine_cap[row['Style']]), 
        axis=1
    )
    
    logger.info(f'Output dataset shape: {df.shape}')
    df.to_csv(output_path, index=False)
    logger.info(f'Saved features to {output_path}')
    return df

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    generate_features(
        config['data_cleaning']['cleaned_data_path'],
        config['featurize']['features_path']
    )
