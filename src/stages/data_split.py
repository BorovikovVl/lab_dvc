#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Разделение данных на train/test"""
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.getcwd())
from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def split_data(config):
    logger = get_logger('DATA_SPLIT')
    logger.info('Starting data split')
    
    df = pd.read_csv(config['featurize']['features_path'])
    
    train_df, test_df = train_test_split(
        df, 
        test_size=config['data_split']['test_size'],
        random_state=42
    )
    
    train_df.to_csv(config['data_split']['trainset_path'], index=False)
    test_df.to_csv(config['data_split']['testset_path'], index=False)
    
    logger.info(f'Train set: {len(train_df)} samples')
    logger.info(f'Test set: {len(test_df)} samples')
    return train_df, test_df

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    split_data(config)
