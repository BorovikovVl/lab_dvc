#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап 3: Обучение модели"""
import pandas as pd
import numpy as np
import yaml
import joblib
import os
import sys
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

sys.path.append(os.getcwd())
from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def train_model(config):
    logger = get_logger('TRAIN')
    logger.info('Starting model training')
    
    train_df = pd.read_csv(config['data_split']['trainset_path'])
    X_train = train_df.drop(columns=['Price(euro)']).values
    y_train = train_df['Price(euro)'].values
    
    power_trans = PowerTransformer(method='box-cox', standardize=True)
    y_train_scaled = power_trans.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    model = Ridge(alpha=config['train']['alpha'])
    model.fit(X_train, y_train_scaled)
    
    cv_scores = cross_val_score(model, X_train, y_train_scaled, cv=config['train']['cv'])
    logger.info(f'CV Scores: {cv_scores}')
    logger.info(f'Mean CV R2: {cv_scores.mean():.4f}')
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, config['train']['model_path'])
    joblib.dump(power_trans, config['train']['power_path'])
    
    logger.info(f'Model saved to {config["train"]["model_path"]}')
    return model, power_trans

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    train_model(config)
