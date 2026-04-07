#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап 4: Оценка качества модели"""
import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import yaml

sys.path.append(os.getcwd())
from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def evaluate_model(config):
    logger = get_logger('EVALUATE')
    logger.info('Starting model evaluation')
    
    test_df = pd.read_csv(config['test']['testset_path'])
    X_test = test_df.drop(columns=['Price(euro)']).values
    y_test = test_df['Price(euro)'].values
    
    model = joblib.load(config['test']['model_path'])
    power_trans = joblib.load(config['test']['power_path'])
    
    y_pred_scaled = model.predict(X_test)
    y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'n_test_samples': int(len(y_test))
    }
    
    logger.info(f'MAE: {mae:.2f}')
    logger.info(f'RMSE: {rmse:.2f}')
    logger.info(f'R2: {r2:.4f}')
    logger.info(f'MAPE: {mape:.4f}')
    
    os.makedirs('dvclive', exist_ok=True)
    with open('dvclive/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': np.abs(y_test - y_pred)
    })
    results_df.to_csv('dvclive/predictions.csv', index=False)
    
    logger.info('Evaluation completed successfully')
    return metrics

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    evaluate_model(config)
