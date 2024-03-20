import os
import sys
import mlflow
import numpy as np
import mlflow.keras
from src.ocr_captcha.logger.logging import logging
from src.ocr_captcha.exception.exception import customexception
from pathlib import Path

import pandas as pd

from src.ocr_captcha.components.data_ingestion import DataIngestion
from src.ocr_captcha.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self, pretrained:bool, model_path: str):
        self.pre_trained = pretrained
        self.model_path = os.path.join("artifact", model_path)
    
    
    def start_data_ingestion(self):
        try:
            obj = DataIngestion()
            train_data_path_x, train_data_path_y, test_data_path_x, test_data_path_y, unique_charachters = obj.initiate_data_ingestion()
            mlflow.log_artifact(train_data_path_x)
            mlflow.log_artifact(train_data_path_y)
            mlflow.log_artifact(test_data_path_x)
            mlflow.log_artifact(test_data_path_y)
            mlflow.log_artifact(unique_charachters)
            return train_data_path_x, train_data_path_y, test_data_path_x, test_data_path_y, unique_charachters
        except Exception as e:
            raise customexception(e, sys)

        
    def initiate_training(self):
        try:
            with mlflow.start_run() as run:
                self.img_height = 100
                self.img_width = 200
                self.batch_size = 32
                mlflow.log_params({
                    "batch_size": self.batch_size,
                    "img_height": self.img_height,
                    "img_width": self.img_width,
                    "model_path": self.model_path,
                    "pre_trained": self.pre_trained
                })
        except Exception as e:
            raise customexception(e, sys)

        
    def train(self, train_data_path_x, train_data_path_y, test_data_path_x, test_data_path_y, unique_charachters):
        try:  
                trainer = ModelTrainer(self.img_height, self.img_width, self.batch_size)
                model_path, history = trainer.initate_model_training(train_data_path_x, train_data_path_y, test_data_path_x, test_data_path_y, unique_charachters, self.pre_trained, self.model_path)
                mlflow.log_artifact(model_path)
                for epoch, val_loss_value in enumerate(history.history["val_loss"]):
                    mlflow.log_metric("val_loss_epoch_" + str(epoch), val_loss_value)
                for epoch, loss_value in enumerate(history.history["loss"]):
                    mlflow.log_metric("loss_epoch_" + str(epoch), loss_value)
        except Exception as e:
            logging.info(e)
            raise customexception(e, sys)


if __name__ == "__main__":
    t_pipe = TrainingPipeline(False, "model.weights.h5")
    t_pipe.train()





 
