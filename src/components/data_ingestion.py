import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from pathlib import Path
from dataclasses import dataclass
from src.utils.utils import split_data
import joblib


@dataclass
class DataIngestionConfig:
    raw_image_path:str = os.path.join("artifact", "raw_img_path.csv")
    raw_labels: str = os.path.join("artifact", "raw_labels.csv")
    train_data_path_x:str = os.path.join("artifact", "train_x.csv")
    train_data_path_y:str = os.path.join("artifact", "train_y.csv")
    test_data_path_x:str = os.path.join("artifact", "test_x.csv")
    test_data_path_y:str = os.path.join("artifact", "test_y.csv")
    unique_charachters:str = os.path.join("artifact", "unique_char.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion (self):
        logging.info("Starting data ingestion")
        try:
            logging.info("Reading The data from the folders...")
            data_dir = Path("E:/ML/ocr_captcha/data/train_images_mlware/train_images")
            images = sorted(list(map(str, list(data_dir.glob("*.png")))))
            ids = [img.split(os.path.sep)[-1] for img in images]
            data_sheet = pd.read_csv("E:/ML/ocr_captcha/data/train-labels_mlware.csv")
            data_dict = {data_sheet['image'][i]: data_sheet['text'][i] for i in range(25000)}
            labels = [data_dict[item] for item in ids]
            characters = set(char for label in labels for char in label)
            characters = sorted(list(characters))
            logging.info(f"Number of images found: {len(images)}")
            logging.info(f"Number of labels found: {len(labels)}")
            logging.info(f"Number of unique characters: {len(characters)}")
            logging.info(f"Characters present: {characters}")
            logging.info("Saving the unique Charachters in the artifacts...")
            df = pd.DataFrame(characters)
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.unique_charachters)), exist_ok=True)
            df.to_csv(self.ingestion_config.unique_charachters, header=False, index=False)
            logging.info("unique Charachters saved")
            max_length = max([len(label) for label in labels])
            logging.info(f"max_length -> {max_length}")
            min_length = min([len(label) for label in labels])
            logging.info(f"min_length -> {min_length}")
            logging.info("Saving the raw_data in artifacts...")
            # image paths
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_image_path)), exist_ok=True)
            df = pd.DataFrame(images, columns=['Image_Path'])
            df.to_csv(self.ingestion_config.raw_image_path, index=False)
            # labels
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_labels)), exist_ok=True)
            data_sheet.to_csv(self.ingestion_config.raw_labels, index=False)
            logging.info("Saved the raw data paths in the artifacts")
            logging.info("Performing the Train-Test Split")
            x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
            logging.info("Split Done")
            logging.info("Saving the train data...")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.train_data_path_x)), exist_ok=True)
            np.savetxt(self.ingestion_config.train_data_path_x, x_train, delimiter=',', fmt='%s')
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.train_data_path_y)), exist_ok=True)
            np.savetxt(self.ingestion_config.train_data_path_y, y_train, delimiter=',', fmt='%s')
            logging.info("Saved the train data in the artifact")
            logging.info("Saving the testing data...")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.test_data_path_x)), exist_ok=True)
            np.savetxt(self.ingestion_config.test_data_path_x, x_valid, delimiter=',', fmt='%s')
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.test_data_path_y)), exist_ok=True)
            np.savetxt(self.ingestion_config.test_data_path_y, y_valid, delimiter=',', fmt='%s')
            logging.info("Saved the testing data")
            logging.info("Data Ingestion Completed!")
            
            return (
                self.ingestion_config.train_data_path_x,
                self.ingestion_config.train_data_path_y,
                self.ingestion_config.test_data_path_x,
                self.ingestion_config.test_data_path_y,
                self.ingestion_config.unique_charachters
            )
        except Exception as e:
            logging.info()
            raise customexception(e, sys)

if __name__ == '__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()