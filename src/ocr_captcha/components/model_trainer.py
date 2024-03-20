import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from src.utils.utils import encode_single_sample_training
import tensorflow as tf
from functools import partial
from keras import layers
import pickle
from src.utils.utils import build_model
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
import json
from src.utils.utils import save_object





@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifact','model.weights.h5')
    char_to_num:str = os.path.join("artifact", "char_to_num.json")
    num_to_char:str = os.path.join("artifact", "num_to_char.json")
class ModelTrainer:
    def __init__(self, img_height, img_width, batch_size):
        self.model_trainer_config = ModelTrainerConfig()
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
    def initate_model_training(self, train_path_x, train_path_y, test_path_x, test_path_y, unique_chars, pre_trained: bool = False, model_path:str = None):
        try:
            logging.info("Getting the train-test | feature-labels and unique charachters from the artifacts...")
            train_x = pd.read_csv(train_path_x, header=None)[0]
            train_y = pd.read_csv(train_path_y, header=None)[0]
            test_x = pd.read_csv(test_path_x, header=None)[0]
            test_y = pd.read_csv(test_path_y, header=None)[0]
            unique_chars = pd.read_csv(unique_chars, header=None)[0]
            logging.info("Got all the data")
            logging.info("Creating the mappings...")
            char_to_num = layers.StringLookup(vocabulary=list(unique_chars), mask_token=None)
            num_to_char = layers.StringLookup(
                vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
            )
            logging.info("Mappings created")
            logging.info("storing the mapping...")
            
            
            
            
            
            
            os.makedirs(os.path.dirname(os.path.join(self.model_trainer_config.char_to_num)), exist_ok=True)
            saved_data = {'config': char_to_num.get_config(), 'weights': char_to_num.get_weights()}
            with open(self.model_trainer_config.char_to_num, "w") as json_file:
                json.dump(saved_data, json_file)
            
            os.makedirs(os.path.dirname(os.path.join(self.model_trainer_config.num_to_char)), exist_ok=True)
            saved_data = {'config': num_to_char.get_config(), 'weights': num_to_char.get_weights()}
            with open(self.model_trainer_config.num_to_char, "w") as json_file:
                json.dump(saved_data, json_file)

            
            
            
            
            
            
            
            logging.info("saved the mappings")
            partial_encode_single_sample_training = partial(encode_single_sample_training, img_height=self.img_height, img_width=self.img_width, char_to_num=char_to_num)
            logging.info("Creating the training Dataset")
            train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
            train_dataset = (
                train_dataset.map(partial_encode_single_sample_training, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(self.batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            )
            logging.info("Creating the validation Dataset")
            validation_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
            validation_dataset = (
                validation_dataset.map(partial_encode_single_sample_training, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(self.batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            )
            logging.info("Building The Model")        
            if len(tf.config.experimental.list_physical_devices('GPU')) != 0:
                logging.info("GPU found, building model  over GPU")
                with tf.device('/device:GPU:0'):
                    model = build_model(self.img_width, self.img_height, char_to_num)
            else:
                logging.info("GPU not found, proceding with CPU...")
                model = build_model(self.img_width, self.img_height, char_to_num) 
            if pre_trained==True:
                model.load_weights(model_path)
                logging.info("Starting with pre trained model")
            else:
                logging.info("Created a model from scratch")
            logging.info("Starting the Training ...")
            epochs = 2
            early_stopping_patience = 1
            # Add early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
            )
            # Create a ModelCheckpoint callback
            checkpoint_path = "checkpoints_80percent/model_checkpoint.weights.h5"
            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,  # Save only the model's weights
                monitor='val_loss',  # Monitor validation loss
                save_best_only=True,  # Save only the best model (based on validation loss)
                verbose=1
            )
            # Train the model
            history = model.fit(
                train_dataset.take(1),
                validation_data=validation_dataset.take(1),
                epochs=epochs,
                callbacks=[early_stopping, checkpoint_callback],
            )
            logging.info("Training Finished")
            logging.info("Saving the model in artifacts ...")
            os.makedirs(os.path.dirname(os.path.join(self.model_trainer_config.trained_model_file_path)), exist_ok=True)
            model.save_weights(self.model_trainer_config.trained_model_file_path)
            logging.info("Model Saved.")
            return self.model_trainer_config.trained_model_file_path, history
        except Exception as e:
            logging.info(e)
            raise customexception(e, sys)


if __name__ == '__main__':
    trainer = ModelTrainer(100, 200, 32)
    train_path_x = Path("E:\\ML\\ocr_captcha\\artifact\\train_x.csv")
    train_path_y= Path("E:\\ML\\ocr_captcha\\artifact\\train_y.csv")
    test_path_x = Path("E:\\ML\\ocr_captcha\\artifact\\test_x.csv")
    test_path_y = Path("E:\\ML\\ocr_captcha\\artifact\\test_y.csv")
    unique_chars = Path("E:\\ML\\ocr_captcha\\artifact\\unique_char.csv")
    model_path = trainer.initate_model_training(train_path_x, train_path_y, test_path_x, test_path_y, unique_chars)
            