import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import json
from src.logger.logging import logging
from src.exception.exception import customexception
from keras.models import load_model
import keras
from keras.src.layers.preprocessing.string_lookup import StringLookup
import pandas as pd
from src.utils.utils import build_model, generate_image_id, encode_single_sample_testing, decode_batch_predictions
import numpy as np


class PredictionPipline:
    def __init__(self, model_path):
        self.model_path = model_path
    def predict(self, img):
        try:
            with open("artifact\char_to_num.json", "r") as json_file:
                saved_data = json.load(json_file)
                char_to_num = StringLookup.from_config(saved_data['config'])
                char_to_num.set_weights(saved_data['weights'])
            with open("artifact\\num_to_char.json", "r") as json_file:
                saved_data = json.load(json_file)
                num_to_char = StringLookup.from_config(saved_data['config'])
                num_to_char.set_weights(saved_data['weights'])
                
            model = build_model(200, 100, char_to_num)
            model.load_weights(self.model_path)
            prediction_model = keras.models.Model(model.input[0], model.get_layer(name="dense2").output)
            ids = generate_image_id()
            to_model = encode_single_sample_testing(img, 100, 200, ids)
            
            # Predict
            batch_input = np.expand_dims(to_model['image'], axis=0)
            preds = prediction_model.predict(batch_input)
            pred_texts = decode_batch_predictions(preds, 6, num_to_char)

            # return predictions
            return pred_texts
            
        except Exception as e:
            logging.info(e)
            raise customexception(e, sys)


if __name__ == '__main__':
    obj = PredictionPipline('artifact\\model.weights.h5')
    image_path = 'data\\test_images_mlware\\test_images\\test-100.png'
    text = obj.predict(image_path)
    print(text)

