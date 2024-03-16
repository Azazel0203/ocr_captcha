import numpy as np
import tensorflow as tf
from keras.src.layers.preprocessing.string_lookup import StringLookup
from keras import layers
import keras
from tensorflow.python.framework.ops import SymbolicTensor, EagerTensor
from tensorflow.python.framework.sparse_tensor import SparseTensor
from typing import Dict, Union
from keras.src.models.functional import Functional
from typing import Tuple, List
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import pickle
import sys
import uuid


def generate_image_id():
    image_id = uuid.uuid4()
    return str(image_id)


def split_data(images: np.ndarray, labels: np.ndarray, train_size: float = 0.8, shuffle: bool =True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

def encode_single_sample_training(img_path: str, label: str, img_height: int, img_width: int, char_to_num:StringLookup) -> Dict[str, Union[tf.Tensor, tf.Tensor]]:
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # heightxwidth -> widthxheight
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

def ctc_label_dense_to_sparse(labels: SymbolicTensor, label_lengths: SymbolicTensor) -> SparseTensor:
    label_shape = tf.shape(labels) # B, T, C
    # print("label_shape:", label_shape)
    num_batches = tf.stack([label_shape[0]])
    # print("num_batches:", num_batches)
    max_num_labels = tf.stack([label_shape[1]])
    # print("max_num_labels:", max_num_labels)
    def range_less_than(old_input, current_input):
        '''
        Creates a boolean mask for the label_lengths we need to pay attention to
        '''
        return tf.expand_dims(
            tf.range(tf.shape(old_input)[1]), 0) < tf.fill(max_num_labels, current_input)
    init = tf.cast(tf.fill([1, label_shape[1]], 0), tf.bool)
    '''
    1. Initialize a tensor filled with zeros with shape 1,T
    2. type_Cast it to bool -> tensor of False
    '''
    # print("init:", init)
    dense_mask = tf.compat.v1.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
    # print("dense_mask:", dense_mask)
    dense_mask = dense_mask[:, 0, :]
    # squeeze the middle dimension
    # print("dense_mask after squeeze:", dense_mask)
    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches), label_shape)
    # print("label_array:", label_array)
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)
    # print("label_ind:", label_ind)
    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]), max_num_labels), tf.reverse(label_shape, [0]),))
    # print("batch_array:", batch_array)
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    # print("batch_ind:", batch_ind)
    indices = tf.transpose(tf.reshape(tf.concat([batch_ind, label_ind], axis=0), [2, -1]))
    # print("indices:", indices)
    vals_sparse = tf.compat.v1.gather_nd(labels, indices)
    # print("vals_sparse:", vals_sparse)
    return tf.SparseTensor(
        tf.cast(indices, tf.int64), vals_sparse, tf.cast(label_shape, tf.int64)
    )

def ctc_batch_cost(y_true: SymbolicTensor, y_pred: SymbolicTensor, input_length: SymbolicTensor, label_length: SymbolicTensor) -> SymbolicTensor:
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    # print(f"label_length: {label_length}")
    '''
    1. removes the dimensions of size 1, here its removing the last dimension
        if the value of last dimension of label_length is 1, it will remove that
    2. casting the label_length to int32
    '''
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    # print(f"input_length: {input_length}")
    '''
    This is basically label_length but for y_pred
    '''
    sparse_labels = tf.cast(ctc_label_dense_to_sparse(y_true, label_length), tf.int32)
    # print(f"sparse_labels: {sparse_labels}")
    '''
    Generating a concentrated sparse matrix
    '''
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0 ,2]) + keras.backend.epsilon())
    # print(f"y_pred: {y_pred}")
    '''
    Add a small value of epsilon before taking log...so as to not take the log of 0 by mistake
    '''
    '''
    returns the actual loss value
    add a singleton dim to the output...so as to represent the batch size
    '''
    return tf.expand_dims(tf.compat.v1.nn.ctc_loss(
        inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ), 1,
    )

class CTCLayer(layers.Layer):
    def __init__(self, trainable=True, name: str=None, dtype=None):
        super().__init__(name=name)
        self.trainable = trainable
        self.loss_fn = ctc_batch_cost

    def call(self, y_true: SymbolicTensor, y_pred: SymbolicTensor) -> SymbolicTensor:
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # At test time, just return the computed predictions
        return y_pred
    
    def get_config(self):
        config = super().get_config()
        config.update({'trainable': self.trainable})
        return config


def build_model(img_width: int, img_height: int, char_to_num: StringLookup) -> Functional:
    # inputs of the model
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    # first conv block
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv1",)(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    # 2nd conv block
    x = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2",)(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    # reshaping just in case
    new_shape = ((img_width//4), (img_height//4) * 128) # each spatial location will be denoted by 64 values
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    # RNNs
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    # output_layer
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)
    output = CTCLayer(name="ctc_loss")(labels, x)
    # define the model
    model = keras.models.Model(
        inputs=[input_img, labels],
        outputs=output,
        name="ocr_model_v1"
    )
    # learning rate
    # Define the initial learning rate and decay rate
    initial_learning_rate = 0.001
    decay_rate = 0.9
    # Define the learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,  # Adjust decay steps as needed
        decay_rate=decay_rate,
        staircase=True  # Set to True for staircase decay
    )
    beta_1 = 0.9
    beta_2 = 0.999
    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1, beta_2=beta_2)
    # compile the model
    model.compile(optimizer=opt)
    return model

# def save_model(model, optimizer, model_path):
#     model.save_weights(model_path)
#     with open("artifact\optimizer.json", "w") as opt_json_file:
#         opt_json_file.write(optimizer.get_config())

def encode_single_sample_testing(img_path: str, img_height: int, img_width: int, ids: str) -> Dict[str, Union[tf.Tensor, tf.Tensor]]:
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # heightxwidth -> widthxheight
    # 6. Map the characters in label to numbers
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "ids": ids}

def ctc_decode(y_pred, input_length, greedy, beam_width=100, top_paths=1) -> Tuple[list, tf.Tensor]:
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon())
    input_length = tf.cast(input_length, tf.int32)
    # print (f"y_pred = {tf.shape(y_pred)}, input_length = {tf.shape(input_length)}")
    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

def decode_batch_predictions(pred, max_length:int, num_to_char: StringLookup) -> List:
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise customexception(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception Error occured in loading object")
        raise customexception(e, sys)

















