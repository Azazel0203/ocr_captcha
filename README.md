
# OCR Captcha Project

This project implements an Optical Character Recognition (OCR) system for solving CAPTCHA challenges. It consists of several modules for data ingestion, model training, and inference. The system is designed to recognize characters in images and produce corresponding text predictions.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [License](#license)

## Introduction

The OCR Captcha Project is developed to tackle the problem of recognizing text in CAPTCHA images. It employs machine learning techniques, particularly convolutional neural networks (CNNs) and recurrent neural networks (RNNs), to perform character recognition. The project is divided into several components:

- **Data Ingestion**: Ingests raw image and label data, performs preprocessing, and prepares datasets for training.
- **Model Training**: Builds and trains deep learning models on the prepared datasets.
- **Inference**: Uses trained models to make predictions on new CAPTCHA images.

## Project Structure

The project is structured as follows:

- **`src/`**: Contains the source code for the project.
  - **`ocr_captcha/`**: Main package for the OCR CAPTCHA project.
    - **`data_ingestion/`**: Module for data ingestion and preprocessing.
    - **`model_training/`**: Module for building, training, and saving models.
    - **`inference/`**: Module for making predictions on new images.
    - **`utils/`**: Utility functions used across modules.
  - **`app.py`**: Main script for running the OCR CAPTCHA system.
- **`requirements.txt`**: File listing all Python dependencies required by the project.
- **`Dockerfile`**: Instructions for building a Docker image for the project.
- **`README.md`**: This file, containing information about the project.

## Setup and Installation

To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python dependencies using `pip install -r requirements.txt`.
3. Optionally, build a Docker image using the provided Dockerfile.

## Usage

To use the OCR CAPTCHA system, you can follow these steps:

1. Run the data ingestion module to prepare the training and testing datasets.
2. Train the models using the prepared datasets.
3. Once trained, you can use the trained models for making predictions on new CAPTCHA images.

Detailed instructions for each step can be found in the respective module's documentation.

## Dependencies

The project relies on several Python packages for its functionality. The main dependencies include:

- TensorFlow
- NumPy
- pandas
- scikit-learn
- Keras

For a complete list of dependencies, refer to the `requirements.txt` file.

## Model Architecture

The OCR CAPTCHA model architecture consists of several layers:

1. **Convolutional Layers**: Extract features from input images.
2. **Pooling Layers**: Reduce spatial dimensions of feature maps.
3. **Reshape Layer**: Prepare feature maps for input to the RNN.
4. **Dense Layers**: Fully connected layers for further feature extraction.
5. **Bidirectional LSTM Layers**: Recurrent layers to handle sequence data.
6. **Dense Output Layer**: Output layer with softmax activation for character classification.
7. **CTC Loss Layer**: Connectionist Temporal Classification layer for training sequence-to-sequence models.

Here's a code snippet illustrating the model architecture:

```python
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
```

## Model Evaluation

After training the model, it's essential to evaluate its performance. Here's a code snippet illustrating how to evaluate the model using test data:

```python
def evaluate_model(model: Model, test_dataset: Dataset) -> Tuple[float, float]:
    # Initialize variables to store total loss and accuracy
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    # Iterate over the test dataset
    for batch in test_dataset:
        # Unpack the batch
        images, labels = batch
        input_length = np.ones(labels.shape[0]) * model.input_shape[1]
        label_length = np.ones(labels.shape[0]) * model.output_shape[1]

        # Perform inference
        predictions = model.predict([images, labels, input_length, label_length])

        # Calculate loss
        batch_loss = ctc_batch_cost(labels, predictions, input_length, label_length)

        # Update total loss
        total_loss += tf.reduce_mean(batch_loss)

        # Calculate accuracy
        total_accuracy += accuracy(labels, predictions)

        # Increment the number of batches
        num_batches += 1

    # Calculate average loss and accuracy
    average_loss = total_loss / num_batches
    average_accuracy = total_accuracy / num_batches

    return average_loss, average_accuracy

# Evaluate the model
test_loss, test_accuracy = evaluate_model(model, test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
```

## Conclusion

The OCR Captcha Project aims to provide a solution for recognizing text in CAPTCHA images using machine learning techniques. By following the steps outlined in this README, users can set up, train, and evaluate the OCR system. Additionally, the provided code snippets demonstrate the model architecture, training process, and evaluation procedure, enabling users to understand and customize the project according to their requirements.

For more detailed documentation and examples, please refer to the source code and accompanying documentation within each module.

## Additional Information

### Data Ingestion
The \`DataIngestion\` class in \`data_ingestion.py\` is responsible for ingesting raw image and label data, performing data preprocessing tasks such as train-test split, and saving the processed data for training. It utilizes Pandas for data manipulation and provides methods to initiate the data ingestion process.

### Model Trainer
The \`ModelTrainer\` class in \`model_trainer.py\` handles the training of the OCR model. It takes preprocessed data, builds the model architecture, trains the model, and saves the trained weights. The class also provides methods for evaluating the model's performance.

### Model Deployment
To deploy the OCR model for inference, users can integrate it into a web application or API service. The model can accept images as input, process them using the trained model, and return the predicted text. Flask or FastAPI can be used to create the API endpoints for model inference.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
