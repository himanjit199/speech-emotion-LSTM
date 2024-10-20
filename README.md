

# Speech Emotion Recognition Using LSTM

## Overview

This project performs **Speech Emotion Recognition (SER)** using an LSTM model, aiming to classify audio recordings of speech into different emotion categories such as happiness, sadness, anger, etc. SER is a crucial task in human-computer interaction, where the goal is to detect emotions from speech data using machine learning and deep learning techniques.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset contains speech samples labeled with different emotions. Each audio file corresponds to a specific emotion like *happy*, *sad*, *angry*, *fearful*, and more. The dataset used for this project can be sourced from open speech emotion datasets such as the **RAVDESS** or **TESS** datasets.

- **Input**: Audio files (.wav format) of speech recordings.
- **Output**: The emotion label corresponding to each audio sample (e.g., happy, sad, angry).

## Project Structure

1. **Data Preprocessing**:
   - **Audio Loading**: Load and visualize audio files.
   - **Feature Extraction**: Extract Mel Frequency Cepstral Coefficients (MFCCs) from the audio, which serve as input features for the model.
   
2. **LSTM Model**:
   - LSTM is used due to its ability to capture temporal dependencies in sequential data (like speech).
   
3. **Training**:
   - The model is trained to predict the emotion of speech based on audio features.
   
4. **Evaluation**:
   - Evaluate the model using accuracy and loss metrics.
   
5. **Visualization**:
   - Visualize the training process, such as accuracy and loss over epochs.

## Dependencies

You will need the following Python packages:

```bash
pip install numpy pandas matplotlib librosa seaborn tensorflow
```

### Additional Libraries:
- **librosa**: For audio processing and feature extraction.
- **IPython.display.Audio**: To play audio files in the notebook.
- **seaborn**: For data visualization.

## Model Architecture

The model uses the following architecture:

1. **MFCC Extraction**: The speech audio is converted into a sequence of MFCCs (features).
2. **LSTM Layer**: Processes the sequence of MFCC features to retain context from the speech.
3. **Dense Output Layer**: Outputs the predicted emotion (one-hot encoded).

### Example Code for Model Definition:

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')  # Assuming 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Key Features

- **MFCC Extraction**: Mel Frequency Cepstral Coefficients are extracted from each audio sample to represent the sound’s characteristics.
- **LSTM for Sequence Learning**: The LSTM layer captures the temporal dependencies in the audio sequence.

## Performance

After training for **100 epochs**, the model achieves:

- **Training Accuracy**: ~88%
- **Validation Accuracy**: ~85%

### Visualization of Model Performance:

```python
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## How to Run

1. **Prepare Dataset**: Download the speech emotion dataset and extract it into the appropriate directory.
2. **Run Notebook**: Execute the Jupyter notebook or Python script, ensuring all dependencies are installed.
3. **Training**: The model will train for 100 epochs by default, but you can adjust the number of epochs in the `fit()` function.
4. **Evaluation**: Evaluate the model on the test data to measure its performance.

## Conclusion

This project demonstrates how to perform **Speech Emotion Recognition** using an LSTM model. By converting speech into MFCC features and feeding them into an LSTM, we are able to classify emotions from audio samples with high accuracy. This technique can be extended to real-time emotion detection in speech-based applications.

## Acknowledgements

- **librosa**: For providing a powerful library for audio analysis and processing.
- Open datasets like **RAVDESS** or **TESS** for providing labeled speech data.

---




