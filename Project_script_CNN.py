import os
import librosa
import numpy as np
import keras.utils
import pandas as pd
import librosa.util
from tqdm import tqdm
import librosa.display
import soundfile as sf
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from IPython.display import Audio
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Manual classification
# Load the audio file
audio_path = "output_1.wav"
heart, fs = librosa.load(audio_path, sr=16000)
# Select a segment
a = heart[1:100000]
# plot the waveform
plt.plot(a)

# segregate the data into S1, S2, systole and diastole
a1=a[9826:12960]
c1=a[13070:15307]
a2=a[15326:17523]
d1=a[17729:22404]
a3=a[22499:24993]
c2=a[25081:27239]
a4=a[27370:29214]
d2=a[29296:34626]
a5=a[34847:37452]
c3=a[37583:39354]
a6=a[39368:42130]
d3=a[42230:47605]
a7=a[47607:49717]
c4=a[49931:52691]
a8=a[52735:54545]
d4=a[67633:73222]
a9=a[73350:76851]
c5=a[76910:78519]
a10=a[78685:81450]
d5=a[81788:85601]
a11=a[86017:88592]
c6=a[88613:90637]
a12=a[90819:92292]
d6=a[92526:97891]

# Stack the segregated data
s1 = np.vstack((a1,a3,a5,a7,a9,a11))
s2 = np.vstack((a2,a4,a6,a8,a10,a12))
systole = np.vstack((c1,c2,c3,c4,c5,c6))
diastole = np.vstack((d1,d2,d3,d4,d5,d6))

# Save it as a csv file
np.savetxt('S1.csv', s1, delimiter=',')
np.savetxt('S2.csv', s2, delimiter=',')
np.savetxt('Systole.csv', systole, delimiter=',')
np.savetxt('Diastole.csv', diastole, delimiter=',')

# load the csv files as dataframe
df1=pd.read_csv('S1.csv')
df2=pd.read_csv('S2.csv')
df3=pd.read_csv('Systole.csv')
df4=pd.read_csv('Systole.csv')

# convert the data into wave format as S1, S2, S3 and S4
samplerate=16000
sf.write("S1_py.wav", np.ravel(df1), samplerate)
sf.write("S2_py.wav", np.ravel(df2), samplerate)
sf.write("S3_py.wav", np.ravel(df3), samplerate)
sf.write("S4_py.wav", np.ravel(df3), samplerate)

# 4 classes : 0 S1, 1 S2, 2 systole, 3 Diastole
class_names = ['S1','S2', 'Systole', 'Diastole']
labels = {'S1' : 0,  'S2' : 1,'Systole' : 2, 'Diastole' : 3}

# Separate the data into x and y
x = []
y = class_names #or labels

for i in range(4):
    file_path = os.path.join('S'+str(i+1)+'_py.wav')
    # @note: resampling can take some time!
    signal, _ = librosa.load(file_path, sr=16000, mono=True, duration=20, 
                             dtype=np.float32)
    x.append(signal)

# Slice the data into frames
x_framed = []
y_framed = []

# choose frame_length and hop_length
for i in range(len(x)):
    frames = librosa.util.frame(x[i], frame_length=16896, hop_length=512)
    x_framed.append(np.transpose(frames))
    y_framed.append(np.full(frames.shape[1], y[i]))
    
# merging sliced frames (x) and labels (y)
 # Each frame is used to generate Mel spectrogram
x_framed = np.asarray(x_framed)
 # Labels are added to each frame
y_framed = np.asarray(y_framed)
x_framed = x_framed.reshape(x_framed.shape[0]*x_framed.shape[1], 
                            x_framed.shape[2])
y_framed = y_framed.reshape(y_framed.shape[0]*y_framed.shape[1], )
print("x_framed shape: ", x_framed.shape) 
print("y_framed shape: ", y_framed.shape)

# Feature extraction using Mel spectrogram
x_features = []
y_features = y_framed

for frame in tqdm(x_framed):
    # Generate a mel-scaled spectrogram
    S_mel = librosa.feature.melspectrogram(y=frame, sr=16000, n_mels=30, 
                                           n_fft=1024, 
                                           hop_length=512, center=False)
# Scale the spectrogram according to reference power so that all 
# the frequencies are visible
    S_mel = S_mel / S_mel.max()
    # Convert to power scale (dB)
    S_log_mel = librosa.power_to_db(S_mel, top_db=80.0)
    x_features.append(S_log_mel)

# Convert the features into numpy array
x_features = np.asarray(x_features)

# Flatten features for scaling
x_features_r = np.reshape(x_features, (len(x_features), 30*32))

# Create a feature scaler
scaler = preprocessing.StandardScaler().fit(x_features_r)

# Apply the feature scaler 
x_features_s = scaler.transform(x_features_r)

# Convert labels to categorical one-hot encoding
code = y_features

# Output data preparation
label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(code)
y_features_cat = keras.utils.to_categorical(vec, num_classes=len(class_names))

# Split the data into test, validation and train sets 
x_train, x_test, y_train, y_test = train_test_split(x_features_s,
                                                    y_features_cat,
                                                    test_size=0.25,
                                                    random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  random_state=1)

print('Training samples:', x_train.shape)
print('Validation samples:', x_val.shape)
print('Test samples:', x_test.shape)

# Building the CNN Model
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(30, 32, 1), 
                        data_format='channels_last'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(9, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

# print model summary for reference
model.summary()

# Choose an optimizer and choose a learning rate
sgd = optimizers.SGD (learning_rate=0.01, momentum=0.9, nesterov=True)

# compile the model and choose the evaluation metrics
model.compile(optimizer=sgd, loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Reshape features to include channel
x_train_r = x_train.reshape(x_train.shape[0], 30, 32, 1)
x_val_r = x_val.reshape(x_val.shape[0], 30, 32, 1)
x_test_r = x_test.reshape(x_test.shape[0], 30, 32, 1)

# Model history
# Train the model
history = model.fit(x_train_r, y_train, validation_data=(x_val_r, y_val),
                    batch_size=600, epochs=50, verbose=1)
# Save the model into an HDF5 file ‘model.h5’
model.save('CNN_4_class.h5')

# Convert the model into tflite format
def representative_dataset_gen():
    for i in range(len(x_train_r)):
        yield [x_train_r[i].reshape((1, ) + x_train_r[i].shape)]

converter = tf.lite.TFLiteConverter.from_keras_model_file("CNN_4_class.h5" )
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('CNN_4_class.tflite','wb') as f:
    f.write(tflite_model)
    
#%%

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.clf()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_loss, color='tab:blue', label='Training Loss')
plt.plot(val_loss, color='chocolate', label='Validation Loss')
plt.title('CNN-4 class model', fontsize=10)
plt.legend()
# plt.savefig('Loss_epoch_CNN_4class_1.png', dpi=400)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure()
plt.clf()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(acc, color='r', label='accuracy')
plt.plot(val_acc, color='g', label='validation accuracy')
plt.legend()

# Evaluate Accuracy
# Next, compare how the model performs on the test dataset:
print('Evaluate model:')
results = model.evaluate(x_test_r, y_test)
print(results)
print('Test loss: {:f}'.format(results[0]))
print('Test accuracy: {:.2f}%'.format(results[1] * 100))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(x_test_r)

y_pred_class_nb = np.argmax(y_pred, axis=1)
y_true_class_nb = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_class_nb, y_pred_class_nb)
np.set_printoptions(precision=2)
print("Accuracy = {:.2f}%".format(accuracy * 100))

classes = class_names
cm = confusion_matrix(y_true_class_nb, y_pred_class_nb, labels=[0,1,2])

# (optional) normalize to get values in %
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='True labels', xlabel='Predicted labels')

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.title('CNN-4 class', fontsize=11)
plt.imshow(cm, cmap=plt.cm.Blues)
# plt.savefig('confusion_CNN_4class_1.png', dpi=400)
#%%

# Assuming you have the test data (x_test_r) and the true labels (y_test) in one-hot encoded format

# Get the model's predictions on the test data
predictions = model.predict(x_test_r)

# Convert the predicted probabilities to class labels (one-hot to integer labels)
predicted_labels = np.argmax(predictions, axis=1)

# Convert true labels from one-hot to integer labels
true_labels = np.argmax(y_test, axis=1)

# Calculate the accuracy for each class
num_classes = len(np.unique(true_labels))
class_accuracy = np.zeros(num_classes)

for class_idx in range(num_classes):
    class_mask = true_labels == class_idx
    correct_predictions = np.sum(predicted_labels[class_mask] == true_labels[class_mask])
    total_instances = np.sum(class_mask)
    class_accuracy[class_idx] = correct_predictions / total_instances

# Print the accuracy for each class
for class_idx in range(num_classes):
    print(f"Accuracy for Class {class_idx}: {class_accuracy[class_idx]:.2f}")


#%%
import numpy as np
from scipy.signal import filtfilt, butter
from scipy.io import wavfile

samplerate, data_2 = wavfile.read('output_1.wav')
audio_signal = data_2[4000:100000]
fs = 16000
def amplitude_envelope(audio_signal, fs, cutoff_freq=100):
    # Rectify the audio signal
    rectified_signal = np.abs(audio_signal)
    
    # Design a low-pass Butterworth filter
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    # Apply the low-pass filter (forward and backward filtering for zero-phase distortion)
    smoothed_signal = filtfilt(b, a, rectified_signal)
    
    return smoothed_signal

# Example usage:
# Assuming 'audio_signal' is the 1-dimensional audio signal, and 'fs' is the sampling rate in Hz.
# 'cutoff_freq' is the cut-off frequency of the low-pass filter.

envelope = amplitude_envelope(audio_signal, fs, cutoff_freq=100)

#%%
predict_prob = model.predict(x_train_r)
def apply_voting_with_sliding_window(predict_prob, window_length, tie_break='random'):
    """
    Apply the voting algorithm with a sliding window to smooth out segmentation output.

    Args:
        predict_prob (numpy array): Predicted probabilities from the model.
                                   Shape: (num_data_points, num_classes).
        window_length (int): Length of the sliding window. Default is 9.
        tie_break (str): How to handle ties in voting. 'random' for random choice,
                         'first' to select the first class in case of ties.

    Returns:
        numpy array: Final class labels after post-processing.
                     Shape: (num_data_points,).
    """
    final_classifications = []
    num_classes = predict_prob.shape[1]

    for i in range(len(predict_prob)):
        # Calculate the starting and ending indices for the sliding window
        start_index = max(0, i - window_length)
        end_index = min(len(predict_prob), i + window_length)

        # Extract the window of probabilities
        window = predict_prob[start_index:end_index]

        # Calculate the vote counts for each class within the window
        vote_counts = np.sum(window, axis=0)

        # Determine the most prevalent class (the one with the highest vote count)
        most_prevalent_classes = np.where(vote_counts == np.max(vote_counts))[0]

        # Handle ties if multiple classes have the same vote count

        # Append the most prevalent class to the final classifications
        final_classifications.append(most_prevalent_classes)

    return np.array(final_classifications)

# Example usage:
# Assuming 'predict_prob' is the predicted probabilities from the CNN model
# 'predict_prob' should be a NumPy array of shape (num_data_points, num_classes)

# Apply the voting algorithm with a sliding window of length 9 and handle ties randomly
final_classifications = apply_voting_with_sliding_window(predict_prob, window_length=15, tie_break='random')

print("Final Classifications after post-processing:", final_classifications)
plt.plot(final_classifications[1:200])


#%%

import numpy as np

# Suppose 'predict_prob' contains the probabilities predicted by the model

# Define the length of the sliding window
window_length = 11

# Create an empty array to store the final classifications
final_classifications = []

# Apply the sliding window to each prediction time point
for i in range(len(predict_prob)):
    # Calculate the starting and ending indices for the sliding window
    start_index = max(0, i - window_length // 2)
    end_index = min(len(predict_prob), i + window_length // 2)
    
    # Extract the window of probabilities
    window = predict_prob[start_index:end_index]
    
    # Assign the final classification for the current time point
    final_classifications.append(np.argmax(predict_prob[i]))

print(final_classifications)
plt.plot(final_classifications[0:10])


import numpy as np

# Suppose 'predict_prob' contains the probabilities predicted by the model

# Define the length of the sliding window
window_length = 11

# Create an empty array to store the final classifications
final_classifications = []

# Apply the sliding window to each prediction time point
for i in range(len(predict_prob)):
    # Calculate the starting and ending indices for the sliding window
    start_index = max(0, i - window_length // 2)
    end_index = min(len(predict_prob), i + window_length // 2)
    
    # Extract the window of probabilities
    window = predict_prob[start_index:end_index]
    
    # Assign the final classification for the current time point
    final_classifications.append(np.argmax(predict_prob[i]))

print(final_classifications)
plt.plot(final_classifications[0:10])
[1, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 1, 0, 2, 2, 1, 2, 0, 2, 1, 1, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 0, 2, 0, 2, 2, 2, 1, 1, 1, 1, 0, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 0, 1, 0, 0, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1, 0, 0, 2, 0, 1, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 1, 1, 1, 2, 0, 1, 0, 1, 0, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 1, 1, 1, 1, 0, 2, 1, 1, 2, 0, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 1, 1, 2, 0, 2, 2, 2, 1, 1, 0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 1, 0, 0, 2, 1, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 1, 0, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 2, 1, 2, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 0, 2, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 0, 2, 0, 1, 2, 0, 0, 0, 2, 0, 2, 0, 1, 2, 2, 2, 2, 1, 1, 0, 2, 1, 1, 0, 1, 1, 2, 0, 1, 1, 0, 2, 2, 1, 2, 2, 0, 1, 2, 1, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 1, 0, 2, 0, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 0, 1, 1, 0, 1, 0, 2, 0, 2, 2, 2, 1, 2, 0, 2, 1, 2, 1, 2, 0, 0, 2, 2, 0, 2, 1, 0, 0, 2, 2, 0, 2, 2, 1, 0, 1, 2, 1, 1, 1, 1, 1, 0, 2, 0, 0, 1, 1, 2, 0, 2, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 2, 2, 0, 1, 0, 1, 2, 0, 2, 1, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 0, 1, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 2, 2, 2, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 2, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 1, 2, 2, 1, 0, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 2, 0, 2, 2, 1, 0, 2, 2, 1, 1, 2, 1, 2, 2, 0, 1, 2, 1, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 2, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 0, 2, 1, 1, 1, 1, 0, 1, 2, 0, 1, 0, 2, 2, 1, 0, 0, 1, 2, 2, 0, 2, 1, 2, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 2, 1, 1, 1, 0, 0, 1, 2, 2, 0, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2, 2, 0, 0, 2, 2, 1, 0, 1, 2, 1, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 2, 0, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2, 2, 1, 0, 2, 2, 0, 2, 0, 0, 0, 1, 1, 2, 1, 2, 0, 1, 0, 0, 2, 2, 1, 0, 1, 1, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2, 2, 2, 1, 1, 2, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 1, 2, 2, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 1, 1, 2, 0, 1, 0, 1, 0, 0, 0, 2, 1, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 0, 2, 2, 1, 2, 2, 2, 0, 0, 2, 1, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 2, 2, 2, 1, 2, 0, 2, 1, 2, 2, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 1, 2, 1, 1, 0, 0, 1, 2, 2, 0, 2, 0, 2, 2, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 1, 2, 0, 1, 1, 0, 2, 1, 0, 1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 2, 0, 2, 0, 2, 1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 0, 1, 1, 2, 2, 1, 2, 2, 0, 1, 1, 1, 2, 0, 2, 1, 0, 1, 0, 0, 0, 2, 2, 2, 2, 1, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 1, 0, 1, 2, 1, 1, 0, 0, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 2, 1, 0, 0, 2, 1, 1, 2, 0, 1, 0, 0, 1, 1, 1, 2, 1, 0, 0, 0, 1]