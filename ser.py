import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Define the emotions dictionary
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions we want to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Function for extracting features from sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Load the data and extract features for each sound file
def load_data(dataset_path, test_size=0.2, save_features=True, load_features=False):
    features_file = 'extracted_features.pkl'

    if load_features and os.path.exists(features_file):
        # Load features from the file if it exists
        with open(features_file, 'rb') as file:
            features = pickle.load(file)
        x_train, x_test, y_train, y_test = train_test_split(features['x'], features['y'], test_size=test_size, random_state=9)
        return x_train, x_test, y_train, y_test

    x, y = [], []
    for folder in glob.glob(os.path.join(dataset_path, '*')):
        print(folder)
        for file in glob.glob(os.path.join(folder, '*.wav')):
            file_name = os.path.basename(file)
            emotion = emotions[file_name.split('-')[2]]
            if emotion not in observed_emotions:
                continue
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)

    # Return outside of the inner loop
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Specify the path to your dataset
dataset_path = r'C:\Users\Local11\Desktop\ser\content'
x_train, x_test, y_train, y_test = load_data(dataset_path, test_size=0.2)
# Reshape features
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Shape of train and test set and Number of features extracted
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')

# Initialise Multi-Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

model.fit(x_train, y_train)

# Save the trained model to a file
model_filename = 'model.joblib'
dump(model, model_filename)

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Save the features to a file (optional)
features_filename = 'extracted_features.pkl'
with open(features_filename, 'wb') as file:
    pickle.dump({'x': x_train, 'y': y_train}, file)

print(f'Model saved to {model_filename}')
print(f'Features saved to {features_filename}')
