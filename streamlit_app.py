import streamlit as st
import librosa
import numpy as np
from joblib import load
import sounddevice as sd
import io
import matplotlib.pyplot as plt

# Load the pre-trained model
model_filename = 'model.joblib'
model = load(model_filename)

# Create a function to extract features from audio
def extract_features(y, sr, mfcc=True, chroma=True, mel=True):
    features = []

    # Convert audio data to floating-point
    y = y.astype(float)

    if mfcc:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.append(np.mean(mfccs, axis=1))
    if chroma:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma, axis=1))
    if mel:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features.append(np.mean(mel, axis=1))

    return np.concatenate(features)

# def record_audio(device_index=0):
#     seconds = 5
#     sample_rate = 44100

#     # Print available devices
#     devices = sd.query_devices()
#     print(devices)

#     recording = sd.rec(
#         int(sample_rate * seconds),
#         samplerate=sample_rate,
#         channels=1,
#         dtype=np.float32,
#         device=device_index
#     )
#     sd.wait()
#     return recording.flatten(), sample_rate

# def plot_waveform(y, sr):
#     plt.figure(figsize=(10, 4))
#     plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y)
#     plt.title('Audio Waveform')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     st.pyplot(plt)

def main():
    st.title("Speech Emotion Recognition App")
    st.markdown(
        """
        This app predicts the emotion in an audio recording. You can  upload an audio file .
        """
    )

    # # Record audio button
    # if st.button("Record Audio"):
    #     st.write("Recording... Speak now!")
        
    #     # Use the default device index (0) or update it based on the available devices
    #     recording, sample_rate = record_audio()
    #     st.write("Recording complete!")

    #     # Extract features from the recorded audio
    #     features = extract_features(recording, sample_rate, mfcc=True, chroma=True, mel=True)

    #     # Reshape features to match the model's input shape
    #     features = features.reshape(1, -1)

    #     # Make prediction using the loaded model
    #     prediction = model.predict(features)[0]

    #     # Display the predicted emotion
    #     st.subheader(f"Predicted Emotion: {prediction}")

    #     # Display audio waveform
    #     plot_waveform(recording, sample_rate)

    # Display the audio file if uploaded
    uploaded_file = st.file_uploader("Or upload an audio file (in WAV format)", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Extract features from the uploaded audio file
        y, sr = librosa.load(uploaded_file, sr=None)
        features = extract_features(y, sr, mfcc=True, chroma=True, mel=True)

        # Reshape features to match the model's input shape
        features = features.reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(features)[0]

        st.subheader("Predicted Emotions:")
        st.write("<div style='padding: 10px; background-color: #f4f4f4; border-radius: 5px;'>", unsafe_allow_html=True)
        for emotion in predictions:
            styled_emotion = f"<span style='margin-right: 10px; padding: 5px; background-color: #3498db; color: #fff; border-radius: 3px; text-transform: uppercase; font-size: 16px;'>{emotion}</span>"
            st.write(styled_emotion, unsafe_allow_html=True)
        st.write("</div>", unsafe_allow_html=True)

        # # Display audio waveform
        # plot_waveform(y, sr)

# Run the app
if __name__ == "__main__":
    main()
