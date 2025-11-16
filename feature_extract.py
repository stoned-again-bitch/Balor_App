import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    return np.hstack([mfccs, tempo, centroid, zcr])

