import joblib
from feature_extract import extract_features
import os
import shutil

classifier_model = joblib.load("music_classifier_svm.pkl")
label_encoder = joblib.load("music_classifier_svm_label_encoder.pkl")

def classify_mp3(file_path):
    feats = extract_features(file_path)
    pred = classifier_model.predict([feats])[0]        
    return label_encoder.inverse_transform([pred])[0]


for file in os.listdir('songs_to_classify/'):
    if file.endswith('.mp3'):
        result = classify_mp3(os.path.join('songs_to_classify',file))
        print(file,">>",result)
        shutil.copy(os.path.join('songs_to_classify',file),os.path.join('music',result))


