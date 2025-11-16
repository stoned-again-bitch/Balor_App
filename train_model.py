import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import joblib
from feature_extract import extract_features


class_dirs = {
        'happy': 'happy/',
        'sad': 'sad/',
        'angry': 'angry/',
        'surprise': 'surprise/',
        'disgust': 'disgust/',
        'neutral': 'neutral/'
        }

def load_data(class_dirs):
    features, labels = [], []
    le = LabelEncoder()
    class_names = list(class_dirs.keys())
    le.fit(class_names)

    for class_name, dir_path in class_dirs.items():
        for file in os.listdir(dir_path):
            if file.endswith('.mp3'):
                feats = extract_features(os.path.join(dir_path, file))
                features.append(feats)
                labels.append(class_name)


    X = np.array(features)
    y = le.transform(labels)
    return X, y, le

X, y, le = load_data(class_dirs)

print(X, y, le)

## TRAINING ##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

print("Trained on", X_train)
print("Will be tested on", X_test)
print("Accuracy: ",accuracy_score(y_test, clf.predict(X_test)))

joblib.dump(clf, "music_classifier_svm.pkl")
joblib.dump(le, "music_classifier_svm_label_encoder.pkl")








