import os
import cv2
import numpy as np
from keras.models import load_model

from process_data import detect_face
from load_data import load_dataset
from embedding import get_embedded_data
from train import normalize

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib

def classify(image, face_model, svm_model):
    image = np.expand_dims(image, 0)
    image_embedding = get_embedded_data(face_model, image)
    image_normalized = normalize(image_embedding)
    
    prediction = svm_model.predict(image_normalized)
    label_encode = LabelEncoder()
    label_encode.classes_ = np.load(os.path.join('model','classes.npy'))
    prediction = label_encode.inverse_transform(prediction)
    return prediction

if __name__ == "__main__":
    filename = 'samples/dahyun.png'
    image = cv2.imread(filename)
    image = detect_face(image)
    face_model = load_model('model/facenet_keras.h5')
    svm_model = joblib.load('model/svm_model.sav')
    pred = classify(image, face_model, svm_model)
    print(pred)
