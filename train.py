import os
import numpy as np
from keras.models import load_model

from load_data import load_dataset
from embedding import get_embedded_data

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib

def normalize(trainX):
    norm_encoder = Normalizer(norm='l2')
    trainX = norm_encoder.transform(trainX)
    return trainX

def get_svm_model(trainX, trainy):

    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    return model

def train_svm():
    input_dir = 'data'
    train_dir = os.path.join(input_dir,'processed_data', 'train')
    validation_dir = os.path.join(input_dir,'processed_data', 'validation')

    trainX, trainy = load_dataset(train_dir)
    testX, testy = load_dataset(validation_dir)

    model = load_model(os.path.join('model','facenet_keras.h5'))

    trainX = get_embedded_data(model, trainX)
    testX = get_embedded_data(model, testX)

    trainX = normalize(trainX)
    testX = normalize(testX)
    
    label_encode = LabelEncoder()
    label_encode.fit(trainy)
    trainy = label_encode.transform(trainy)
    testy = label_encode.transform(testy)

    np.save(os.path.join('model','classes.npy'), label_encode.classes_)

    model = get_svm_model(trainX, trainy)
    filename = os.path.join('model', 'svm_model.sav')
    joblib.dump(model, filename)
    print("SVM model saved!")

    pred_train = model.predict(trainX)
    pred_test = model.predict(testX)
    score_train = accuracy_score(trainy, pred_train)
    score_test = accuracy_score(testy, pred_test)

    print("Accuracy\nTrain : ",score_train,"\n","Test : ", score_test)

if __name__ == "__main__":
    train_svm()
