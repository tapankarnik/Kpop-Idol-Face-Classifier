import os
import numpy as np
from keras.models import load_model
from load_data import load_dataset

def get_embedding(model, face):
    #For FaceNet we need to standardize the input
    std, mean = face.std(), face.mean()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    prediction = model.predict(sample)
    return prediction[0]

def get_embedded_data(trainX, testX):

    model = load_model(os.path.join('model','facenet_keras.h5'))

    newTrainX = list()
    for face in trainX:
        embedding = get_embedding(model, face)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)

    newTestX = list()
    for face in testX:
        embedding = get_embedding(model, face)
        newTestX.append(embedding)
    newTestX = np.asarray(newTestX)
    return newTrainX, newTestX

if __name__=='__main__':
    input_dir = 'data'
    train_dir = os.path.join(input_dir,'processed_data', 'train')
    validation_dir = os.path.join(input_dir,'processed_data', 'validation')

    trainX, trainy = load_dataset(train_dir)
    testX, testy = load_dataset(validation_dir)

    trainX, testX = get_embedded_data(trainX, testX)
    