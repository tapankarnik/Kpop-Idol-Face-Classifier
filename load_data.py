import os
import numpy as np
import cv2

def load_images(directory):
    filenames = os.listdir(directory)
    faces = list()
    for filename in filenames:
        face = cv2.imread(os.path.join(directory, filename))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue

        faces = load_images(path)

        labels = [subdir for _ in range(len(faces))]

        print("Loaded ", len(faces), " examples for class ", subdir, ".")

        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


if __name__=='__main__':
    input_dir = 'data'
    train_dir = os.path.join(input_dir,'processed_data', 'train')
    validation_dir = os.path.join(input_dir,'processed_data', 'validation')

    trainX, trainy = load_dataset(train_dir)
    testX, testy = load_dataset(validation_dir)

    

    
    


    