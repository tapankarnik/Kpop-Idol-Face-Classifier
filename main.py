import os
import numpy as np
from load_data import load_images, load_dataset
from process_data import detect_face
from embedding import get_embedded_data
from train import get_svm_model, normalize

from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import joblib


if __name__=='__main__':
    input_dir = os.path.join('data', 'processed_data','test')

    images = load_images(input_dir)
    cropped_images = list()

    for i in range(len(images)):
        cropped_images.append(detect_face(images[i]))

    face_model = load_model(os.path.join('model', 'facenet_keras.h5'))
    cropped_images = get_embedded_data(face_model, cropped_images)

    cropped_images = normalize(cropped_images)

    model = joblib.load(os.path.join('model', 'svm_model.sav'))

    pred_test = model.predict(cropped_images)
    pred_proba = model.predict_proba(cropped_images)

    label_encode = LabelEncoder()
    label_encode.classes_ = np.load(os.path.join('model','classes.npy'))
    predicted_names = label_encode.inverse_transform(pred_test)
    
    for i, image in enumerate(images):
        # plt.figure()
        plt.imshow(image)
        plt.title("Predicted: "+predicted_names[i]+" with "+str(round(pred_proba[i][pred_test[i]]*100, 2))+"% confidence.")
        plt.show()
