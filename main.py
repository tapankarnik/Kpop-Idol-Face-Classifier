import os
from load_data import load_images, load_dataset
from process_data import detect_face
from embedding import get_embedded_data
from classify import get_svm_model, normalize

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


if __name__=='__main__':
    input_dir = os.path.join('data', 'processed_data','test')
    train_dir = os.path.join('data', 'processed_data','train')

    images = load_images(input_dir)
    cropped_images = list()

    for i in range(len(images)):
        cropped_images.append(detect_face(images[i]))
    trainX, trainy = load_dataset(train_dir)
    
    trainX, cropped_images = get_embedded_data(trainX, cropped_images)

    trainX, cropped_images = normalize(trainX, cropped_images)
    label_encode = LabelEncoder()
    label_encode.fit(trainy)
    trainy = label_encode.transform(trainy)

    model = get_svm_model(trainX, trainy)

    pred_test = model.predict(cropped_images)
    pred_proba = model.predict_proba(cropped_images)

    predicted_names = label_encode.inverse_transform(pred_test)
    
    for i, image in enumerate(images):
        # plt.figure()
        plt.imshow(image)
        plt.title("Predicted: "+predicted_names[i]+" with "+str(round(pred_proba[i][pred_test[i]]*100, 2))+"% confidence.")
        plt.show()