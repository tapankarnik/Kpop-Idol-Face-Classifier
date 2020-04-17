import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from keras.models import load_model
from classify import classify

app = Flask(__name__)

def load():
    global face_model, svm_model, names
    face_model = load_model(os.path.join('model','facenet_keras.h5'))
    svm_model = joblib.load(os.path.join('model','svm_model.sav'))
    names = np.load(os.path.join('model','classes.npy'))

@app.route('/predict',methods=['POST'])
def predict():


    filestr = request.files['image'].read()
    npimg = np.fromstring(filestr, np.uint8)
    image  = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    pred = classify(image, face_model, svm_model)
    return pred

@app.route('/help', methods=['GET'])
def helpfunc():
    s = "I can recognize "
    for i in range(len(names)):
        if i==len(names)-1:
                s += " and "+names[i]
        else:
                s += " "+names[i]+","

    return s

if __name__ == "__main__":
    print("Loading model... Please wait.")
    load()
    app.run(debug=True, use_reloader=False, threaded=False)
