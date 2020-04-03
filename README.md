# Kpop Idol Face Classifier

This is a face recognition model aimed at classifying your favourite Kpop singers.

Uses a FaceNet + SVM model to train and classify faces.

## Requirements

- Keras 2.3.1 ( No need to have a GPU )
- dlib
- sklearn
- numpy
- OpenCV

Install the above (preferebly in a conda environment) and download the data.

Download the FaceNet model from [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn), and place it inside the model folder.

## Data

You can use your own dataset, where each person should have at least 10 images.

The directory structure should look like this

- data
    - raw_data
        - Person_1
            - pic1.jpg
            - pic2.jpg
        - Person 2
            - pic1.jpg
            - pic2.jpg

Now run process.py, which will crop the faces and put them in the processed_data/train folder.

## Usage

Once the data has been prepared, download any picture and put it under the processed_data/test folder. Run main.py with 'python main.py'.

Cheers!
