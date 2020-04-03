import argparse
import dlib
import cv2
import os
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description="crops images")
    parser.add_argument("--input", type=str, default=os.path.join('data','raw_data'), help="Enter the input foldername of the images.")
    parser.add_argument("--output", type=str, default=os.path.join('data','processed_data','train'), help="Enter the output foldername of the images.")
    return parser.parse_args()

def detect_face(image):
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 0)
    num_faces = len(rects)
    if num_faces==1:
        x1 = rects[0].left()
        y1 = rects[0].top()
        x2 = rects[0].right()
        y2 = rects[0].bottom()

        cropped_image = crop_image(image, x1, y1, x2, y2)
        cropped_image = cv2.resize(cropped_image, (160, 160))
        return cropped_image
    else:
        print("Not 1 face!")
        return None

def crop_image(image, x1, y1, x2, y2):
    cropped_image = image[y1:y2, x1:x2, :]
    return cropped_image

if __name__ == '__main__':
    
    args = get_arguments()
    input_dir = args.input
    output_dir = args.output
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    class_list = os.listdir(input_dir)
    for class_name in class_list:
        if not os.path.exists(os.path.join(output_dir,class_name)):
            os.makedirs(os.path.join(output_dir,class_name))

    for class_name in class_list:
        filenames = [f for f in os.listdir(os.path.join(input_dir, class_name))]
        
        # predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat') 

        for filename in filenames:
            print("Loaded ", filename)
            image = cv2.imread(os.path.join(input_dir, class_name, filename))
            try:
                cropped_image = detect_face(image)
                if cropped_image is None:
                    continue
                cv2.imwrite(os.path.join(output_dir, class_name, filename), cropped_image)
                print(filename, " written!")
            except Exception as e:
                print("Some Exception -> ", e)
