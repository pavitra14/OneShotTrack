# Import the prerequisite libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dense
from keras.models import Model, model_from_json, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
import cv2
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import pickle
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from PIL import Image
from PIL import ImageTk
import imutils
import time
# from utils import LRN2D
# import utils
from colorama import Fore, Back, Style

def load_seperate():
    # load json and create model
    json_file = open('NN/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("NN/model.h5")
    print("Loaded model from disk")
    return loaded_model

def image_to_embedding(image, model):
    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA) 
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    # model.save("NN/model_all.h5")
    return embedding

def recognize_face(face_image, input_embeddings, model):

    embedding = image_to_embedding(face_image, model)
    
    minimum_distance = 200
    name = None
    
    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():
        
       
        euclidean_distance = np.linalg.norm(embedding-input_embedding)
        

        # print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))

        
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
    
    if minimum_distance < 0.68:
        return str(name)
    else:
        return None

import glob

def create_input_image_embeddings(model):
    input_embeddings = {}
    print("Creating Image Embeddings...")
    for file in glob.glob("images/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        c = person_name.split("_")[1]
        person_name = person_name.split("_")[0]
        print("Processing image for {} - {}".format(person_name, c))
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(image_file, model)
    print("Serializing embeddings...")
    with open("NN/embeddings.pickle", "wb") as f:
        f.write(pickle.dumps(input_embeddings))
    return input_embeddings

def recognize_faces_in_cam(model, t = None):
    input_embeddings = pickle.loads(open("NN/embeddings.pickle", "rb").read())
    cv2.namedWindow("OneShot")
    cv2.resizeWindow("OneShot", 100, 100)
    vc = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    time.sleep(2.0)
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        height, width, channels = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Loop through all the faces detected 
        identities = []
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h

            face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]    
            identity = recognize_face(face_image, input_embeddings, model)
            
            if identity is not None:
                print(Fore.RED + "IDENTITY DETECTED: " + Fore.GREEN + "{}".format(str(identity)))
                print(Style.RESET_ALL)
                img = cv2.rectangle(frame,(x1, y1),(x2, y2),(0,255,0),2)
                cv2.putText(img, str(identity), (x1+5,y1-5), font, 1, (0,0,255), 2)            
                if t is not None:
                    t.AddToList(str(identity))

        key = cv2.waitKey(100)
        cv2.imshow("OneShot", img)

        if key == 27: # exit on ESC
            break
    # vc.release()
    t.reset()
    cv2.destroyAllWindows()


def build_data_set():
    enrollment_no  = input("Enter Student enrollment no: ")
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    while(True):
        ret, img = cam.read()
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)     
            count += 1
            # Save the captured image into the datasets folder
            filename = enrollment_no + "_" + str(count) + ".jpg"
            cv2.imwrite("images/"+ filename, img[y1:y2,x1:x2])
            print("Capturing face: " + filename)
            cv2.imshow('image', img)
            
        k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break
    cam.release()
    cv2.destroyAllWindows()