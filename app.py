from prepare import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from gui import SharedCell
# Global Variables
print("Loading model from file")
model = load_model("NN/model_all.h5", custom_objects={'tf':tf})
print("Model Loaded from file")
def menu():
    print("[1] Register a Student")
    print("[2] Extract Embeddings")
    print("[3] Start Recognizer")
    print("[0] Exit")
    choice  = int(input("Enter your choice[1-3]: "))
    if choice == 1:
        build_data_set()
        menu()
    elif choice == 2:
        create_input_image_embeddings(model)
        menu()
    elif choice == 3:
        cell = SharedCell()
        print("Shared Cell created, please start gui.py")
        recognize_faces_in_cam(model, cell)
        menu()
    else:
        exit()

if __name__ == '__main__':
    menu()