import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from io import StringIO 
from tqdm import tqdm
from PIL import Image
import io
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array, load_img
from keras.utils.vis_utils import plot_model
from glob import glob
# loading in the model to predict on the data

pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def welcome():
    return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
def predict_func(img): 
    img = cv2.resize(np.float32(img), (224, 224))
    img = np.reshape(np.float32(img), [-1,224, 224,3])
    result = np.argmax(model.predict(img))
    if result == 0: 
        st.success("Ths Image is Recyclable")
      
    elif result ==1: 
        st.success("This Image is Organic")

    return result
      
# this is the main function in which we define our webpage
def main():
    #st.title("Waste Classification")

    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Waste Classification  </h1>
    </div>
    """
        # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    image = load_image()
    if st.button("Predict"):
        res=predict_func(image)
        
    
       




if __name__=='__main__':
    main()
