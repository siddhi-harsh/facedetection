import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
st.set_option('deprecation.showfileUploaderEncoding', False)
# Loading saved model from Drive.
from keras.models import load_model
model = load_model('FDPCNN1.h5')

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Upgrade</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Session on CNN </p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title(""" Face Mask Detection""")
file= st.file_uploader("Please upload image", type=("jpeg", "png","jpg"))

import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(64, 64)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  result = model.predict(img_reshape)
  print(result)
  #training_set.class_indices
  if result[0][0] == 1:
    prediction = "Person without Mask" 
    print(prediction)
  else:
    prediction = 'Person with Mask'
    print(prediction)#x = np.expand_dims(x, axis=1)
  
  
  return prediction
if file is None:
  st.text("Please upload an Image of person ")
else:
  image=Image.open(file)
  #image=np.array(image)
  #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  #image = cv2.imdecode(file_bytes, 1)
  st.image(image,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Predict whether Person is with Mask or without Mask"):
  result=import_and_predict(image)
  st.success('Model has predicted the image  is  of  {}'.format(result))
if st.button("About"):
  st.header(" siddhi")
  st.subheader("learner")
  
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Model Developed using CNN</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
