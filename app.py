import streamlit as st 
import tensorflow as tf 


st.set_page_config(page_title="MNIST Prediction App", layout="centered")

st.title("Application de prédiction MNIST")

st.caption("Cette application permet de prédire les chiffres manuscrits en utilisant un modèle pré-entraîné sur le dataset MNIST.")

# chargement du modèle pré-entraîné 

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("./model/mnist_model.h5")
    return model 

with st.spinner("Chargement du modèle..."): 
    model = load_model()
    st.success("Modèle chargé avec succès!") 

from PIL import Image  # conda install anaconda::pillow
import numpy as np 

st.header("📥 Charger une image")

uploaded = st.file_uploader("Choisissez une image 28×28 (ou plus grande)", type=["png", "jpg", "jpeg"])


def preprocess_image(img):
    
    img = img.convert("L").resize((28, 28))
    
    arr = np.array(img, dtype="float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1) 
    return arr, img 

if uploaded: 
    img = Image.open(uploaded)
    st.success("Image téléversée avec succès!")
    
    if st.button("Prétraiter l'image"):
        arr, processed_img = preprocess_image(img)
        st.image(processed_img, caption="Image prétraitée (28x28 en niveaux de gris)", width=150)