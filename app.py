import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
icon = Image.open("icon.jpeg")
st.set_page_config(page_title="Leaf Disease Detection",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded"
                   #menu_items={'About': """# This OCR app is created by Gokulraj Pandiyarajan*!"""}
                   )
st.markdown("<h3 style='text-align: center; color: white; '><i>Leaf Disease Detection</i></h3>", unsafe_allow_html=True)


def setting_background():
    st.markdown(f""" 
    <style>
        .stApp {{
            background: linear-gradient(to right, #92CED3, #544e4d);
            background-size: cover;
            transition: background 0.5s ease;
        }}
        h1,h2,h3,h4,h5,h6 {{
            color:FFD0D0;
            font-family: 'Roboto', sans-serif;
        }}
        .stButton>button {{
            color: #4e4376;
            background-color: #f3f3f3;
            transition: all 0.3s ease-in-out;
        }}
        .stButton>button:hover {{
            color: #f3f3f3;
            background-color: #2b5876;
        }}
        .stTextInput>div>div>input {{
            color: #4e4376;
            background-color: #f3f3f3;
        }}
    </style>
    """,unsafe_allow_html=True) 
setting_background()


selected = option_menu(None, ["About","Upload & Detection"], 
                       icons=["home","cloud-upload-alt"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "25px", "text-align": "centre", "margin": "0px", "--hover-color": "#AB63FA", "transition": "color 0.3s ease, background-color 0.3s ease"},
                               "icon": {"font-size": "18px"},
                               "container" : {"max-width": "6000px", "padding": "10px", "border-radius": "5px"},
                               "nav-link-selected": {"background-color": "#AB63FA", "color": "white","font-style": "italic"}})

if selected=="About":
    col1, col2=st.columns(2)
    with col1:
        st.image("./bg1.jpg")
    with col2:
        st.markdown("<h4 style='text-align: center; color: white; '><i> About Leaf Disease Detector</i></h4>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:black;'> <i> Welcome to Leaf Disease Detector! Our mission is to empower farmers, gardeners, and plant enthusiasts to protect their plants from diseases. By using advanced image recognition technology, our app provides quick and accurate diagnoses.</i></h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:black;'> <i> How does it work? Simply upload a clear image of the affected leaf, and our AI-powered model analyzes it to identify potential diseases. We've trained our model on a vast dataset of plant leaf images to ensure reliable results.</i></h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:black;'><i> Accuracy matters. While our model is highly accurate, it's important to note that it's a tool to assist, not replace, expert diagnosis. We recommend consulting with a plant pathologist for serious concerns.</i></h5>",unsafe_allow_html=True)
        


if selected=="Upload & Detection":
    st.markdown("### Upload a Leaf Image")
    uploaded_image = st.file_uploader("upload here",label_visibility="collapsed",type=["png","jpeg","jpg"])

    if uploaded_image is not None:
        
        def save_image(uploaded_image):
            with open(os.path.join("uploaded_images",uploaded_image.name), "wb") as f:
                f.write(uploaded_image.getbuffer())   
        save_image(uploaded_image)

        st.text(uploaded_image.name)
        current_directory=os.getcwd()
        img_path=os.path.join(current_directory,"uploaded_images",uploaded_image.name)
        st.text(img_path)
        st.image(uploaded_image)
        if st.button("Predict"):
            model=load_model("exception.h5")
            class_names=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
                         'Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
                         'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
                         'Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight',
                         'Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot',
                         'Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
            img = image.load_img(img_path, target_size=(256, 256))  # Adjust target_size as per your model's input size
            # Preprocess the image
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image if required by your model
            predictions = model.predict(img_array)
            # Interpret the output
            predicted_class = np.argmax(predictions, axis=1)
            st.write(f'Predicted class: {predicted_class}')
            result=class_names[predicted_class[0]]
            st.markdown(f"<h3 style='text-align: center; color: white; '><i>{result}</i></h3>", unsafe_allow_html=True)
            
    
