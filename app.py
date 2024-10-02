import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
from base64 import b64encode

@st.cache_resource
def get_predictor_model():
    from model import Model
    model = Model()
    return model

# Function to convert PIL Image to Base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode()
    return img_str

header = st.container()
model = get_predictor_model()

# Header
st.title("Street Incident Classifier")
st.markdown("""
    Welcome to the **Street Incident Classifier**! 🎥  
    This app helps you identify if there is a **fight**, **fire**, **car crash**, or if **everything is okay** based on the image you upload.
""")

# File uploader section
st.sidebar.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@900&display=swap');
        .logo {
            font-family: 'Poppins', sans-serif;
            font-weight: 900; 
            font-size: 3rem; /* Increased font size for better visibility */
            line-height: 3.5rem; /* Adjust line height accordingly */
            text-transform: uppercase;
            color: #c4cbf5;
        }

        .logo::first-letter {
            color: #1959ad; /* First letter color */
        }
    </style>
    <a href='https://www.youtube.com' style='text-decoration: none; color: inherit;'>
        <h1 class='logo'>AETHER</h1>
    </a>
    """, 
    unsafe_allow_html=True
)

st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file...")

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file).convert('RGB')

    # Convert image to numpy array for processing
    image_array = np.array(image)

    # Make prediction
    st.subheader("Prediction")
    with st.spinner('Analyzing the image...'):
        label_text = model.predict(image=image_array)['label'].title()
        
        # Display the prediction with enhanced styling
        styled_success_message = f"""
            <div style="font-size: 1.5rem; font-weight: bold; color: #f4f4f4; background-color: #c3504b; padding: 10px; border-radius: 5px; margin-bottom:5px;">
                Predicted label: <strong>{label_text}</strong>
            </div>
        """

        st.markdown(styled_success_message, unsafe_allow_html=True)
    
    # Display the image with rounded corners
    image_html = f"""
        <div style="border-radius: 10px; overflow: hidden; border: 2px solid #c3504b; margin-bottom: 15px;">
            <img src="data:image/png;base64,{image_to_base64(image)}" style="width: 100%; border-radius: 10px;"/>
        </div>
    """
    st.markdown(image_html, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("App created with [Streamlit](https://streamlit.io/).")
