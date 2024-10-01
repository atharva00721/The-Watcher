# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image


# @st.cache_resource
# def get_predictor_model():
#     from model import Model
#     model = Model()
#     return model


# header = st.container()
# model = get_predictor_model()

# # Header
# st.title("Street Incident Classifier")
# st.markdown("""
#     Welcome to the **Street Incident Classifier**! 🎥  
#     This app helps you identify if there is a **fight**, **fire**, **car crash**, or if **everything is okay** based on the image you upload.
# """)

# # File uploader section
# st.sidebar.markdown(
#     """
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@900&display=swap');
#         .logo {
#             font-family: 'Poppins', sans-serif;
#             font-weight: 900; 
#             font-size: 3rem; /* Increased font size for better visibility */
#             line-height: 3.5rem; /* Adjust line height accordingly */
#             text-transform: uppercase;
#             color: #c4cbf5;
#         }

#         .logo::first-letter {
#             color: #1959ad; /* First letter color */
#         }
#     </style>
#     <a href='https://www.youtube.com' style='text-decoration: none; color: inherit;'>
#         <h1 class='logo'>AETHER</h1>
#     </a>
#     """, 
#     unsafe_allow_html=True
# )

# st.sidebar.header("Upload Your Image")
# uploaded_file = st.sidebar.file_uploader("Choose an image file...")

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.subheader("Uploaded Image")
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption="Original Image", use_column_width=True)

#     # Convert image to numpy array for processing
#     image_array = np.array(image)

#     # Make prediction
#     st.subheader("Prediction")
#     with st.spinner('Analyzing the image...'):
#         label_text = model.predict(image=image_array)['label'].title()
#         st.success(f'Predicted label: **{label_text}**')

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.write("App created with [Streamlit](https://streamlit.io/).")

### VIDEO

# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image
# import tempfile
# import os


# @st.cache_resource
# def get_predictor_model():
#     from model import Model
#     model = Model()
#     return model


# header = st.container()
# model = get_predictor_model()

# # Header
# st.title("Street Incident Classifier")
# st.markdown("""
#     Welcome to the **Street Incident Classifier**! 🎥  
#     This app helps you identify if there is a **fight**, **fire**, **car crash**, or if **everything is okay** based on the image or video you upload.
# """)

# # File uploader section
# st.sidebar.header("Upload Your File")
# uploaded_file = st.sidebar.file_uploader("Choose an image or video file...", type=["jpg", "jpeg", "png", "mp4"])

# if uploaded_file is not None:
#     # Check if the uploaded file is a video
#     if uploaded_file.name.endswith('.mp4'):
#         # Save the uploaded video to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_file_path = temp_file.name

#         # Display the uploaded video
#         st.subheader("Uploaded Video")
#         st.video(temp_file_path)

#         # Process the video
#         st.subheader("Processing Video...")
#         video_stream = cv2.VideoCapture(temp_file_path)
#         frame_predictions = []
        
#         while True:
#             ret, frame = video_stream.read()
#             if not ret:
#                 break

#             # Convert the frame to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Make prediction
#             label_text = model.predict(image=frame_rgb)['label'].title()
#             frame_predictions.append(label_text)

#         # Release the video capture
#         video_stream.release()

#         # Delete the temporary file
#         os.remove(temp_file_path)

#         # Display predictions
#         st.success("Predictions for each frame:")
#         for i, prediction in enumerate(frame_predictions):
#             st.write(f"Frame {i + 1}: **{prediction}**")

#     else:
#         # Display the uploaded image
#         st.subheader("Uploaded Image")
#         image = Image.open(uploaded_file).convert('RGB')
#         st.image(image, caption="Original Image", use_column_width=True)

#         # Convert image to numpy array for processing
#         image_array = np.array(image)

#         # Make prediction
#         st.subheader("Prediction")
#         with st.spinner('Analyzing the image...'):
#             label_text = model.predict(image=image_array)['label'].title()
#             st.success(f'Predicted label: **{label_text}**')

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown(
#     """
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@900&display=swap');
#         .logo {
#             font-family: 'Poppins', sans-serif;
#             font-weight: 900; 
#             font-size: 3rem; /* Increased font size for better visibility */
#             line-height: 3.5rem; /* Adjust line height accordingly */
#             text-transform: uppercase;
#             color: #c4cbf5;
#         }

#         .logo::first-letter {
#             color: #1959ad; /* First letter color */
#         }
#     </style>
#     <a href='https://www.youtube.com' style='text-decoration: none; color: inherit;'>
#         <h1 class='logo'>AETHER</h1>
#     </a>
#     """, 
#     unsafe_allow_html=True
# )

import cv2
import numpy as np
import streamlit as st
from PIL import Image


@st.cache_resource
def get_predictor_model():
    from model import Model
    model = Model()
    return model


header = st.container()
model = get_predictor_model()

# Header
st.title("Street Incident Classifier")
st.markdown("""Welcome to the **Street Incident Classifier**! 🎥  
    This app helps you identify if there is a **fight**, **fire**, **car crash**, or if **everything is okay** based on the image you upload or capture from your camera.
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

# Camera input section
st.sidebar.header("Capture Image from Camera")
camera_input = st.camera_input("Capture an image...")

if camera_input is not None:
    # Display the captured image in mirror effect
    st.subheader("Captured Image (Mirror Effect)")
    image = Image.open(camera_input).convert('RGB')
    
    # Flip the image horizontally for mirror effect
    mirror_image = np.array(image)[:, ::-1]
    
    # Convert back to PIL Image for display
    mirror_image = Image.fromarray(mirror_image)
    st.image(mirror_image, caption="Captured Image (Mirror)", use_column_width=True)

    # Convert mirror image to numpy array for processing
    image_array = np.array(mirror_image)

    # Make prediction
    st.subheader("Prediction")
    with st.spinner('Analyzing the image...'):
        label_text = model.predict(image=image_array)['label'].title()
        st.success(f'Predicted label: **{label_text}**')

# Footer
st.sidebar.markdown("---")
st.sidebar.write("App created with [Streamlit](https://streamlit.io/).")
