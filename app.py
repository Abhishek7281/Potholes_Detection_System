import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import os
import tempfile

def load_model():
    # Load the saved model using joblib
    model_data = joblib.load("pothole_detection_model1.pkl")

    # Load the network weights and configuration separately
    net = cv2.dnn.readNet(model_data["weights"], model_data["config"])
    classes = model_data["classes"]
    conf_threshold = model_data["confThreshold"]
    nms_threshold = model_data["nmsThreshold"]

    # Create and return the detection model
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    return model, classes, conf_threshold, nms_threshold

def detect_potholes(img, model, classes, conf_threshold, nms_threshold):
    class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

    for (class_id, score, box) in zip(class_ids, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)

    return img

def main():
    # Load the model
    model, classes, conf_threshold, nms_threshold = load_model()

    # Streamlit title without custom styling
    st.title("Potholes Detection System")

    # Navigation
    nav_option = st.sidebar.selectbox("Navigation", ["Home", "About", "Services", "Contact"])

    if nav_option == "Home":
        st.markdown("<h2 style='color: #00695c;'>Welcome to the Home Page</h2>", unsafe_allow_html=True)
    elif nav_option == "About":
        st.markdown("<h2 style='color: #00695c;'>About Us</h2>", unsafe_allow_html=True)
        st.write("Welcome to our Pothole Detection System! We are dedicated to making roads safer by detecting and addressing potholes efficiently.")
    elif nav_option == "Services":
        st.markdown("<h2 style='color: #00695c;'>Our Services</h2>", unsafe_allow_html=True)
        st.markdown("- Pothole Detection and Reporting")
        st.markdown("- Pothole Repair and Maintenance")
    elif nav_option == "Contact":
        st.markdown("<h2 style='color: #00695c;'>Contact Us</h2>", unsafe_allow_html=True)
        st.write("Feel free to reach out to us for any inquiries or assistance.")
        st.write("Email: info@potholedetection.com")
        st.write("Phone: +1 (123) 456-7890")

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
    if uploaded_file is not None:
        is_video = uploaded_file.type.startswith('video/')

        if is_video:
            # Save the video file temporarily
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

            # Open the video file
            video = cv2.VideoCapture(temp_video_path)

            while True:
                ret, frame = video.read()

                if not ret:
                    break

                detected_frame = detect_potholes(frame, model, classes, conf_threshold, nms_threshold)

                st.image([frame, detected_frame], caption=['Original Frame', 'Detected Potholes'], width=300)

            video.release()

            # Remove the temporary file
            os.remove(temp_video_path)
        else:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            if st.button("Detect Potholes", key="detect_button"):
                detected_img = detect_potholes(img_array, model, classes, conf_threshold, nms_threshold)
                st.image([image, detected_img], caption=['Original Image', 'Detected Potholes'], width=300)



    # Social Media Section (Placed after the main content)
    st.markdown("""
        <style>
            .social-media {
                display: flex;
                justify-content: space-between;
                margin-top: 450px;
            }
            .social-media a img {
                width: 30px;
                height: 30px;
                margin-right: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="social-media">
            <a href="https://www.linkedin.com/" target="_blank"><img src="https://www.clipartmax.com/png/middle/240-2405475_facebook-twitter-google-instagram-linkedin-linkedin-logo-png-download.png" alt="LinkedIn">LinkedIn</a>
            <a href="https://www.facebook.com/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6c/Facebook_Logo_2023.png" alt="Facebook">Facebook</a>
            <a href="https://www.instagram.com/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/2048px-Instagram_logo_2016.svg.png" alt="Instagram">Instagram</a>
            <a href="https://twitter.com/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Logo_of_Twitter.svg/512px-Logo_of_Twitter.svg.png" alt="Twitter">Twitter</a>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
