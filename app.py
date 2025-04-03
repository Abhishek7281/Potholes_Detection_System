# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import joblib
# import os
# import tempfile

# def load_model():
#     # Load the saved model using joblib
#     # model_data = joblib.load("pothole_detection_model1.pkl")

#     # Load the network weights and configuration separately
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     classes ="classes"
#     conf_threshold = 0.25
#     # 0.6
#     nms_threshold = 0.15
#     # 0.4

#     # Create and return the detection model
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

#     return model, classes, conf_threshold, nms_threshold

# def detect_potholes(img, model, classes, conf_threshold, nms_threshold):
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

#     for (class_id, score, box) in zip(class_ids, scores, boxes):
#         cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)

#     return img

# def main():
#     # Load the model
#     model, classes, conf_threshold, nms_threshold = load_model()

#     # Streamlit title without custom styling
#     st.title("Potholes Detection System")

#     # Navigation
#     nav_option = st.sidebar.selectbox("Navigation", ["Home", "About", "Services", "Contact"])

#     if nav_option == "Home":
#         st.markdown("<h2 style='color: #00695c;'>Welcome to the Home Page</h2>", unsafe_allow_html=True)
#     elif nav_option == "About":
#         st.markdown("<h2 style='color: #00695c;'>About Us</h2>", unsafe_allow_html=True)
#         st.write("Welcome to our Pothole Detection System! We are dedicated to making roads safer by detecting and addressing potholes efficiently.")
#     elif nav_option == "Services":
#         st.markdown("<h2 style='color: #00695c;'>Our Services</h2>", unsafe_allow_html=True)
#         st.markdown("- Pothole Detection and Reporting")
#         st.markdown("- Pothole Repair and Maintenance")
#     elif nav_option == "Contact":
#         st.markdown("<h2 style='color: #00695c;'>Contact Us</h2>", unsafe_allow_html=True)
#         st.write("Feel free to reach out to us for any inquiries or assistance.")
#         st.write("Email: info@potholedetection.com")
#         st.write("Phone: +1 (123) 456-7890")

#     uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith('video/')

#         if is_video:
#             # Save the video file temporarily
#             temp_video = tempfile.NamedTemporaryFile(delete=False)
#             temp_video.write(uploaded_file.read())
#             temp_video_path = temp_video.name

#             # Open the video file
#             video = cv2.VideoCapture(temp_video_path)

#             while True:
#                 ret, frame = video.read()

#                 if not ret:
#                     break

#                 detected_frame = detect_potholes(frame, model, classes, conf_threshold, nms_threshold)

#                 st.image([frame, detected_frame], caption=['Original Frame', 'Detected Potholes'], width=300)

#             video.release()

#             # Remove the temporary file
#             os.remove(temp_video_path)
#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)

#             if st.button("Detect Potholes", key="detect_button"):
#                 detected_img = detect_potholes(img_array, model, classes, conf_threshold, nms_threshold)
#                 st.image([image, detected_img], caption=['Original Image', 'Detected Potholes'], width=300)



#     # Social Media Section (Placed after the main content)
#     st.markdown("""
#         <style>
#             .social-media {
#                 display: flex;
#                 justify-content: space-between;
#                 margin-top: 450px;
#             }
#             .social-media a img {
#                 width: 30px;
#                 height: 30px;
#                 margin-right: 10px;
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     st.markdown(
#         """
#         <div class="social-media">
#             <a href="https://www.linkedin.com/" target="_blank"><img src="https://www.clipartmax.com/png/middle/240-2405475_facebook-twitter-google-instagram-linkedin-linkedin-logo-png-download.png" alt="LinkedIn">LinkedIn</a>
#             <a href="https://www.facebook.com/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6c/Facebook_Logo_2023.png" alt="Facebook">Facebook</a>
#             <a href="https://www.instagram.com/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/2048px-Instagram_logo_2016.svg.png" alt="Instagram">Instagram</a>
#             <a href="https://twitter.com/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Logo_of_Twitter.svg/512px-Logo_of_Twitter.svg.png" alt="Twitter">Twitter</a>
#         </div>
#         """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import joblib
# import os
# import tempfile
# import shutil

# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     classes = "classes"
#     conf_threshold = 0.25
#     nms_threshold = 0.15

#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

#     return model, classes, conf_threshold, nms_threshold

# def detect_potholes(img, model, classes, conf_threshold, nms_threshold):
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
#     for (class_id, score, box) in zip(class_ids, scores, boxes):
#         cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
#     return img

# def main():
#     model, classes, conf_threshold, nms_threshold = load_model()
#     st.title("Pothole Detection System")

#     uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith('video/')
#         if is_video:
#             temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#             temp_video.write(uploaded_file.read())
#             temp_video_path = temp_video.name
#             video = cv2.VideoCapture(temp_video_path)
#             output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 detected_frame = detect_potholes(frame, model, classes, conf_threshold, nms_threshold)
#                 out.write(detected_frame)

#             video.release()
#             out.release()
#             st.video(output_video_path)
#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")
#             os.remove(temp_video_path)
#             os.remove(output_video_path)
#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model, classes, conf_threshold, nms_threshold)
#             detected_pil = Image.fromarray(detected_img)
#             st.image([image, detected_pil], caption=['Original Image', 'Detected Potholes'], width=300)
#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)
#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")
#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os

# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
#     for (class_id, score, box) in zip(class_ids, scores, boxes):
#         cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
#     return img

# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("\U0001F6E3️ Pothole Detection System")
#     st.sidebar.header("Navigation")
#     nav_option = st.sidebar.radio("Go to", ["Home", "Upload Image/Video", "About"])
    
#     model, conf_threshold, nms_threshold = load_model()
    
#     if nav_option == "Home":
#         st.markdown("""<h2 style='text-align: left;'>Welcome to the Pothole Detection System! 🛣️</h2>""", unsafe_allow_html=True)
#         st.image("https://upload.wikimedia.org/wikipedia/commons/3/35/Pothole.jpg", width=700)
#         st.write("This tool helps in detecting potholes in images and videos using Deep Learning.")
    
#     elif nav_option == "Upload Image/Video":
#         uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
#         if uploaded_file is not None:
#             is_video = uploaded_file.type.startswith('video/')
#             if is_video:
#                 temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#                 temp_video.write(uploaded_file.read())
#                 temp_video_path = temp_video.name
                
#                 video = cv2.VideoCapture(temp_video_path)
#                 output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 fps = int(video.get(cv2.CAP_PROP_FPS))
#                 frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                
#                 progress_bar = st.progress(0)
#                 total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frame_count = 0
                
#                 while True:
#                     ret, frame = video.read()
#                     if not ret:
#                         break
#                     detected_frame = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                     out.write(detected_frame)
#                     frame_count += 1
#                     progress_bar.progress(min(frame_count / total_frames, 1.0))
                
#                 video.release()
#                 out.release()
#                 st.video(output_video_path)
#                 with open(output_video_path, "rb") as file:
#                     st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")
#                 os.remove(temp_video_path)
#                 os.remove(output_video_path)
                
#             else:
#                 image = Image.open(uploaded_file)
#                 img_array = np.array(image)
#                 detected_img = detect_potholes(img_array, model, conf_threshold, nms_threshold)
#                 detected_pil = Image.fromarray(detected_img)
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.image(image, caption='Original Image', use_column_width=True)
#                 with col2:
#                     st.image(detected_pil, caption='Detected Potholes', use_column_width=True)
                
#                 temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#                 detected_pil.save(temp_image_path)
#                 with open(temp_image_path, "rb") as file:
#                     st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")
#                 os.remove(temp_image_path)
    
#     elif nav_option == "About":
#         st.markdown("""<h2 style='text-align: left;'>About This Project</h2>""", unsafe_allow_html=True)
#         st.write("This project uses deep learning to detect potholes in images and videos. It aims to assist road maintenance authorities in identifying and addressing road damages efficiently.")
#         st.write("🔹 Uses YOLO for object detection\n🔹 Processes images and videos in real-time\n🔹 Provides downloadable results")

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os

# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
#     for (class_id, score, box) in zip(class_ids, scores, boxes):
#         cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
#     return img

# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")

#     model, conf_threshold, nms_threshold = load_model()

#     uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
    
#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith('video/')
        
#         if is_video:
#             temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#             temp_video.write(uploaded_file.read())
#             temp_video_path = temp_video.name
            
#             video = cv2.VideoCapture(temp_video_path)
#             output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0
            
#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 detected_frame = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                 out.write(detected_frame)
#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))
            
#             video.release()
#             out.release()
#             st.video(output_video_path)
            
#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")
            
#             os.remove(temp_video_path)
#             os.remove(output_video_path)
        
#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model, conf_threshold, nms_threshold)
#             detected_pil = Image.fromarray(detected_img)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption='Original Image', use_container_width=True)
#             with col2:
#                 st.image(detected_pil, caption='Detected Potholes', use_container_width=True)
            
#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)
            
#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")
            
#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os

# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
#     for (class_id, score, box) in zip(class_ids, scores, boxes):
#         cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
#     return img

# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")

#     model, conf_threshold, nms_threshold = load_model()

#     uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
    
#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith('video/')
        
#         if is_video:
#             temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#             temp_video.write(uploaded_file.read())
#             temp_video_path = temp_video.name
            
#             video = cv2.VideoCapture(temp_video_path)
#             output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0
            
#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 detected_frame = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                 out.write(detected_frame)
#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))
            
#             video.release()
#             out.release()
#             st.video(output_video_path)
            
#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")
            
#             os.remove(temp_video_path)
#             os.remove(output_video_path)
        
#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model, conf_threshold, nms_threshold)
#             detected_pil = Image.fromarray(detected_img)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption='Original Image', width=625)  # Set width
#             with col2:
#                 st.image(detected_pil, caption='Detected Potholes', width=625)  # Set width
            
#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)
            
#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")
            
#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os

# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
#     for (class_id, score, box) in zip(class_ids, scores, boxes):
#         cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
#     return img

# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")

#     model, conf_threshold, nms_threshold = load_model()

#     uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
    
#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith('video/')
        
#         if is_video:
#             temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#             temp_video.write(uploaded_file.read())
#             temp_video_path = temp_video.name
            
#             video = cv2.VideoCapture(temp_video_path)
#             output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0
            
#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 detected_frame = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                 out.write(detected_frame)
#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))
            
#             video.release()
#             out.release()

#             st.success("✅ Video processing complete!")
            
#             # **Now Display the Video**
#             st.video(output_video_path)  # Display processed video
            
#             # **Allow Download**
#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")
            
#             os.remove(temp_video_path)
#             os.remove(output_video_path)
        
#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model, conf_threshold, nms_threshold)
#             detected_pil = Image.fromarray(detected_img)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption='Original Image', width=625)  # Set width
#             with col2:
#                 st.image(detected_pil, caption='Detected Potholes', width=625)  # Set width
            
#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)
            
#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")
            
#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os

# # ✅ Load YOLO Model
# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# # ✅ Pothole Detection Function (Now Includes Confidence Score)
# # def detect_potholes(img, model, conf_threshold, nms_threshold):
# #     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

# #     for (class_id, score, box) in zip(class_ids, scores, boxes):
# #         x, y, w, h = box
# #         confidence = float(score)  # Convert score to float

# #         # ✅ Bounding Box Color: Green
# #         bbox_color = (0, 255, 0)  
# #         thickness = 3  # Make bounding box thicker

# #         # ✅ Draw Bounding Box
# #         cv2.rectangle(img, (x, y), (x + w, y + h), bbox_color, thickness)

# #         # ✅ Display Confidence Score
# #         label = f"{confidence:.2f}"  # Format to 2 decimal places
# #         font_scale = 1
# #         font_thickness = 2
# #         text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

# #         # ✅ Create Background for Text
# #         text_x, text_y = x, y - 10
# #         cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 10, text_y + 5), bbox_color, -1)
        
# #         # ✅ Put Text on Image
# #         cv2.putText(img, label, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)  # Black text
    
# #     return img

# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     """
#     Detect potholes and display confidence scores with green bounding boxes.
#     """
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

#     for (class_id, score, box) in zip(class_ids.flatten(), scores.flatten(), boxes):
#         x, y, w, h = box.astype(int)
#         confidence = float(score)  # Ensure confidence is a float

#         # ✅ Bounding Box Color: Green
#         bbox_color = (0, 255, 0)  
#         thickness = 3  # Make bounding box thicker

#         # ✅ Draw Bounding Box
#         cv2.rectangle(img, (x, y), (x + w, y + h), bbox_color, thickness)

#         # ✅ Display Confidence Score
#         label = f"{confidence:.2f}"  # Format confidence to 2 decimal places
#         font_scale = 1
#         font_thickness = 2
#         text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

#         # ✅ Create Background for Text
#         text_x, text_y = x, max(y - 10, 20)  # Ensure text is within bounds
#         cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), 
#                       (text_x + text_size[0] + 10, text_y + 5), bbox_color, -1)

#         # ✅ Put Text on Image
#         cv2.putText(img, label, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
#                     font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)  # Black text
    
#     return img


# # ✅ Streamlit UI
# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")

#     model, conf_threshold, nms_threshold = load_model()

#     uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4"])
    
#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith('video/')

#         if is_video:
#             temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#             temp_video.write(uploaded_file.read())
#             temp_video_path = temp_video.name

#             video = cv2.VideoCapture(temp_video_path)
#             output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0

#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 detected_frame = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                 out.write(detected_frame)
#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))

#             video.release()
#             out.release()

#             st.success("✅ Video processing complete!")
            
#             # **Now Display the Video**
#             st.video(output_video_path)  # Display processed video
            
#             # **Allow Download**
#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")
            
#             os.remove(temp_video_path)
#             os.remove(output_video_path)

#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model, conf_threshold, nms_threshold)
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption='Original Image', width=625)  # Set width
#             with col2:
#                 st.image(detected_pil, caption='Detected Potholes', width=625)  # Set width

#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)

#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

#original
# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile

# # ✅ Increase Upload Limit to 1GB
# os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"  # 1GB limit

# # ✅ Load YOLO Model
# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# # ✅ Pothole Detection Function
# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     class_ids, scores, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

#     for (class_id, score, box) in zip(class_ids.flatten(), scores.flatten(), boxes):
#         x, y, w, h = box.astype(int)
#         confidence = float(score)

#         bbox_color = (0, 255, 0)  
#         thickness = 3

#         cv2.rectangle(img, (x, y), (x + w, y + h), bbox_color, thickness)
        
#         label = f"{confidence:.2f}"
#         font_scale = 1
#         font_thickness = 2
#         text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

#         text_x, text_y = x, max(y - 10, 20)
#         cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), 
#                       (text_x + text_size[0] + 10, text_y + 5), bbox_color, -1)
        
#         cv2.putText(img, label, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
#                     font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
#     return img

# # ✅ Streamlit UI
# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")

#     model, conf_threshold, nms_threshold = load_model()

#     # ✅ File Uploader With Increased File Size Handling
#     uploaded_file = st.file_uploader("Choose an image or video (Up to 1GB)...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         temp_dir = tempfile.mkdtemp()  # ✅ Use Temp Directory
#         file_path = os.path.join(temp_dir, uploaded_file.name)

#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())  # Save large files to disk instead of RAM

#         st.success(f"✅ File uploaded: {uploaded_file.name} (Size: {round(len(uploaded_file.getvalue()) / (1024*1024), 2)} MB)")

#         is_video = uploaded_file.type.startswith('video/')

#         if is_video:
#             video = cv2.VideoCapture(file_path)
#             output_video_path = os.path.join(temp_dir, "processed_video.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0

#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 detected_frame = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                 out.write(detected_frame)
#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))

#             video.release()
#             out.release()

#             st.success("✅ Video processing complete!")
#             st.video(output_video_path)

#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#         else:
#             image = Image.open(file_path)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model, conf_threshold, nms_threshold)
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption='Original Image', width=625)
#             with col2:
#                 st.image(detected_pil, caption='Detected Potholes', width=625)

#             output_image_path = os.path.join(temp_dir, "processed_image.png")
#             detected_pil.save(output_image_path)

#             with open(output_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

# if __name__ == "__main__":
#     main()


#File having no potholes resolve
# Original 

# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile

# # ✅ Increase Upload Limit to 1GB
# os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"  # 1GB limit

# # ✅ Load YOLO Model
# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# # ✅ Pothole Detection Function with Fix
# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     """
#     Detect potholes and display confidence scores with green bounding boxes.
#     Ensures safety checks to prevent errors when no detections are found.
#     """
#     # ✅ Run detection
#     detections = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

#     # ✅ Check if the detections are valid before unpacking
#     if not detections or len(detections) != 3:
#         return img  # Return original image if no detections

#     class_ids, scores, boxes = detections

#     # ✅ Ensure class_ids, scores, and boxes are valid
#     if class_ids is None or scores is None or boxes is None:
#         return img
#     if len(class_ids) == 0 or len(scores) == 0 or len(boxes) == 0:
#         return img

#     # ✅ Process valid detections
#     for (class_id, score, box) in zip(class_ids.flatten(), scores.flatten(), boxes):
#         x, y, w, h = box.astype(int)
#         confidence = float(score)

#         bbox_color = (0, 255, 0)  
#         thickness = 3

#         cv2.rectangle(img, (x, y), (x + w, y + h), bbox_color, thickness)

#         label = f"{confidence:.2f}"
#         font_scale = 1
#         font_thickness = 2
#         text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

#         text_x, text_y = x, max(y - 10, 20)
#         cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), 
#                       (text_x + text_size[0] + 10, text_y + 5), bbox_color, -1)
        
#         cv2.putText(img, label, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
#                     font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

#     return img


# # ✅ Streamlit UI
# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")

#     model, conf_threshold, nms_threshold = load_model()

#     uploaded_file = st.file_uploader("Choose an image or video (Up to 1GB)...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         temp_dir = tempfile.mkdtemp()  # ✅ Use Temp Directory
#         file_path = os.path.join(temp_dir, uploaded_file.name)

#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())  # Save large files to disk instead of RAM

#         st.success(f"✅ File uploaded: {uploaded_file.name} (Size: {round(len(uploaded_file.getvalue()) / (1024*1024), 2)} MB)")

#         is_video = uploaded_file.type.startswith('video/')

#         if is_video:
#             video = cv2.VideoCapture(file_path)
#             output_video_path = os.path.join(temp_dir, "processed_video.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0

#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 detected_frame = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                 out.write(detected_frame)
#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))

#             video.release()
#             out.release()

#             st.success("✅ Video processing complete!")
#             st.video(output_video_path)

#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#         else:
#             image = Image.open(file_path)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model, conf_threshold, nms_threshold)
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption='Original Image', width=625)
#             with col2:
#                 st.image(detected_pil, caption='Detected Potholes', width=625)

#             output_image_path = os.path.join(temp_dir, "processed_image.png")
#             detected_pil.save(output_image_path)

#             with open(output_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

# if __name__ == "__main__":
#     main()



# Modifications & Additions
# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import zipfile
# import pandas as pd

# # ✅ Increase Upload Limit to 1GB
# os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"

# # ✅ Load YOLO Model
# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.15
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# # ✅ Pothole Detection Function
# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     detections = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
#     if not detections or len(detections) != 3:
#         return img, []
    
#     class_ids, scores, boxes = detections
#     detected_boxes = []
#     for (class_id, score, box) in zip(class_ids.flatten(), scores.flatten(), boxes):
#         x, y, w, h = box.astype(int)
#         confidence = float(score)
#         detected_boxes.append((x, y, x + w, y + h, confidence))
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         cv2.putText(img, f"{confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
#     return img, detected_boxes

# # ✅ Streamlit UI
# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")
#     model, conf_threshold, nms_threshold = load_model()
#     uploaded_file = st.file_uploader("Choose an image or video (Up to 1GB)...", type=["jpg", "png", "jpeg", "mp4"])
    
#     if uploaded_file is not None:
#         temp_dir = tempfile.mkdtemp()
#         file_path = os.path.join(temp_dir, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
        
#         is_video = uploaded_file.type.startswith('video/')
#         if is_video:
#             video = cv2.VideoCapture(file_path)
#             output_video_path = os.path.join(temp_dir, "detected_frames", "processed_video.mp4")
#             frames_dir = os.path.join(temp_dir, "detected_frames/frames")
#             os.makedirs(frames_dir, exist_ok=True)
            
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
#             detection_data = []
#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
                
#                 detected_frame, boxes = detect_potholes(frame, model, conf_threshold, nms_threshold)
#                 out.write(detected_frame)
#                 frame_filename = f"frame_{int(video.get(cv2.CAP_PROP_POS_FRAMES)):04d}.png"
#                 frame_path = os.path.join(frames_dir, frame_filename)
#                 cv2.imwrite(frame_path, detected_frame)
                
#                 for (x1, y1, x2, y2, confidence) in boxes:
#                     detection_data.append([frame_filename, x1, y1, x2, y2, confidence])
            
#             video.release()
#             out.release()
            
#             csv_path = os.path.join(temp_dir, "detected_frames", "pothole_coordinates.csv")
#             df = pd.DataFrame(detection_data, columns=["Frame", "X1", "Y1", "X2", "Y2", "Confidence"])
#             df.to_csv(csv_path, index=False)
            
#             zip_path = os.path.join(temp_dir, "detected_frames.zip")
#             with zipfile.ZipFile(zip_path, 'w') as zipf:
#                 for root, _, files in os.walk(os.path.join(temp_dir, "detected_frames")):
#                     for file in files:
#                         zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
            
#             st.video(output_video_path)
            
#             with open(zip_path, "rb") as file:
#                 st.download_button("Download Detected Frames & Coordinates", file, file_name="detected_frames.zip", mime="application/zip")

# if __name__ == "__main__":
#     main()





# With gps coordinates
import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import zipfile
import pandas as pd

# ✅ Increase Upload Limit to 1GB
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"

# ✅ Load YOLO Model
def load_model():
    net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
    conf_threshold = 0.25
    nms_threshold = 0.15
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    return model, conf_threshold, nms_threshold

# ✅ Pothole Detection Function
def detect_potholes(img, model, conf_threshold, nms_threshold):
    detections = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
    
    if not detections or len(detections) != 2:  # It should return exactly two values
        return img, []
    
    class_ids, boxes = detections  # YOLOv4 Tiny may not return scores separately
    detected_boxes = []
    
    for (class_id, box) in zip(class_ids, boxes):
        x, y, w, h = map(int, box)
        confidence = 0.5  # Default confidence if not returned
        
        detected_boxes.append((x, y, x + w, y + h, confidence))
        
        color = (255, 0, 0)  # Blue color for bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, f"{confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return img, detected_boxes

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="Pothole Detection", layout="wide")
    st.title("🛣️ Pothole Detection System")
    
    if "model" not in st.session_state:
        st.session_state.model, st.session_state.conf_threshold, st.session_state.nms_threshold = load_model()
    
    uploaded_video = st.file_uploader("Choose a video (Up to 1GB)...", type=["mp4"])
    uploaded_gps = st.file_uploader("Upload GPS Coordinates CSV (Mandatory)", type=["csv"])
    
    if uploaded_video is not None and uploaded_gps is not None:
        if st.button("Start Processing"):
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_video.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_video.read())
            
            gps_df = pd.read_csv(uploaded_gps)
            
            video = cv2.VideoCapture(file_path)
            output_video_path = os.path.join(temp_dir, "processed_video.mp4")
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(video.get(cv2.CAP_PROP_FPS))
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
            detection_data = []
            frame_index = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                detected_frame, boxes = detect_potholes(frame, st.session_state.model, st.session_state.conf_threshold, st.session_state.nms_threshold)
                
                if boxes:
                    frame_filename = f"frame_{frame_index:04d}.png"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_path, detected_frame)
                    
                    if frame_index < len(gps_df):
                        gps_row = gps_df.iloc[frame_index]
                        latitude, longitude = gps_row['Latitude'], gps_row['Longitude']
                    else:
                        latitude, longitude = None, None
                    
                    for (x1, y1, x2, y2, confidence) in boxes:
                        detection_data.append([frame_filename, x1, y1, x2, y2, confidence, latitude, longitude])
                
                out.write(detected_frame)
                frame_index += 1
            
            video.release()
            out.release()
            
            excel_path = os.path.join(temp_dir, "pothole_coordinates.xlsx")
            df = pd.DataFrame(detection_data, columns=["Frame", "X1", "Y1", "X2", "Y2", "Confidence", "Latitude", "Longitude"])
            df.to_excel(excel_path, index=False)
            
            zip_path = os.path.join(temp_dir, "processed_results.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(output_video_path, "processed_video.mp4")
                zipf.write(excel_path, "pothole_coordinates.xlsx")
                for frame in os.listdir(frames_dir):
                    zipf.write(os.path.join(frames_dir, frame), os.path.join("frames", frame))
            
            with open(zip_path, "rb") as file:
                if st.download_button("Download All Processed Data (ZIP)", file, file_name="processed_results.zip", mime="application/zip"):
                    st.session_state.clear()
                    st.rerun()

if __name__ == "__main__":
    main()
