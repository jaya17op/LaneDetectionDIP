import streamlit as st
import tempfile
import cv2
import numpy as np
from lane import LaneDetection
from streamlit_option_menu import option_menu

# Sidebar UI
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 10px;'>
            <h2 style='color: #4CAF50; margin-bottom: 0;'>🚦 Lane App</h2>
            <p style='color: #6c757d; font-size: 14px;'>Navigation Panel</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = option_menu(
        menu_title=None,
        options=["Home", "About", "Lane Detection"],
        icons=["house", "info-circle", "car-front-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0px", "background-color": "#"},
            "icon": {"color": "#4CAF50", "font-size": "20px"},
            "nav-link": {
                "font-size": "17px",
                "text-align": "left",
                "margin": "5px 0",
                "padding": "10px",
                "border-radius": "8px",
                "color": "#",
                "--hover-color": "#262730"
            },
            "nav-link-selected": {
                "background-color": "#262730",
                "color": "white",
                "font-weight": "bold"
            },
        }
    )

# Lane detection function
def run_lane_detection(video_path):
    lane = LaneDetection()
    src_pts = [(200, 580), (600, 395), (740, 395), (1080, 580)]
    dst_pts = [(0, 300), (0, 0), (300, 0), (300, 300)]

    K, D = lane.readCameraParameters('2')
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.undistort(frame, K, D, None, K)
        image = cv2.resize(frame, (1280, 600))
        warped, H = lane.getWarpedImage(image, np.float32(src_pts), np.float32(dst_pts))
        mask = lane.doColorThresholding(warped)
        hist, left_lane_ix, right_lane_ix = lane.generateHistogram(mask)
        result, sliding, turning = lane.getLanes(mask, left_lane_ix, right_lane_ix)

        size = (image.shape[:2][1], image.shape[:2][0])
        reverse_warp = cv2.warpPerspective(result, np.linalg.inv(H), size)
        final = cv2.addWeighted(image, 1, reverse_warp, 1, 0)
        cv2.putText(final, turning, (450, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        stframe.image(final, channels="BGR")

    cap.release()

# Page routing
if page == "Home":
    st.markdown(
        """
        <style>
        .title {
            font-size: 40px;
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #6c757d;
            margin-bottom: 40px;
        }
        .feature {
            font-size: 18px;
            padding: 10px;
            margin: 10px 0;
            background-color: *FFF000;
            border-left: 5px solid #4CAF50;
            border-radius: 8px;
        }
        </style>
        <div class='title'>🚗 Welcome to the Lane Detection System</div>
        <div class='subtitle'>Real-time road lane detection using computer vision and Streamlit</div>
        """,
        unsafe_allow_html=True
    )

    # st.image(r"data\0_b5ptHu0y7wUeMddy.jpg", use_column_width=True, caption="Smart Driving with Lane Detection")

    st.markdown(
        """
        <div class='feature'>✅ Upload dashcam videos and detect lanes in real-time.</div>
        <div class='feature'>✅ Get lane guidance (Left/Right) overlayed on the video.</div>
        <div class='feature'>✅ Uses OpenCV, NumPy, and perspective transforms.</div>
        <div class='feature'>✅ Designed for Dataset 2 - Challenge Dataset.</div>
        <div class='feature'>📊 Built with Streamlit for easy deployment and UI.</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("👉 Use the sidebar to get started with **Lane Detection** or learn more in the **About** section.")


elif page == "About":
    st.title("🚗 Lane Detection System")
    
    st.markdown("---")
    
    st.markdown("""
    ### 👨‍💻 Project Overview
    This project presents a **real-time lane detection system** built using **Python**, **OpenCV**, and **Streamlit**. It's designed to identify road lanes from dashcam footage and visualize key insights like lane curvature and vehicle deviation — making it an ideal foundation for driver-assistance and autonomous navigation systems.
    
    ---
    ### 🔍 Key Features
    - 🎥 **Real-Time Detection** from dashcam videos
    - 🔄 **Perspective Transform** for top-down road view
    - 🎯 **Color Thresholding & Histogram Peaks** for lane localization
    - 🧠 **Lane Deviation Indicator** (left or right drift detection)
    - 🖼️ **Visual Feedback** with intuitive overlays
    - ⏱️ Smooth UI via **Streamlit** for real-time interactivity
    
    ---
    ### 🧰 Tools & Libraries
    - 🐍 Python 3
    - 📷 OpenCV
    - 🧮 NumPy
    - 🌐 Streamlit
    
    ---
    ### 🗂️ Dataset Info
    - Trained and tested on **Challenge Dataset (Dataset 2)**
    - Captures various driving conditions with road curvature and shadows
    
    ---
    ### 📌 Use Cases
    - 🧭 Advanced Driver Assistance Systems (ADAS)
    - 📚 Computer Vision Projects in Academia
    - 🚘 Autonomous Vehicle Research & Prototyping
    - 📊 Real-time traffic behavior analysis
    
    ---
    ### 🎓 Background
    Developed as a **university-level computer vision project**, this application showcases the real-world potential of lane detection using fundamental image processing techniques. It's a great learning resource for anyone interested in self-driving car tech, real-time video processing, and machine vision.

    ---
    ### 🤝 Contributors
    - Jayadhar
    - Giridhar
    - Sachin
    - Karthik
    - Sujan
    - Greeshma

    ---
    📂 **Repository**: [GitHub Link](https://github.com/jaya17op/LaneDetectionDIP)
    """)

    # st.image("data/sample_output.jpg", caption="Sample Lane Detection Output", use_column_width=True)


elif page == "Lane Detection":
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtext {
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .uploaded-success {
            color: green;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🚗 Lane Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Challenge Video (Dataset 2 Only)<br>Upload your video to see the lane detection results.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        with st.spinner("Processing video..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            st.markdown('<p class="uploaded-success">✅ Video uploaded successfully!</p>', unsafe_allow_html=True)
            run_lane_detection(tmp_path)
    else:
        st.info("Please upload a valid video file to get started.")