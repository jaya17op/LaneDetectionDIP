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

    # Read camera parameters (adjust call as per your LaneDetection implementation)
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
    This project presents a **real-time lane detection system** built using **Python**, **OpenCV**, and **Streamlit**.
    """)
elif page == "Lane Detection":
    st.markdown('<div class="main-title">🚗 Lane Detection</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        with st.spinner("Processing video..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            run_lane_detection(tmp_path)
    else:
        st.info("Please upload a valid video file to get started.")