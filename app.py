import streamlit as st
import tempfile
import cv2
import os
from lane_detection import detect_lane

st.title("Lane Detection System 🚗")
st.write("Upload a video file to detect lanes and get deviation warnings.")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))  # Resize to a manageable size

        lane_frame = detect_lane(frame)

        # Convert BGR to RGB for Streamlit display
        lane_frame_rgb = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2RGB)

        stframe.image(lane_frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

    # Ensure file is closed before deleting
    import time
    time.sleep(1)
    os.remove(video_path)

