# Import necessary libraries
import streamlit as st
import io
import os
import cv2
import utils
from s_calculator import SpeedCalCulator 

config_con = utils.read_config()
sc_con = config_con.get("root_dir")
artifact_con = config_con.get("artifact")

SC_DIR_NAME = os.path.join(artifact_con.get("root_dir"), sc_con.get("root_dir"))

sc = SpeedCalCulator()

st.title("Speed Calculator")

# Create a file uploader widget for selecting a video file
uploaded_file = st.file_uploader("Choose a Video...", type=["mp4"])
temporary_location = False

# Check if a video file has been uploaded
if uploaded_file is not None:
    g = io.BytesIO(uploaded_file.read())  # Create a BytesIO object
    temporary_location = os.path.join(SC_DIR_NAME, sc_con.get("input_video_file_name"))

    with open(temporary_location, 'wb') as out:
        out.write(g.read())  # Write the uploaded file to a temporary location as bytes

    if st.button("Calculate Speed"):

        sc.combine_all()
        print("<<<<<<   Speed Calculate DONE   >>>>>>")

        # Get the path to the output video
        output_video_path = os.path.join(SC_DIR_NAME, sc_con.get("output_video_file_name"))

        # Open the output video file and read its content
        with open(output_video_path, "rb") as f:
            con = f.read()

        # Create a download button for the output video
        st.download_button(label="Download The Video",
                           data=con,
                           file_name='output_video.mp4',
                           )