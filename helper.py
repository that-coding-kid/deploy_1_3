from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube

import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def input_video(conf, model):
    #User selects a video file using Streamlit's file_uploader

    video_source = st.sidebar.file_uploader("Choose a video...", type=("mp4", "avi", "mov"))
    
    # If a video file is selected
    if video_source is not None:
        # Get the path of the selected video file
        print(video_source.name)
        video_path = os.path.abspath(video_source.name)
        print(video_path)
        
        # Process the video here
        st.success(f"Selected video: {video_path}")
        
        # Datasetcreation.py
        
        # Function to subtract timestamps and calculate time difference in seconds
        def subtract_timestamps(timestamp1, timestamp2):
            time_format = "%H:%M:%S"
            dt1 = datetime.strptime(timestamp1, time_format)
            dt2 = datetime.strptime(timestamp2, time_format)
            time_difference = dt1 - dt2
            total_seconds = time_difference.total_seconds()
            return total_seconds
        
        # Define video length in minutes and seconds
        m_VideoLength = 21
        s_VideoLength = 16
        
        # Define start and end timestamps for the desired clip
        clip_start_time = "10:00:12"
        clip_end_time = "10:19:59"
        
        # Calculate the duration of the clip in seconds
        clipDuration = subtract_timestamps(clip_end_time, clip_start_time)
        
        # Calculate the total duration of the video in seconds
        videoDuration = (m_VideoLength * 60) + s_VideoLength
        
        # Calculate the offset constant for slicing the video
        offSetConstant = clipDuration / videoDuration
        
        # Define the starting timestamp for slicing the video
        start_time_stamp = 11
        
        # Define a cycle of durations for each clip
        cycle = [50, 35, 40]
        
        # Define the total duration of all clips
        total_duration = 19 + (21 * 60)
        
        # Initialize a counter variable
        i = 0
        
        # Loop to slice the video into multiple clips
        while True:
            # Get the index of the current cycle duration
            index = i % 3
            
            # Calculate the end timestamp for the current clip
            slice_end = start_time_stamp + (cycle[index] / offSetConstant)
            
            # If the end timestamp exceeds the total duration, break the loop
            if slice_end > total_duration:
                break
            
            # Define the name of the current clip
            clip_name = f"clip{i}.mp4"
            
            # Define the input video path
            input_video = f"{video_path}"
            
            # Create a folder named subclips if it doesn't exist
            subclips_folder = "subclips"
            if not os.path.exists(subclips_folder):
                os.makedirs(subclips_folder)
            
            # Use ffmpeg to extract the subclip from the input video
            output_path = os.path.join(subclips_folder, clip_name)
            ffmpeg_tools.ffmpeg_extract_subclip(input_video, start_time_stamp, slice_end, output_path)
            
            # Increment the counter variable and update the start timestamp
            i += 1
            start_time_stamp = slice_end
    else:
        st.warning("Please choose a video file.")
