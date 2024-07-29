import mediapipe as mp
from helpers import *
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
# init mediapipe pose api
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


class PoseVideoProcessor(VideoProcessorBase):
  def __init__(self):
    self.feedback = ""
    super().__init__()

  def recv(self, frame):
    image = frame.to_ndarray(format='bgr24')
    results = get_keypoints(image)
    if results:
      (keypoints, pose_landmarks) = results
      mp_drawing.draw_landmarks(
        image, pose_landmarks, mp_pose.POSE_CONNECTIONS
      )

      feedback = get_feedback(keypoints)
      
      cv2.putText(img=image, text=feedback, org=(10, 30), 
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                  color=(0, 255, 0), thickness=2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI setup
st.title("Real-time Squat form Feedback with Mediapipe")
st.subheader("By: Advaith Malka")


webrtc_ctx = webrtc_streamer(
  key="squat form analyizer",
  mode=WebRtcMode.SENDRECV,
  video_processor_factory=PoseVideoProcessor,
  rtc_configuration={
    "iceServers": get_ice_servers(),
    "iceTransportPolicy": "relay",
  },
  media_stream_constraints={"video": True, "audio": False},
  async_processing=True,
)

st.markdown("""
## Notes for Using the Squat Analysis App

1. **Make sure the camera is straight and not at an angle with respect to the floor.**
2. **Keep the camera perpendicular to you when squatting.**
""")
