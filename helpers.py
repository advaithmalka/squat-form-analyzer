import mediapipe as mp
import cv2
import numpy as np
mp_pose = mp.solutions.pose
def get_keypoints(image):
  # Convert the image to RGB format
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Initialize the Pose estimator
  with mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.5) as pose: # confidence determines speed
    # Process the image through the Pose estimator
    results = pose.process(image_rgb)

    # Initialize an empty list to store the keypoints
    keypoints = []
    if not results.pose_landmarks: return None
    # Iterate over the detected keypoints
    for landmark in results.pose_landmarks.landmark:
      # Get the coordinates of the landmark
      x, y, z = landmark.x, landmark.y, landmark.z
      # Append the keypoint to the list
      keypoints.append((x, y, z))
    
    return (keypoints, results.pose_landmarks)
  
def get_feedback(keypoints):
    # angles = extract_angles(keypoints)
    feedback = "Great squat form!"
    
    # Knee angle feedback
    body_parts = mp_pose.PoseLandmark
    left_knee_parallelism = check_parallelism((keypoints[body_parts.LEFT_KNEE.value][0], keypoints[body_parts.LEFT_KNEE.value][1]), 
                  (keypoints[body_parts.LEFT_HIP.value][0], keypoints[body_parts.LEFT_HIP.value][1]))
    right_knee_parallelism = check_parallelism((keypoints[body_parts.RIGHT_KNEE.value][0], keypoints[body_parts.RIGHT_KNEE.value][1]), 
                  (keypoints[body_parts.RIGHT_HIP.value][0], keypoints[body_parts.RIGHT_HIP.value][1]))
    
    THRESHOLD = 10
    left_knee_check = (left_knee_parallelism < THRESHOLD) ^ (abs(left_knee_parallelism - 180) < THRESHOLD)
    right_knee_check = (right_knee_parallelism < THRESHOLD) ^ (abs(right_knee_parallelism - 180) < THRESHOLD)
    if  not (left_knee_check or right_knee_check):
        feedback = "Squat lower"         
    
    return feedback

def check_parallelism(p1: tuple[float, float], p2:tuple[float, float]):
    v1= np.array([1,0]) # parallel ground vector
    v2= np.array([p2[0] - p1[0], p2[1] - p1[1]])

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return False  # vectors have zero length, can't be parallel
    
    # check if vectors are parallel using dot product 
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    # Ensure the cosine value is in the valid range [-1, 1] to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees