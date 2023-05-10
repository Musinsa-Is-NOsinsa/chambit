import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic


_POSE_LANDMARKS_UPPER = [
    mp_holistic.PoseLandmark.LEFT_SHOULDER,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    mp_holistic.PoseLandmark.LEFT_ELBOW,
    mp_holistic.PoseLandmark.RIGHT_ELBOW,
]


def get_upper_landmarks(image):
    upper_landmarks = []
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
    ) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:

            image_height, image_width, _ = image.shape
            for landmark in _POSE_LANDMARKS_UPPER:
                x = results.pose_landmarks.landmark[landmark].x * image_width
                y = results.pose_landmarks.landmark[landmark].y * image_height
                upper_landmarks.append([x, y])

    return upper_landmarks
