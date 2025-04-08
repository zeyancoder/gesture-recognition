import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.framework.formats import landmark_pb2

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 5, 9, 13, 17]

GESTURE_MAP = {
    "Fist": [0, 0, 0, 0, 0],
    "Hello": [1, 1, 1, 1, 1],
    "Thumbs Up": [1, 0, 0, 0, 0],  
    "Thumbs Down": [1, 0, 0, 0, 0], 
    "Peace": [0, 1, 1, 0, 0],
    "Go!": [0, 1, 0, 0, 0],
    "Call Me": [1, 0, 0, 0, 1],
    "Three Fingers": [0, 1, 1, 1, 0],
    "Rock Sign": [1, 0, 0, 0, 1],
    "OK Sign": [1, 1, 1, 0, 1],
    "Finger Gun": [1, 1, 0, 0, 0],
    "Spiderman": [1, 0, 0, 1, 1],
    "Love You": [1, 1, 0, 0, 1],
    "Middle Finger": [0, 0, 1, 0, 0],
}

model_path = "models/hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands_style = mp.solutions.hands

def get_finger_states(landmarks, threshold=0.03):
    fingers = []

    fingers.append(1 if landmarks[4].x > landmarks[3].x + threshold else 0)

    for tip_idx, mcp_idx in zip(FINGER_TIPS[1:], FINGER_MCP[1:]):
        tip_y = landmarks[tip_idx].y
        mcp_y = landmarks[mcp_idx].y
        fingers.append(1 if mcp_y - tip_y > threshold else 0)

    return fingers

def classify_gesture(fingers, landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    wrist = landmarks[0]

    # Handle Thumb-only gestures
    if fingers == [1, 0, 0, 0, 0]:
        vertical_thresh = 0.1
        horizontal_thresh = 0.15

        y_diff = thumb_tip.y - wrist.y
        x_diff = abs(thumb_tip.x - wrist.x)

        if y_diff < -vertical_thresh:
            return "Thumbs Up"
        elif y_diff > vertical_thresh:
            return "Thumbs Down"
        elif x_diff > horizontal_thresh:
            return "Thumb Extended"
        else:
            return ""

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"

    for name, pattern in GESTURE_MAP.items():
        if fingers == pattern:
            return name

    return ""



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam error")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    annotated_frame = frame.copy()

    try:
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                landmark_list = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                        for lm in hand_landmarks
                    ]
                )

                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmark_list,
                    mp_hands_style.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                fingers = get_finger_states(hand_landmarks, threshold=0.03)
                gesture = classify_gesture(fingers, hand_landmarks)

                x = int(hand_landmarks[0].x * frame.shape[1])
                y = int(hand_landmarks[0].y * frame.shape[0]) - 20
                cv2.putText(annotated_frame, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2, cv2.LINE_AA)

    except Exception as e:
        print("🔥 Error:", e)

    cv2.imshow("Advanced Gesture Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
