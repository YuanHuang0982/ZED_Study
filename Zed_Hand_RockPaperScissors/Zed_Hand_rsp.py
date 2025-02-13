import cv2
import numpy as np
import pyzed.sl as sl
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set resolution to HD1080
init_params.camera_fps = 30  # Set FPS
zed.open(init_params)

# Function to determine if a finger is bent
def is_finger_bent(landmarks, finger_tip_index, finger_base_index):
    return 1 if landmarks[finger_tip_index].y > landmarks[finger_base_index].y else 0

# Gesture recognition function
def recognize_gesture(landmarks):
    # Store the bending state of each finger as 1 or 0
    index_finger_bent = is_finger_bent(landmarks, 8, 6)  # Index finger
    middle_finger_bent = is_finger_bent(landmarks, 12, 10)  # Middle finger
    ring_finger_bent = is_finger_bent(landmarks, 16, 14)  # Ring finger
    pinky_finger_bent = is_finger_bent(landmarks, 20, 18)  # Pinky finger
    thumb_bent = is_finger_bent(landmarks, 4, 3)  # Thumb

    # Array of finger bending states
    finger_states = [thumb_bent, index_finger_bent, middle_finger_bent, ring_finger_bent, pinky_finger_bent]

    # Gesture recognition
    if finger_states == [0, 0, 0, 1, 1]:
        return "Scissors"  # Scissors
    elif finger_states == [0, 1, 1, 1, 1]:
        return "Rock"  # Rock
    elif finger_states == [0, 0, 0, 0, 0]:
        return "Paper"  # Paper
    else:
        return None  # Other

try:
    while True:
        # Capture image from ZED camera
        image = sl.Mat()
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()  # Returns an array of shape (720, 1280, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert 4 channels to 3 channels

            # Hand detection with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # If hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Recognize gesture
                    gesture = recognize_gesture(hand_landmarks.landmark)
                    cv2.putText(frame, gesture, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

                    # Print landmark coordinates (for debugging)
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        print(f"Landmark {i}: (x: {landmark.x}, y: {landmark.y})")

            # Display results
            cv2.imshow("ZED 2i Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    zed.close()
    cv2.destroyAllWindows()
