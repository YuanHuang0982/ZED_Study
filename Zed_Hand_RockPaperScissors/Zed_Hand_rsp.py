import cv2
import numpy as np
import pyzed.sl as sl
import mediapipe as mp

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ZED 카메라 초기화
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # HD1080 해상도 설정
init_params.camera_fps = 30  # FPS 설정
zed.open(init_params)

# 손가락의 굽힘 상태를 판단하는 함수
def is_finger_bent(landmarks, finger_tip_index, finger_base_index):
    return 1 if landmarks[finger_tip_index].y > landmarks[finger_base_index].y else 0

# 제스처 인식 함수
def recognize_gesture(landmarks):
    # 각 손가락의 굽힘 상태를 1과 0으로 저장
    index_finger_bent = is_finger_bent(landmarks, 8, 6)  # 검지
    middle_finger_bent = is_finger_bent(landmarks, 12, 10)  # 중지
    ring_finger_bent = is_finger_bent(landmarks, 16, 14)  # 약지
    pinky_finger_bent = is_finger_bent(landmarks, 20, 18)  # 새끼손가락
    thumb_bent = is_finger_bent(landmarks, 4, 3)  # 엄지

    # 굽힘 상태 배열
    finger_states = [thumb_bent, index_finger_bent, middle_finger_bent, ring_finger_bent, pinky_finger_bent]

    # 제스처 판별
    if finger_states == [0, 0, 0, 1, 1]:
        return "Scissors"  # 가위
    elif finger_states == [0, 1, 1, 1, 1]:
        return "Rock"  # 바위
    elif finger_states == [0, 0, 0, 0, 0]:
        return "Paper"  # 보
    else:
        return None  # 기타

try:
    while True:
        # ZED 카메라에서 이미지 캡처
        image = sl.Mat()
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()  # (720, 1280, 4) 형태의 배열을 반환합니다.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 4채널을 3채널로 변환

            # MediaPipe로 손 인식
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # 손 랜드마크가 있는 경우
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 제스처 인식
                    gesture = recognize_gesture(hand_landmarks.landmark)
                    cv2.putText(frame, gesture, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

                    # 랜드마크 좌표 출력 (디버깅용)
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        print(f"Landmark {i}: (x: {landmark.x}, y: {landmark.y})")

            # 결과 출력
            cv2.imshow("ZED 2i Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    zed.close()
    cv2.destroyAllWindows()