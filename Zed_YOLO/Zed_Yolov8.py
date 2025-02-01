import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # YOLOv8 사전 학습된 모델

# ZED 카메라 초기화
def initialize_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD720
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED 카메라 초기화 실패")
        exit(1)
    return zed

# 메인 함수
def main():
    zed = initialize_zed()
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    print("실시간으로 사람을 검출합니다. 종료하려면 'q'를 누르세요.")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # ZED에서 RGB 이미지 가져오기
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame = frame[:, :, :3].copy()  # BGRA → BGR

            # YOLOv8 객체 탐지
            results = model(frame)  # YOLOv8 추론
            for result in results:
                boxes = result.boxes  # Boxes 객체
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 좌표
                    conf = box.conf[0].cpu().numpy()  # 신뢰도
                    class_id = int(box.cls[0].cpu().numpy())  # 클래스 ID

                    # "사람"만 필터링 (COCO 데이터셋에서 클래스 ID 0은 "person")
                    if class_id == 0 and conf > 0.5:
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        # 사람 박스 그리기
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
                        # 신뢰도 표시
                        cv2.putText(frame, f'Person: {conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 결과 출력
            cv2.imshow("ZED + YOLOv8 Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()