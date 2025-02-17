import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # YOLOv8 pre-trained model

# Initialize ZED camera
def initialize_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD720
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to initialize ZED camera")
        exit(1)
    return zed

# Main function
def main():
    zed = initialize_zed()
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    print("Detecting people in real-time. Press 'q' to quit.")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Get RGB image from ZED
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame = frame[:, :, :3].copy()  # BGRA → BGR

            # YOLOv8 object detection
            results = model(frame)  # YOLOv8 inference
            for result in results:
                boxes = result.boxes  # Boxes object
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Coordinates
                    conf = box.conf[0].cpu().numpy()  # Confidence
                    class_id = int(box.cls[0].cpu().numpy())  # Class ID

                    # Filter only "person" (class ID 0 in COCO dataset is "person")
                    if class_id == 0 and conf > 0.5:
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        # Draw bounding box for person
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
                        # Display confidence
                        cv2.putText(frame, f'Person: {conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display results
            cv2.imshow("ZED + YOLOv8 Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
