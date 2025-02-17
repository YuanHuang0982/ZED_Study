import pyzed.sl as sl
import cv2
import numpy as np
import time
import signal
import threading
from ultralytics import YOLO

stop_signal = False

def signal_handler(signal, frame):
    global stop_signal
    stop_signal = True
    time.sleep(0.5)
    exit()

def grab_and_process(index, zed, model):
    global stop_signal
    runtime = sl.RuntimeParameters()
    left_image = sl.Mat()
    depth_map = sl.Mat()

    camera_name = f"ZED {index}"
    print(f"Starting camera thread: {camera_name}")  # Debugging message

    # Set window size (640x480)
    cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)  # Make the window resizable
    cv2.resizeWindow(camera_name, 640, 480)  # Resize the window

    while not stop_signal:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            frame = left_image.get_data().copy()

            # Debugging: Verify frame capture
            print(f"Camera {index}: Frame captured")

            # 🔹 Convert RGBA → RGB (compatible with YOLOv8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Perform YOLOv8 object detection
            results = model(frame)

            person_detected = False

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, cls in zip(boxes, classes):
                    if int(cls) == 0:  # class 0 = person
                        person_detected = True
                        x1, y1, x2, y2 = map(int, box)
                        person_center_x = (x1 + x2) // 2
                        person_center_y = (y1 + y2) // 2

                        err, depth_value = depth_map.get_value(person_center_x, person_center_y)
                        distance_text = "{}mm".format(round(depth_value)) if np.isfinite(depth_value) else "No Depth Data"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, distance_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            if not person_detected:
                cv2.putText(frame, "No person detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # 🔹 Debugging: Display a blank screen if no valid frame exists
            if frame is None or frame.shape[0] == 0:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                print(f"Camera {index}: No valid frame, showing blank screen")

            cv2.imshow(camera_name, frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            stop_signal = True
            break

    print(f"Closing {camera_name}")
    zed.close()

def main():
    global stop_signal
    signal.signal(signal.SIGINT, signal_handler)

    print("Initializing ZED Cameras...")

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")
    # Force disable the model's fuse() function
    model.model.fuse = lambda *args, **kwargs: model.model

    # Get the list of available ZED cameras
    cameras = sl.Camera.get_device_list()
    if len(cameras) < 2:
        print("At least two ZED cameras are required!")
        exit()

    # Initialize ZED cameras
    zed_list = []
    threads = []
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Open each camera
    for cam in cameras[:2]:  # Use only two cameras
        print(f"Opening ZED {cam.serial_number}")  # Debugging message
        init_params.set_from_serial_number(cam.serial_number)
        zed = sl.Camera()
        if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open ZED {cam.serial_number}")
            continue
        zed_list.append(zed)

    if len(zed_list) != 2:
        print("Both cameras could not be opened.")
        exit()

    print("Cameras successfully opened. Starting threads...")

    # Start a thread for each camera
    for index, zed in enumerate(zed_list):
        thread = threading.Thread(target=grab_and_process, args=(index, zed, model))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
    print("\nFINISH")

if __name__ == "__main__":
    main()
