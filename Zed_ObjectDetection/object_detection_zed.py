import cv2
import numpy as np
import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error opening camera: {err}")
    exit()

# Set initialization parameters for object detection
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True  # Objects will keep the same ID between frames
obj_param.enable_segmentation = True  # Outputs 2D masks over detected objects
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

# Enable positional tracking if object tracking is enabled
if obj_param.enable_tracking:
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

# Enable object detection
error = zed.enable_object_detection(obj_param)
if error != sl.ERROR_CODE.SUCCESS:
    print(f"Error enabling object detection: {error}")
    zed.close()
    exit()

objects = sl.Objects()  # Structure containing all the detected objects
# Set runtime parameters
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
obj_runtime_param.detection_confidence_threshold = 30  # Note: This should be between 0 and 100

cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_objects(objects, obj_runtime_param)  # Retrieve the detected objects

        img = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        
        # Ensure img_cv is a numpy array with correct type
        img_cv = np.array(img.get_data()).copy()

        if objects.is_new:
            obj_arr = objects.object_list
            print(f"Detected objects: {len(obj_arr)}")

            for obj in obj_arr:
                top_left = obj.bounding_box_2d[0]
                bottom_right = obj.bounding_box_2d[2]

                cv2.rectangle(img_cv, (int(top_left[0]), int(top_left[1])),
                                     (int(bottom_right[0]), int(bottom_right[1])),
                                     (0, 255, 0), 2)
                
                label = f"{obj.label} ({int(obj.confidence*100)}%)"  # Convert confidence to percentage

                cv2.putText(img_cv, label, (int(top_left[0]), int(top_left[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Object detection with ZED", img_cv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.disable_object_detection()
zed.close()
cv2.destroyAllWindows()