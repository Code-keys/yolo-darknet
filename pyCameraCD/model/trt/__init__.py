from .yolo_trt import *
#from .yolov3_trt import YoLov3TRT
from .yolov4_trt import *

# __all__ = ['YoLov5TRT',"YoLov3TRT","YoLov4TRT"]
__all__ = ["YoLov4TRT","YoLoTRT"]



engine_file_path = "/home/nvidia/pyCameraCD/data/yolo.engine"

# a  YoLoTRT instance
yolov5_wrapper = YoLoTRT(engine_file_path,"/home/nvidia/pyCameraCD/data/classes.names")

# from https://github.com/ultralytics/yolov5/tree/master/inference/images
input_image_paths = ["/home/nvidia/pyCameraCD/data/bin/test.png", "/home/nvidia/pyCameraCD/data/bin/test.png"]


image = cv2.imread(input_image_paths[0])

img = yolov5_wrapper.draw_detect_results(image)

cv2.imwrite("/home/nvidia/pyCameraCD/data/bin/test1.png",img)

yolov5_wrapper.destory()