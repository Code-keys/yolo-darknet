import ctypes, os, cv2
from ctypes import *
import numpy as np
import platform
from PIL import Image, ImageDraw, ImageFont

cwd = os.path.dirname(os.path.realpath(__file__))
os.environ['PATH'] = cwd + ';' + os.environ['PATH']

sysstr = platform.system()
if sysstr == "Windows":
    lib = ctypes.cdll.LoadLibrary(os.path.join(cwd, 'yolo.dll'))
elif sysstr == 'Linux':
    lib = ctypes.cdll.LoadLibrary(os.path.join(cwd, "./libyolov3.so"))
    CDLL(os.path.join(cwd, "libmypluginsv3.so"))
    
class Detection(Structure):
    _fields_ = [("bbox", c_float*4),
                ("conf", c_float),
                ("classid", c_float),
                ("class_confidence", c_float)
                ]

#https://www.jianshu.com/p/0306a9898d68
class auto_YOLOv3Detector(object):
    def __init__(self, weights='yolov5s.engine', classnames ='', input_dim=640, conf_thresh=0.25, nms_thresh=0.45):
        super(auto_YOLOv3Detector, self).__init__()
        lib.init_detect.argtypes = [POINTER(c_char), c_int, c_float, c_float]
        lib.init_detect.restype = c_void_p
        lib.perform_detect.argtypes = [c_void_p, c_int, c_int, POINTER(c_ubyte), POINTER(c_int)]
        lib.perform_detect.restype = POINTER(Detection)
        lib.uninit_detect.argtypes = [c_void_p,]
        lib.free_detect.argtypes = [POINTER(Detection), c_int]
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.obj = lib.init_detect(bytes(weights, 'utf-8'), int(input_dim), float(self.conf_thresh), float(self.nms_thresh))

        if os.path.exists(classnames):
            with open(classnames, 'r', encoding='utf-8') as fp:
                self.class_names = [line.strip() for line in fp.readlines()]
        else:
            self.class_names = [str(i) for i in range(1000)]
        self.dets = []
        self.weights = weights
        self.img = None
        self.font = None

    def __del__(self):
        lib.uninit_detect(self.obj)

    @property
    def CLASSES(self):
        return self.class_names

    def detect(self, im0, thresh=0.25):

        img = np.ascontiguousarray(im0)
        (rows, cols) = (img.shape[0], img.shape[1])
        num = c_int(0)
        pnum = pointer(num)
        pdata = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        _dets = lib.perform_detect(self.obj, rows, cols, pdata, pnum)
        num = pnum[0]
        self.dets.clear()
        for i in range(num):
            box = _dets[i].bbox
            conf = _dets[i].conf
            id = _dets[i].classid
            if conf <= thresh: #area filter
                continue
            self.dets.append((self.class_names[int(id)], conf, (box[0], box[1], box[2], box[3])))
        # lib.free_detect(_dets, num)
        return self.dets

    def showchinese(self, img, pos, str, color=(0,255,0)):
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        if self.font is None:
            scale = 40.0 / 1920.0
            self.font = ImageFont.truetype(os.path.dirname(__file__)+"/simhei.ttf", int(scale*img.shape[1]), encoding="utf-8")
        draw.text(pos, str, font=self.font, fill=color)
        img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        return img_OpenCV

    def draw_detect_results(self, im, detections=[], color=(0, 0, 255), width=2):
        if detections is None or len(detections) == 0:

            detections = self.detect(im)
        self.img = im

        for det in detections:
            class_name = det[0]  # 目标类别名称
            prob = det[1]  # 目标概率
            cx = int(det[2][0])  # 目标中心点横坐标
            cy = int(det[2][1])  # 目标中心点纵坐标
            w = int(det[2][2])  # 目标宽度
            h = int(det[2][3])  # 目标高度
            l = cx - w // 2
            t = cy - h // 2
            cv2.rectangle(self.img, (l, t), (l + w, t + h), color, thickness=width)
            info = '%s' % (class_name)
            t = max(10, t - 2)
            # cv2.putText(im, info, (l, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 128, 0), width)
            # self.img = self.showchinese(self.img, (l, t), info)
        return self.img

    def __del__(self):
        try:
            lib.uninit_detect(self.obj)

        finally:
            pass

if __name__ == "__main__":
    # load custom plugins

    engine_file_path = "/home/nvidia/pyCameraCD/data/yolo.engine"

    # load coco labels
    print(os.path.dirname(__file__) +"/bin")

    categories = ["ship", "ship"]

    # a  YoLoTRT instance
    yolov5_wrapper = auto_YOLOv3Detector(engine_file_path,"/home/nvidia/pyCameraCD/data/classes.names")

    # from https://github.com/ultralytics/yolov5/tree/master/inference/images
    input_image_paths = ["/home/nvidia/pyCameraCD/data/bin/1.jpg", "/home/nvidia/pyCameraCD/data/bin/test.png"]


    image = cv2.imread(input_image_paths[0])

    img = yolov5_wrapper.draw_detect_results(image)


    image = cv2.imwrite("/home/nvidia/pyCameraCD/data/bin/111111.png",img)
