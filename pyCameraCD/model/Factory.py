import threading

import numpy as np

from .darknet import *
from .trt import *
# sys.path.append('./darknet')

class myThread(threading.Thread):
    def __init__(self, func, args, out=False):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.out = out

    def run(self):
        try:
            self.result = self.func(*self.args)
        except Exception:
            pass

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

class ModelFactory(object): # Adaptor && Factory
    def __init__(self):
        self.fps = 0
        self.model = None
    def draw_detect_results(self,img):
        if self.model is not None:
            return self.model.draw_detect_results(img)

    @classmethod
    def build(self, string, args=()):
        name = str.lower(string)
        try:
            if "rt" in name:
                if "yolov5" in name:
                    self.model = YoLov5TRT(args[1])
                elif "yolov4" in name:
                    self.model = YoLov4TRT(*args)
                elif "yolov4" in name:
                    self.model = YoLov3TRT(*args)
            elif "dark" in name:
                self.model = Darknet(*args)
            else:
                pass  # self.model = other(*args)
                print("check again ")
        except:
            print("check again ")
        return self.model

    def slice_infer(self, img, fast=False):
        if fast:
            return slice2Batch_detect(model=self.model.infer, img=img,
                                      chip_size=min(img.shape[0], img.shape[1]) / 2 + 150,
                                      slide_size=min(img.shape[0], img.shape[1]) / 2 + 1)
        return self.model.infer(img)

def py_cpu_nms(dets, thresh):
    # dets:(m,5)  thresh:scaler

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []

    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # 计算iou
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]  # index下标是从1开始的，而where计算的下标是从0开始的，故需要+1

    return keep

def slice_detect(self, img, Model_infer_ptr, slide_size, chip_size):
    height, width, channel = img.shape
    slide_h, slide_w = slide_size
    hn, wn = chip_size
    # TODO: check the corner case
    # import pdb; pdb.set_trace()
    total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]

    for i in (range(int(width / slide_w + 1))):
        for j in range(int(height / slide_h) + 1):
            subimg = np.zeros((hn, wn, channel))
            # print('i: ', i, 'j: ', j)
            chip = img[j * slide_h:j * slide_h + hn, i * slide_w:i * slide_w + wn, :3]
            subimg[:chip.shape[0], :chip.shape[1], :] = chip

            chip_detections = Model_infer_ptr(subimg)

            # print('result: ', result)
            for cls_id, name in enumerate(self.classnames):
                chip_detections[cls_id][:, :8][:, ::2] = chip_detections[cls_id][:, :8][:, ::2] + i * slide_w
                chip_detections[cls_id][:, :8][:, 1::2] = chip_detections[cls_id][:, :8][:, 1::2] + j * slide_h
                # import pdb;pdb.set_trace()
                try:
                    total_detections[cls_id] = np.concatenate((total_detections[cls_id], chip_detections[cls_id]))
                except:
                    import pdb
                    pdb.set_trace()

    return total_detections

def slice2Batch_detect(model, img, slide_size, chip_size):  # for middle picture
    pass

def slice_MulThreadDetect(img, Model_infer_ptr, slide_size, chip_size):
    pass
    """
    img: [ nb, channel, w, h]
    model: a point of infer function

    1\ using big_batch to inference
    2\ using threading to inference

    types = 1
    import threading
    from  tqdm import  tqdm
    from threading import Lock,Thread

    height, width, channel = img.shape
    slide_h, slide_w = slide_size
    hn, wn = chip_size
    # TODO: check the corner case
    # import pdb; pdb.set_trace()
    total_detections = [np.zeros((0, 9)) for _ in range(len(classnames))]

    if types == 1:

        threads =  list()
        for i in tqdm(range(int(width / slide_w + 1))):
            for j in range(int(height / slide_h) + 1):

                subimg = np.zeros((hn, wn, channel))
                # print('i: ', i, 'j: ', j)
                chip = img[j * slide_h:j * slide_h + hn, i * slide_w:i * slide_w + wn, :3]
                subimg[:chip.shape[0], :chip.shape[1], :] = chip

               

                chip_detections = Model_infer_ptr(self.model, subimg)

        for i in tqdm(range(int(width / slide_w + 1))):
            for j in range(int(height / slide_h) + 1):

                # print('result: ', result)
                for cls_id, name in enumerate(self.classnames):
                    chip_detections[cls_id][:, :8][:, ::2] = chip_detections[cls_id][:, :8][:, ::2] + i * slide_w
                    chip_detections[cls_id][:, :8][:, 1::2] = chip_detections[cls_id][:, :8][:, 1::2] + j * slide_h
                    # import pdb;pdb.set_trace()
                    try:
                        total_detections[cls_id] = np.concatenate((total_detections[cls_id], chip_detections[cls_id]))
                    except:
                        import pdb
                        pdb.set_trace()

        return total_detections
    """

def slice2Batch_MulThreadDetect(model, img, slide_size, chip_size):  # for middle picture
    pass

if __name__ == '__main__':
    model = ModelFactory.build("darknet", ())
    import cv2

    img = cv2.imread("../resources/black.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = model.detect_from_image(img)

    cv2.namedWindow('title', cv2.WINDOW_NORMAL)
    cv2.imshow('title', result)
    k = cv2.waitKey(1)
