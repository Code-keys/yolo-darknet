import os
import cv2
from .YOLOv3utils import *

class YOLODetector(object):
    def __init__(self, configPath, weightPath, metaPath,dllname="libdarknet", gpuid=0):
        """
        :功能    初始化函数
        :输入    dllname - 用于设定GPU模式还是CPU模式，
                configPath - 模型配置文件路径
                weightPath - 模型权重文件路径
                metaPath - 模型数据文件路劲
                gpuid - GPU索引
        :输出   无
        :说明   无
        """
        super(YOLODetector).__init__()
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.dllname = dllname
        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath
        self.gpuid = gpuid

        self.loadNames()
        self.loadDLL()
        print("dark.so loaded!")
        self.loadConfig()
        self.img = None
        self.infer_fps = 0

    def __del__(self):
        """
        :功能   析构函数
        :输入   无
        :输出   无
        :说明   无
        """
        if self.netMain is not None:
            self.free_network_ptr(self.netMain)

    def loadNames(self):
        """
        :功能   加载类别文件
        :输入   无
        :输出   无
        :说明   无
        """
        if self.altNames is None:
            # In Python 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                        
                        print(result)
                    else:
                        
                        result = None
                    try:
                        print(result)
                        if os.path.exists(result):
                            with open(result, encoding='utf-8') as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def loadConfig(self):
        """
        :功能   加载模型
        :输入   无
        :输出   无
        :说明   无
        """
        configPath = self.configPath
        weightPath = self.weightPath
        metaPath = self.metaPath
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")
        if self.netMain is None:
            self.netMain = self.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaPath = self.convert_datafile(metaPath)
            self.metaMain = self.load_meta(self.metaPath.encode("ascii"))

    def getNames(self):
        return self.altNames

    def loadDLL(self):
        """
        :功能   加载DLL，获取主要函数指针
        :输入   无
        :输出   无
        :说明   无
        """
        self.hasGPU = True
        if os.name == "nt":
            cwd = os.path.dirname(__file__)
            os.environ['PATH'] = cwd + ';' + os.environ['PATH']
            winGPUdll = os.path.join(cwd, "bin/win32/%s.dll" % self.dllname)
            winNoGPUdll = os.path.join(cwd, "%s_nogpu.dll" % self.dllname)
            envKeys = list()
            for k, v in os.environ.items():
                envKeys.append(k)
            try:
                try:
                    tmp = os.environ["FORCE_CPU"].lower()
                    if tmp in ["1", "true", "yes", "on"]:
                        raise ValueError("ForceCPU")
                    else:
                        print("Flag value '" + tmp + "' not forcing CPU mode")
                except KeyError:
                    # We never set the flag
                    if 'CUDA_VISIBLE_DEVICES' in envKeys:
                        if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                            raise ValueError("ForceCPU")
                    try:
                        global DARKNET_FORCE_CPU
                        if DARKNET_FORCE_CPU:
                            raise ValueError("ForceCPU")
                    except NameError:
                        pass
                if not os.path.exists(winGPUdll):
                    raise ValueError("NoDLL")
                try:
                    self.lib = CDLL(winGPUdll, RTLD_GLOBAL)
                except WindowsError:
                    hasGPU = False
                    if os.path.exists(winNoGPUdll):
                        self.lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
                        print("Notice: CPU-only mode")
            except (KeyError, ValueError):
                hasGPU = False
                if os.path.exists(winNoGPUdll):
                    self.lib = CDLL(winNoGPUdll, RTLD_LOCAL)
                    print("Notice: CPU-only mode")
                else:
                    # Try the other way, in case no_gpu was
                    # compile but not renamed
                    self.lib = CDLL(winGPUdll, RTLD_LOCAL)
                    print(
                        "Environment variables indicated a CPU run, but we didn't find `" + winNoGPUdll + "`. Trying a GPU run anyway.")
        else:
            self.lib = CDLL(os.path.dirname(__file__) + f"/lib/unix/{self.dllname}.so",RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self. predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        if self.hasGPU:
            set_gpu = self.lib.cuda_set_device
            set_gpu.argtypes = [c_int]
            set_gpu(self.gpuid)

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int),
                                      c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self. make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.free_network_ptr = self.lib.free_network_ptr
        self.free_network_ptr.argtypes = [c_void_p]
        self.free_network_ptr.restype = c_void_p

        self. do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

    def convert_datafile(self, metaPath):
        """
        :功能   用于测试或继续训练过程中重新生成数据文件
        :输入   模型保存路径
        :输出   data/temp.data
        :说明   若出错，则界面进行提示
        """
        if not os.path.exists(metaPath):
            return
        lines = []
        new_path = metaPath.replace('.data', '_t.data')
        with open(metaPath, 'r', encoding='utf-8', errors='ignore') as fp:
            for line in fp.readlines():
                if not '.names' in line:
                    lines.append(line.strip())
                else:
                    lines.append('names=%s' % metaPath.replace('.data', '.names').strip())

        with open(new_path, 'w', encoding='utf-8', errors='ignore') as fp_write:
            for line in lines:
                fp_write.write("%s\n" % line)
        return new_path

    def detect_from_image(self, net, meta, image, thresh=0.25, hier_thresh=0.5, nms=0.5, debug=False):
        """
        :功能   从图像中检测目标
        :输入   net - 模型指针
               meta - 类别指针
               image - 图像数据
               thresh - 检测阈值
               hier_thresh - 检测阈值
               nms - IOU阈值
               debug - 用于确定是否输出调试信息
        :输出   目标检测结果列表 [(classname, prob, (x, y, w, h))]
        :说明   无
        """
        custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im, arr = array_to_image(custom_image)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(net, im)
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        if nms:
            self.do_nms_sort(dets, num, meta.classes, nms)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_detections(dets, num)
        return res

    def detect_from_file(self, net, meta, filename, thresh=0.25, hier_thresh=0.5, nms=0.5, debug=False):
        """
        :功能   从文件中检测目标
        :输入   net - 模型指针
               meta - 类别指针
               filename - 图像文件路径
               thresh - 检测阈值
               hier_thresh - 检测阈值
               nms - IOU阈值
               debug - 用于确定是否输出调试信息
        :输出   目标检测结果列表 [(classname, prob, (x, y, w, h))]
        :说明   无
        """
        im = self.load_image(filename, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(net, im)
        # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        if nms:
            self.do_nms_sort(dets, num, meta.classes, nms)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def detect(self, image, imageorfile=0, thresh=0.25):
        """
        :功能   检测目标
        :输入   image - 图像数据或图像文件路径
               imageorfile - 图像或文件标志
               thresh - 检测阈值
        :输出   目标检测结果列表 [(classname, prob, (x, y, w, h))]
        :说明   无
        """
        # 显示图像
        if imageorfile == 0:
            detections = self.detect_from_image(self.netMain, self.metaMain, image, thresh)
        elif imageorfile == 1:
            detections = self.detect_from_file(self.netMain, self.metaMain, image, thresh)
        return detections

    def draw_detect_results(self, im, detections, color=(0, 0, 255)):
        """
        :功能   绘制检测结果
        :输入   im - 图像数据
               detections - 检测结果列表
               color - 目标框线条颜色
        :输出   无
        :说明   无
        """
        for det in detections:
            class_name = det[0]  # 目标类别名称
            prob = det[1]  # 目标概率
            cx = int(det[2][0])  # 目标中心点横坐标
            cy = int(det[2][1])  # 目标中心点纵坐标
            w = int(det[2][2])  # 目标宽度
            h = int(det[2][3])  # 目标高度
            l = cx - w // 2
            t = cy - h // 2
            cv2.rectangle(im, (l, t), (l + w, t + h), color, thickness=1)
            info = '%s,%.3f' % (class_name, prob)
            t = max(10, t - 2)
            cv2.putText(im, info, (l, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 128, 0), 1)
