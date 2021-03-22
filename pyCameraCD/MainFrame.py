import cv2
import os
import sys
import time
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QActionGroup, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from pypylon import pylon
from model import *
from mainwindownn import Ui_MainWindow
             


class camProcess(QThread):
    sinOut = pyqtSignal(dict)

    def __init__(self, fps=30):
        super(camProcess, self).__init__()
        # global monito
        self.fps = fps
        self.fpsOut = fps
        self.switch = 0
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self.type = "ip"
        except:
            self.camera = cv2.VideoCapture(0)
            self.type = "usb"

    def run(self):
        self.switch = 1
        self.camera.Open()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        if self.camera.IsGrabbing():
            print("cap ing !")
        while self.camera.IsGrabbing():
            s = time.time()
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()[::2, ::2, ...]  # [1024, 1280, 3]

                # print(img.shape)

                self.sinOut.emit({"img": img, "fps": self.fpsOut})
                del image, img

            grabResult.Release()
            del grabResult

            re_time = 1 / self.fps - time.time() + s
            if re_time > 0:
                time.sleep(re_time)
            self.fpsOut = 1 / (time.time() - s)
            if self.switch == 0:
                self.camera.StopGrabbing()
                self.camera.Close()
                break

    @pyqtSlot()
    def close(self):
        try:
            self.switch = 0
            self.camera.StopGrabbing()
            self.camera.Close()
        except:
            print("Closing Error !")

    def __del__(self):
        try:
            self.camera.StopGrabbing()
            self.camera.Close()
            # self.Pipe.close()
        except:
            print("Closing Error !")


class f_Process(QThread):
    SigOut = pyqtSignal(dict)

    def __init__(self, parent, speed=0.03):
        super(f_Process, self).__init__()
        self.parent = parent
        self.speed = speed
        self.switch = False

    def run(self):
        self.switch = True
        if self.parent.source_type == "video":
            cap = cv2.VideoCapture(self.parent.video_path)
            while cap.isOpened():
                # s = time.time()
                ret, frame = cap.read()
                if (not ret) or (not self.parent.show_statue) or (not self.switch):
                    break
                self.SigOut.emit({"img": frame[::2,::2,::], "fps": self.parent.fps})
                # e = time.time()-s
                time.sleep(1 / self.parent.fps)
        else:
            if self.parent.source_type == "txt":
                iter = self.parent.txts
            elif self.parent.source_type == "folder":
                iter = self.parent.folders
            elif self.parent.source_type == "pic":
                iter = [self.parent.pic_path] * 60
            
            for i in iter:
                if  (not self.parent.show_statue) or (not self.switch):
                    print("stop reading !")
                    return                     
                temp1 = cv2.imread(i)
                temp1 = cv2.cvtColor(temp1, cv2.COLOR_BGR2RGB)
                time.sleep(1 / (self.parent.fps * self.speed))
                self.SigOut.emit({"img": temp1, "fps": self.parent.fps * self.speed })
            self.run()


class DetProcess(QThread):

    def __init__(self, parent):
        super(DetProcess, self).__init__()
        self.parent = parent
        if parent.detector is None:
            self.parent.detector = self.parent.ModelFactory.build("auto",("./data/yolo.engine","./data/classes.names"))
        self.parent.detect_state = True

    def run(self):
        time.sleep(1)
        print("Detect ing...")
        while 1:
            s = time.time()
            if not self.parent.detect_state:
                break
            result = self.parent.detector.draw_detect_results(self.parent.dict["img"])
            # result = self.parent.dict["img"]-100
            # print(result.shape)
            self.parent.dict["result"] = result  # self.parent.dict["img"] - 100  #### drawed picture
            self.parent.dict["D_fps"] = (1 / (time.time() - s))


class MainFrame(Ui_MainWindow, QtWidgets.QMainWindow):
    fpsSig = pyqtSignal(float)
    DetFpsSig = pyqtSignal(float)
    DetSig = pyqtSignal(dict)
    dict = dict()
    dict["result"] = np.ones([3, 512, 640]) * 255
    dict["img"] = np.ones([3, 512, 640]) * 255
    dict["D_fps"] = 0

    def __init__(self):
        super(MainFrame, self).__init__()
        self.setupUi(self)
        #  界 面 前 端 美 化
        # self.setWindowTitle("Detection")
        # window_pale = QtGui.QPalette()
        # window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("./resources/sea.jpg")))
        # self.setPalette(window_pale)
        # runtime  标志位
        self.show_statue = False
        self.detect_state = False

        # data 数据流
        self.fps = 35
        self.cursor = -1
        self.source_type = ""  # mei ju
        self.Folder_path = ""
        self.folders = []
        self.pic_path = None
        self.video_path = None
        self.videos = []
        self.txt_path = None
        self.txts = []

        # model 算法
        self.detector = None
        self.ModelFactory = ModelFactory()
        self.detectortype = "tensorrt"  # mei ju

        self.cfg_path = None
        self.weight_path = None
        self.data_path = None

        # 信号 + 槽
        self.init_event()


    def converDict(self, dict):

        temp1 = np.array(dict["img"])

        self.dict["fps"] = dict["fps"]
        self.dict["img"] = temp1

        frame = QImage(temp1, temp1.shape[1], temp1.shape[0], temp1.shape[1] * 3, QImage.Format_RGB888)
        self.label_L.setScaledContents(False)
        self.label_L.setPixmap(QPixmap.fromImage(frame))
        self.statusBar.showMessage("|     source fps: %.1f     |" % (self.dict["fps"]))
        if self.detect_state and self.show_statue:
            temp1 = np.array(self.dict["result"])
            frame = QImage(temp1, temp1.shape[1], temp1.shape[0], temp1.shape[1] * 3, QImage.Format_RGB888)
            self.label_R.setScaledContents(False)
            self.label_R.setPixmap(QPixmap.fromImage(frame))
            self.statusBar.showMessage(
                "|     source fps: %.1f     |" % (self.dict["fps"]) + "     |     detection fps: %.1f     |" % (
                    self.dict["D_fps"]))

    def Imshow(self):
        if self.source_type == "":
            self.CAPT()
        if self.source_type == "camera":
            if self.show_statue:
                return
            try:
                self.show_statue = True
                self.CamProcess = camProcess(self.fps)
                self.CamProcess.daemon = True
                self.CamProcess.sinOut.connect(self.converDict)
                # self.CamProcess.setDaemon(True)
                self.CamProcess.start()
            except Exception as e:
                print(e.__str__())
                print("camera open error !")
        else :
            if self.show_statue:
                return
            try:
                self.show_statue = True
                self.f_Process = f_Process(self, 0.028)
                self.f_Process.daemon = True
                self.f_Process.SigOut.connect(self.converDict)
                self.f_Process.start()
            except Exception as e:
                print(e.__str__())

    def stopShow(self):
        self.show_statue = False
        self.detect_state = False

        if self.source_type == "camera":
            try:
                self.CamProcess.sinOut.disconnect(self.converDict)
                self.CamProcess.switch = 0
                time.sleep(1 / self.fps)
                print("camera closing ...")
                self.CamProcess.quit()
                self.CamProcess.wait()
                del self.CamProcess
                time.sleep(0.5)
                print("CLOSED")
            except Exception as e:
                print(e.__str__())
        else:
            try:
                self.f_Process.switch = 0
                self.f_Process.SigOut.disconnect(self.converDict)
                time.sleep(1 / self.fps)
                print("file system closing ...")
                self.f_Process.quit()
                self.f_Process.wait()
                del self.f_Process
                time.sleep(0.5)
                print("file system closed")
            except Exception as e:
                print(e.__str__())

    def Det(self):
        if (self.detect_state) or (not self.show_statue):
            return
        self.detect_state = True

        self.DetThread = DetProcess(self)
        self.DetThread.setObjectName("")
        self.DetThread.start()

    def stopDet(self):
        try:
            self.detect_state = False
            time.sleep(0.1)
            self.DetThread.quit()
            self.DetThread.wait()
            del self.DetThread
            try:
                self.detector.destory()
            except:
                del self.detector
            # self.DetThread.join()
            # self.DetThread.destroyed()
            # self.DetThread.terminate()
            print("det stoped !")
        except Exception as e:
            print(e.__str__())

    def RESET(self):
        self.stopDet()
        self.stopShow()

    def PIC(self):
        self.pic_path,_ = QFileDialog.getOpenFileName(self, "choose picture file (*.jpg)", directory="./",
                                                    filter="All Files (*)")
        self.source_type = "pic"
        self.statusBar.showMessage(f"selected pic type including 1 files:{self.pic_path}")
        self.cursor = -1
        self.action_pic.setChecked(True)

    def VIDEO(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "choose video file (*.mp4)", directory="./",
                                                         filter="All Files (*)")
        self.source_type = "video"
        self.action_video.setChecked(True)

    def FOLDER(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "choose a path  (./)", directory="./")
        print(self.folder_path)
        self.folders = [os.path.join(self.folder_path,i) for i in os.listdir(self.folder_path) if
                        i.endswith(".jpg") or i.endswith(".jpeg") or i.endswith(".png") or i.endswith("bmp")]
        if self.folders.__len__() == 0:
            btn = QMessageBox.warning(self, "warning", self.tr("当前文件及未找到标准格式图像，重设还是退出 ？"),
                                      QMessageBox.Reset | QMessageBox.Cancel, QMessageBox.Reset)
            if btn == QMessageBox.Reset:
                self.FOLDER()
            else:
                return
        self.source_type = "folder"
        self.statusBar.showMessage(f"selected folder type: including {self.folders.__len__() } files")
        self.action_folder.setChecked(True)

    def CAPT(self):
        self.source_type = "camera"
        self.statusBar.showMessage(f"selected camera type: ip ")
        self.action_camera.setChecked(True)

    def TXT(self):
        self.txt_path = QFileDialog.getOpenFileName(self, "choose txt contained the path of pics  (*.txt)",
                                                         directory="./",
                                                         filter="All Files (*)")[0]
        self.txts = [i for i in open(self.txt_path,"r").read().strip().split("\n")]
        self.source_type = "txt"
        self.statusBar.showMessage(f"selected txt type: including {self.txts.__len__() } files")
        self.action_folder.setChecked(True)
        self.cursor = -1

    def DARKNET(self):
        self.cfg_path = QFileDialog.getOpenFileName(self, "choose cfg file (*.cfg)", directory="./data",
                                                    filter="All Files (*);;cfg model Files (*.cfg)")[0]
        time.sleep(0.45)
        self.weight_path = QFileDialog.getOpenFileName(self, "choose weight file (*.weights/*.*)",
                                                       directory="./data",
                                                       filter="All Files (*);;weights Files (*.weights)")[0]
        time.sleep(0.45)
        self.data_path = QFileDialog.getOpenFileName(self, "choose txt file (*.data/*.txt/*.*)", directory="./data",
                                                     filter="All Files (*);;data Files (*.txt)")[0]
        self.detectortype = "darknet"
        self.detector = self.ModelFactory.build("darknet", (self.cfg_path, self.weight_path, self.data_path))

    def RTV5(self):

        # self.cfg_path = QFileDialog.getOpenFileName(self, "choose cfg file (*.cfg)", directory="./",
        #                                             filter="All Files (*);;cfg Files (*.cfg)")
        # time.sleep(0.35)
        self.weight_path = QFileDialog.getOpenFileName(self, "choose weight file (*.engine)", directory="./data",
                                                       filter="All Files (*);; engine Files (*.engine)")[0]
        # time.sleep(0.35)
        self.data_path = QFileDialog.getOpenFileName(self, "choose class data file : classes.names(*.cfg/*.yaml/*.py/*.*)", directory="./data",
                                                     filter="All Files (*);;txt Files (*.txt);;yaml Files (*.yaml)")[0]
        self.detectortype = "tensorrt"
        self.action_yolov5_rt.setChecked(True)
        self.detector = self.ModelFactory.build("tensorrt", (self.weight_path, self.data_path))
    def RTV4(self):
        pass
    def RTV3(self):
        pass

    def OTHER(self):
        self.weight_path = QFileDialog.getOpenFileName(self, "choose weight file (*.engine)", directory="./data",
                                                       filter="All Files (*);; engine Files (*.engine)")[0]
        time.sleep(0.35)
        self.data_path = QFileDialog.getOpenFileName(self, "choose class data file : classes.names(*.cfg/*.yaml/*.py/*.*)", directory="./data",
                                                     filter="All Files (*);;txt Files (*.txt);;yaml Files (*.yaml)")[0]

        self.detectortype = "auto"
        self.action_other.setChecked(True)
        self.detector = self.ModelFactory.build("auto", (self.weight_path, self.data_path))

    def information(self):
        QMessageBox.about(self, "copy@right", "版权所有:杭州电子科技大学 \nEmail:hdu@hdu.edu.cn ()")

    def init_event(self):

        self.BN_1.clicked.connect(self.stopShow)
        self.BN_2.clicked.connect(self.Imshow)
        self.BN_3.clicked.connect(self.Det)
        self.BN_4.clicked.connect(self.stopDet)
        self.action_Reset.triggered.connect(self.RESET)
        self.actioncopr_right.triggered.connect(self.information)

        self.action_pic.triggered.connect(self.PIC)
        self.action_video.triggered.connect(self.VIDEO)
        self.action_txt.triggered.connect(self.TXT)
        self.action_folder.triggered.connect(self.FOLDER)
        self.action_camera.triggered.connect(self.CAPT)

        self.actGroupS = QActionGroup(self)
        self.actGroupS.addAction(self.action_pic)
        self.actGroupS.addAction(self.action_video)
        self.actGroupS.addAction(self.action_txt)
        self.actGroupS.addAction(self.action_folder)
        self.actGroupS.addAction(self.action_camera)

        self.action_darknet.triggered.connect(self.DARKNET)
        self.action_yolov3_rt.triggered.connect(self.RTV3)
        self.action_yolov4_rt.triggered.connect(self.RTV4)
        self.action_yolov5_rt.triggered.connect(self.RTV5)
        self.action_other.triggered.connect(self.OTHER)

        self.actGroup = QActionGroup(self)
        self.actGroup.addAction(self.action_darknet)
        self.actGroup.addAction(self.action_yolov3_rt)
        self.actGroup.addAction(self.action_yolov4_rt)
        self.actGroup.addAction(self.action_yolov5_rt)
        self.actGroup.addAction(self.action_other)

        self.showMaximized()
        
