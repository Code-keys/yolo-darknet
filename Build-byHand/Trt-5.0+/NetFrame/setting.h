#ifndef _SETTING_H
#define _SETTING_H


// #define USE_FP16  // comment out this if want to use FP16
#define USE_INT8  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25
#define BATCH_SIZE 1

// const char* PATH_WTS = "/home/nvidia/CX/VisDrone/yolov5sm-s/weights/best.wts";
// const char* PATH_WTS = "/home/nvidia/CX/VisDrone/yolov5sm.wts";
// const char* PATH_WTS = "/home/nvidia/CX/yolov5-cfg/weights/yolov5sm-prune/yolov5sm-pr.wts";
const char* PATH_WTS = "/home/nvidia/CX/VisDrone/yolov5s.wts";

const char* PATH_CALIBRATOR_IMGS = "/home/nvidia/CX/VisDrone/test/";
const char* PATH_CALIBRATOR = "./int8.calib";

#ifdef  USE_INT8
// const char* PATH_ENGINE = "/home/nvidia/CX/VisDrone/yolov5sm-s/weights/best_int8.engine";
const char* PATH_ENGINE = "/home/nvidia/CX/VisDrone/yolov5s-int8.engine";
#elif defined(USE_FP16)
// const char* PATH_ENGINE = "/home/nvidia/CX/VisDrone/yolov5sm-s/weights/best_fp16.engine";
// const char* PATH_ENGINE = "/home/nvidia/CX/VisDrone/yolov5sm_fp16.engine";
// const char* PATH_ENGINE = "/home/nvidia/CX/yolov5-cfg/weights/yolov5sm-prune/yolov5sm-Fp16.engine";
const char* PATH_ENGINE = "/home/nvidia/CX/VisDrone/yolov5s-fp16.engine";
#else
// const char* PATH_ENGINE = "/home/nvidia/CX/VisDrone/yolov5sm-s/weights/best_fp32.engine"; 
const char* PATH_ENGINE = "/home/nvidia/CX/VisDrone/yolov5s-fp32.engine";
#endif


#endif 
