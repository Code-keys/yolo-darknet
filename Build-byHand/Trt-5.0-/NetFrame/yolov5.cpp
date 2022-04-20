#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "plugin_factory.h"
#include "yololayer.h"
#include "mish.h"
#include <opencv2/opencv.hpp>
#include <dirent.h>



/*************** set the typeof YOLOv5 ****************/ 
#define YOLOv5_cfg_ratio  1  //s :1 , m: 2//3 , l : 2, x : 5//4 
//(YOLOv5_cfg_ratio 不可加括号)  and Not Forget modify the layername before the YOLO_Layer
static char* PATH_WTS ="/home/huanyu/DetAPI/models/yolov5s.wts";
static char* PTH_ENGINE = "/home/huanyu/DetAPI/models/yolov5s.engine";

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5
#define BATCH_SIZE 1

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h* img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Yolo::Detection& a, Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = NMS_THRESH) {
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + DETECTION_SIZE * i + 4] <= BBOX_CONF_THRESH) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnMish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
    std::cout << linx << std::endl;
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{s, s});
    conv1->setPadding(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    auto mish = new MishPlugin();
    ITensor* inputTensors[] = {bn1->getOutput(0)};
    auto mish_ = network->addPlugin(inputTensors, 1, *mish);
    assert(mish_);
    mish_->setName(("mish" + std::to_string(linx)).c_str());
    return mish_;
}

ILayer* convBnSilu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
    std::cout << linx << std::endl;
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{s, s});
    conv1->setPadding(DimsHW{p, p});

    std::string cvLayerName =  "module_list." + std::to_string(linx) + ".conv2d" ;
    conv1->setName(cvLayerName.c_str()); 

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    nvinfer1::ITensor* bnOutput = bn1->getOutput(0);
    nvinfer1::IActivationLayer* sig = network->addActivation(
                *bnOutput, nvinfer1::ActivationType::kSIGMOID);
    assert(sig != nullptr);
    auto silu = network->addElementWise( *bnOutput, *sig->getOutput(0), nvinfer1::ElementWiseOperation::kPROD );
    assert(silu != nullptr);
    std::string sigLayerName =  "module_list." + std::to_string(linx) + ".silu" ;
    silu->setName(sigLayerName.c_str());
    return silu;
}


ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
    std::cout << linx << std::endl;
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{s, s});
    conv1->setPadding(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    ITensor* inputTensors[] = {bn1->getOutput(0)};
    auto lr = plugin::createPReLUPlugin(0.1);
    auto lr1 = network->addPlugin(inputTensors, 1, *lr);
    assert(lr1);
    lr1->setName(("leaky" + std::to_string(linx)).c_str());
    return lr1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, DataType dt) {
    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape { 1, 1, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights( PATH_WTS );
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // define each layer.
    
    auto l0 = convBnSilu(network, weightMap, *data, 12 * YOLOv5_cfg_ratio , 3, 2, 1, 0);
    auto l1 = convBnSilu(network, weightMap, *l0->getOutput(0), 32 * YOLOv5_cfg_ratio , 3, 1, 1, 1);

    auto l2 = convBnSilu(network, weightMap, *l1->getOutput(0), 64 * YOLOv5_cfg_ratio, 3, 2, 1, 2);
    auto l3 = convBnSilu(network, weightMap, *l2->getOutput(0), 32 * YOLOv5_cfg_ratio, 1, 1, 0, 3);  
    auto l4 = l2;
    auto l5 = convBnSilu(network, weightMap, *l2->getOutput(0), 32 * YOLOv5_cfg_ratio, 1, 1, 0, 5);
    auto l6 = convBnSilu(network, weightMap, *l5->getOutput(0), 32 * YOLOv5_cfg_ratio, 1, 1, 0, 6);
    auto l7 = convBnSilu(network, weightMap, *l6->getOutput(0), 32 * YOLOv5_cfg_ratio, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*l7->getOutput(0), *l5->getOutput(0), ElementWiseOperation::kSUM);
    ITensor* inputTensors9[] = {ew8->getOutput(0), l3->getOutput(0)};
    auto cat9 = network->addConcatenation(inputTensors9, 2);    
    auto l10 = convBnSilu(network, weightMap, *cat9->getOutput(0), 64 * YOLOv5_cfg_ratio, 1, 1, 0, 10);

    printf("DBUG %d\n\n", __LINE__ );

    auto l11 = convBnSilu(network, weightMap, *l10->getOutput(0) , 128 * YOLOv5_cfg_ratio, 3, 2, 1, 11); 
    auto l12 = convBnSilu(network, weightMap, *l11->getOutput(0) , 64 * YOLOv5_cfg_ratio, 1, 1, 0, 12);
    auto l13 = l11; 
    auto l14 = convBnSilu(network, weightMap, *l13->getOutput(0) , 64 * YOLOv5_cfg_ratio, 1, 1, 0, 14);
    auto l15 = convBnSilu(network, weightMap, *l14->getOutput(0) , 64 * YOLOv5_cfg_ratio, 1, 1, 0, 15);
    auto l16 = convBnSilu(network, weightMap, *l15->getOutput(0) , 64 * YOLOv5_cfg_ratio, 3, 1, 1, 16);
    auto ew17 = network->addElementWise(*l16->getOutput(0), *l14->getOutput(0), ElementWiseOperation::kSUM);
    auto l18 = convBnSilu(network, weightMap, *ew17->getOutput(0) , 64 , 1, 1, 0, 18);
    auto l19 = convBnSilu(network, weightMap, *l18->getOutput(0) , 64 , 3, 1, 1, 19);
    auto ew20 = network->addElementWise(*l19->getOutput(0), *ew17->getOutput(0), ElementWiseOperation::kSUM);
    auto l21 = convBnSilu(network, weightMap, *ew20->getOutput(0) , 64 * YOLOv5_cfg_ratio, 1, 1, 0, 21);
    auto l22 = convBnSilu(network, weightMap, *l21->getOutput(0) , 64 * YOLOv5_cfg_ratio, 3, 1, 1, 22);
    auto ew23 = network->addElementWise(*l22->getOutput(0), *ew20->getOutput(0), ElementWiseOperation::kSUM);
    ITensor* inputTensors22[] = {ew23->getOutput(0), l12->getOutput(0)};
    auto cat24 = network->addConcatenation(inputTensors22, 2);    
    auto l25 = convBnSilu(network, weightMap, *cat24->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 25);
 
    printf("DBUG %d\n\n", __LINE__ );
    
    auto l26 = convBnSilu(network, weightMap, *l25->getOutput(0), 256 * YOLOv5_cfg_ratio, 3, 2, 1, 26);
    auto l27 = convBnSilu(network, weightMap, *l26->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 27);
    auto l28 = l26;
    auto l29 = convBnSilu(network, weightMap, *l28->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 29);
    auto l30 = convBnSilu(network, weightMap, *l29->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 30);
    auto l31 = convBnSilu(network, weightMap, *l30->getOutput(0) , 128 * YOLOv5_cfg_ratio, 3, 1, 1, 31);
    auto ew32 = network->addElementWise(*l31->getOutput(0), *l29->getOutput(0), ElementWiseOperation::kSUM);
    auto l33 = convBnSilu(network, weightMap, *ew32->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 33);
    auto l34 = convBnSilu(network, weightMap, *l33->getOutput(0) , 128 * YOLOv5_cfg_ratio, 3, 1, 1, 34);
    auto ew35 = network->addElementWise(*l34->getOutput(0), *ew32->getOutput(0), ElementWiseOperation::kSUM);
    auto l36 = convBnSilu(network, weightMap, *ew35->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 36);
    auto l37 = convBnSilu(network, weightMap, *l36->getOutput(0) , 128 * YOLOv5_cfg_ratio, 3, 1, 1, 37);
    auto ew38 = network->addElementWise(*l37->getOutput(0), *ew35->getOutput(0), ElementWiseOperation::kSUM);
    ITensor* inputTensors38[] = {ew38->getOutput(0), l27->getOutput(0)};
    auto cat39 = network->addConcatenation(inputTensors38, 2);
    auto l40 = convBnSilu(network, weightMap, *cat39->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 40);
  
    printf("DBUG %d\n\n", __LINE__ );

    auto l41 = convBnSilu(network, weightMap, *l40->getOutput(0), 512 * YOLOv5_cfg_ratio, 3, 2, 1, 41); 
    auto l42 = convBnSilu(network, weightMap, *l41->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 42);
    auto pool43 = network->addPooling(*l42->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
    pool43->setPadding(DimsHW{2, 2});
    pool43->setStride(DimsHW{1, 1});
    auto l44 = l42; 
    auto pool45 = network->addPooling(*l42->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
    pool45->setPadding(DimsHW{4, 4});
    pool45->setStride(DimsHW{1, 1}); 
    auto l46 = l42; 
    auto pool47 = network->addPooling(*l42->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
    pool47->setPadding(DimsHW{6, 6});
    pool47->setStride(DimsHW{1, 1}); 
    ITensor* inputTensors48[] = {pool47->getOutput(0), pool45->getOutput(0), pool43->getOutput(0), l42->getOutput(0)};
    auto cat48 = network->addConcatenation(inputTensors48, 4);
    auto l49 = convBnSilu(network, weightMap, *cat48->getOutput(0), 512 * YOLOv5_cfg_ratio, 1, 1, 0, 49);
  
    auto l50 = convBnSilu(network, weightMap, *l49->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 50);
    auto l51 = l49;
    auto l52 = convBnSilu(network, weightMap, *l51->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 52);
    auto l53 = convBnSilu(network, weightMap, *l52->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 53);
    auto l54 = convBnSilu(network, weightMap, *l53->getOutput(0), 256 * YOLOv5_cfg_ratio, 3, 1, 1, 54);
    ITensor* inputTensorsl55[] = {l54->getOutput(0), l50->getOutput(0)};
    auto cat55 = network->addConcatenation(inputTensorsl55, 2);
    auto l56 = convBnSilu(network, weightMap, *cat55->getOutput(0), 512 * YOLOv5_cfg_ratio, 1, 1, 0, 56); 
    auto l57 = convBnSilu(network, weightMap, *l56->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 57); 
    
    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256  * 2 * 2 * YOLOv5_cfg_ratio ));
    for (int i = 0; i < 256  * 2 * 2 * YOLOv5_cfg_ratio ; i++) {   deval[i] = 1.0;  }
    Weights deconvwts58{DataType::kFLOAT, deval, 256  * 2 * 2 * YOLOv5_cfg_ratio };
    IDeconvolutionLayer* deconv58 = network->addDeconvolution(*l57->getOutput(0), 256 * YOLOv5_cfg_ratio , DimsHW{2, 2}, deconvwts58, emptywts);
    assert(deconv58);
    deconv58->setStride(DimsHW{2, 2});
    deconv58->setNbGroups(256 * YOLOv5_cfg_ratio );
    ITensor* inputTensors59[] = {l40->getOutput(0), deconv58->getOutput(0)};
    auto cat59 = network->addConcatenation(inputTensors59, 2);
    auto l60 = convBnSilu(network, weightMap, *cat59->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 60);
    auto l61 = cat59;
    auto l62 = convBnSilu(network, weightMap, *l61->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 62);
    auto l63 = convBnSilu(network, weightMap, *l62->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 63);
    auto l64 = convBnSilu(network, weightMap, *l63->getOutput(0) , 128 * YOLOv5_cfg_ratio, 3, 1, 1, 64);
    ITensor* inputTensors65[] = {l64->getOutput(0), l60->getOutput(0)};
    auto cat65 = network->addConcatenation(inputTensors65, 2);
    auto l66 = convBnSilu(network, weightMap, *cat65->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 66); 
    auto l67 = convBnSilu(network, weightMap, *l66->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 67); 
    
    printf("DBUG %d\n\n", __LINE__ );
    
    Weights deconvwts68{DataType::kFLOAT, deval, 128  * 2 * 2 * YOLOv5_cfg_ratio };
    IDeconvolutionLayer* deconv68 = network->addDeconvolution(*l67->getOutput(0) , 128 * YOLOv5_cfg_ratio, DimsHW{2, 2}, deconvwts68, emptywts);
    assert(deconv68);
    deconv68->setStride(DimsHW{2, 2});
    deconv68->setNbGroups(128 * YOLOv5_cfg_ratio);
    ITensor* inputTensors69[] = {l25->getOutput(0), deconv68->getOutput(0)};
    auto cat69 = network->addConcatenation(inputTensors69, 2);
    auto l70 = convBnSilu(network, weightMap, *cat69->getOutput(0) , 64 * YOLOv5_cfg_ratio, 1, 1, 0, 70);
    auto l71 = cat69;
    auto l72 = convBnSilu(network, weightMap, *l71->getOutput(0) , 64 * YOLOv5_cfg_ratio, 1, 1, 0, 72);
    auto l73 = convBnSilu(network, weightMap, *l72->getOutput(0) , 64 * YOLOv5_cfg_ratio, 1, 1, 0, 73);
    auto l74 = convBnSilu(network, weightMap, *l73->getOutput(0) , 64 * YOLOv5_cfg_ratio, 3, 1, 1, 74);
    ITensor* inputTensors75[] = {l74->getOutput(0), l70->getOutput(0)};
    auto cat75 = network->addConcatenation(inputTensors75, 2);
    auto l76 = convBnSilu(network, weightMap, *cat75->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 76);
    auto convl77 = network->addConvolution(*l76->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.77.Conv2d.weight"], weightMap["module_list.77.Conv2d.bias"]);
    assert(convl77); 
    // 78 is yolo layer

    printf("DBUG %d\n\n", __LINE__ );

    auto l79 = l76;
    auto l80 = convBnSilu(network, weightMap, *l79->getOutput(0) , 128 * YOLOv5_cfg_ratio, 3, 2, 1, 80);
    
    ITensor* inputTensors181[] = {l80->getOutput(0), l67->getOutput(0)};
    auto cat81 = network->addConcatenation(inputTensors181, 2);
    auto l82 = convBnSilu(network, weightMap, *cat81->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 82);
    auto l83 = cat81;
    auto l84 = convBnSilu(network, weightMap, *l83->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 84);
    auto l85 = convBnSilu(network, weightMap, *l84->getOutput(0) , 128 * YOLOv5_cfg_ratio, 1, 1, 0, 85);
    auto l86 = convBnSilu(network, weightMap, *l85->getOutput(0) , 128 * YOLOv5_cfg_ratio, 3, 1, 1, 86);
    ITensor* inputTensors87[] = {l86->getOutput(0), l84->getOutput(0)};
    auto cat87 = network->addConcatenation(inputTensors87, 2);
    auto l88 = convBnSilu(network, weightMap, *cat87->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 88);  
    IConvolutionLayer* convl89 = network->addConvolution(*l88->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.89.Conv2d.weight"], weightMap["module_list.89.Conv2d.bias"]);
    assert(convl89);
    
    printf("DBUG %d\n\n", __LINE__ );
    // 90 is yolo Layer

    auto l91 = l88;
    auto l92 = convBnSilu(network, weightMap, *l91->getOutput(0), 256 * YOLOv5_cfg_ratio, 3, 2, 1, 92);
    
    ITensor* inputTensors93[] = {l92->getOutput(0), l57->getOutput(0)};
    auto cat93 = network->addConcatenation(inputTensors93, 2);
    auto l94 = convBnSilu(network, weightMap, *cat93->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 94);
    auto l95 = cat93; 
    auto l96 = convBnSilu(network, weightMap, *l95->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 96);
    auto l97 = convBnSilu(network, weightMap, *l96->getOutput(0), 256 * YOLOv5_cfg_ratio, 1, 1, 0, 97);
    auto l98 = convBnSilu(network, weightMap, *l97->getOutput(0), 256 * YOLOv5_cfg_ratio, 3, 1, 1, 98);
    ITensor* inputTensors99[] = {l98->getOutput(0), l94->getOutput(0)};
    auto cat99 = network->addConcatenation(inputTensors99, 2);
    auto l100 = convBnSilu(network, weightMap, *cat99->getOutput(0), 512 * YOLOv5_cfg_ratio, 1, 1, 0, 100);  
    IConvolutionLayer* convl101 = network->addConvolution(*l100->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.101.Conv2d.weight"], weightMap["module_list.101.Conv2d.bias"]);
    assert(convl101);
    
    
    printf("DBUG %d\n\n", __LINE__ );
    // 102 is yolo layer



    auto yolo = new YoloLayerPlugin();
    ITensor* inputTensors_yolo[] = {convl77->getOutput(0), convl89->getOutput(0), convl101->getOutput(0)};
    auto yolo_ = network->addPlugin(inputTensors_yolo, 3, *yolo);
    assert(yolo_);
    yolo_->setName("yolo_");

    yolo_->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*yolo_->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
#ifdef USE_FP16
    builder->setFp16Mode(true);
#endif
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p( PTH_ENGINE );
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file( PTH_ENGINE , std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    PluginFactory pf;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, &pf);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    int fcount = 0;
    for (int f = 0; f < file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - BATCH_SIZE + 1 + b]);
            
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img);
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE]);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - BATCH_SIZE + 1 + b]);
            for (size_t j = 0; j < res.size(); j++) {
                float *p = (float*)&res[j];
                for (size_t k = 0; k < 7; k++) {
                //    std::cout << p[k] << ", ";
                }
                //std::cout << std::endl;
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("_" + file_names[f - BATCH_SIZE + 1 + b], img);
        }
        fcount = 0;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    //Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
