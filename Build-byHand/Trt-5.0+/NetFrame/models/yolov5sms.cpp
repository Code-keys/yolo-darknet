#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "utils.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yololayer.h"
#include "mish.h"
#include "calibrator.h"

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


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
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
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

ILayer* convBnMish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
ITensor& input, int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(linx)).c_str(), pluginData);
    ITensor* inputTensors[] = {bn1->getOutput(0)};
    auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
    return mish;
}

ILayer* convBnSilu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
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
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt ) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights( PATH_WTS );
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // define each layer.
    auto l0 = convBnSilu(network, weightMap, *data, 12 , 3, 2, 1, 0);
    auto l1 = convBnSilu(network, weightMap, *l0->getOutput(0), 48 / 3 * 2, 3, 1, 1, 1);

    auto l2 = convBnSilu(network, weightMap, *l1->getOutput(0), 96 / 3 * 2, 3, 2, 1, 2);
    auto l3 = convBnSilu(network, weightMap, *l2->getOutput(0), 96 / 3 * 2, 1, 1, 0, 3);
    auto l4 = convBnSilu(network, weightMap, *l3->getOutput(0), 96 / 3 * 2, 3, 1, 1, 4);
    auto ew5 = network->addElementWise(*l4->getOutput(0), *l2->getOutput(0), ElementWiseOperation::kSUM);
    auto l6 = convBnSilu(network, weightMap, *ew5->getOutput(0), 96 / 3 * 2, 1, 1, 0, 6);
    auto l7 = convBnSilu(network, weightMap, *l6->getOutput(0), 96 / 3 * 2, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*l7->getOutput(0), *ew5->getOutput(0), ElementWiseOperation::kSUM);
  

    auto l9 = convBnSilu(network, weightMap, *ew8->getOutput(0), 192 / 3 * 2, 3, 2, 1, 9); 
    auto l10 = convBnSilu(network, weightMap, *l9->getOutput(0), 96 / 3 * 2, 1, 1, 0, 10);
    auto l11 = l9; 
    auto l12 = convBnSilu(network, weightMap, *l11->getOutput(0), 96 / 3 * 2, 1, 1, 0, 12);
    auto l13 = convBnSilu(network, weightMap, *l12->getOutput(0), 96 / 3 * 2, 1, 1, 0, 13);
    auto l14 = convBnSilu(network, weightMap, *l13->getOutput(0), 96 / 3 * 2, 3, 1, 1, 14);
    auto ew15 = network->addElementWise(*l14->getOutput(0), *l12->getOutput(0), ElementWiseOperation::kSUM);
    auto l16 = convBnSilu(network, weightMap, *ew15->getOutput(0), 96 / 3 * 2, 1, 1, 0, 16);
    auto l17 = convBnSilu(network, weightMap, *l16->getOutput(0), 96 / 3 * 2, 3, 1, 1, 17);
    auto ew18 = network->addElementWise(*l17->getOutput(0), *ew15->getOutput(0), ElementWiseOperation::kSUM);
    auto l19 = convBnSilu(network, weightMap, *ew18->getOutput(0), 96 / 3 * 2, 1, 1, 0, 19);
    auto l20 = convBnSilu(network, weightMap, *l19->getOutput(0), 96 / 3 * 2, 3, 1, 1, 20);
    auto ew21 = network->addElementWise(*l20->getOutput(0), *ew18->getOutput(0), ElementWiseOperation::kSUM);
    ITensor* inputTensors22[] = {ew21->getOutput(0), l10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);    
    auto l23 = convBnSilu(network, weightMap, *cat22->getOutput(0), 192 / 3 * 2, 1, 1, 0, 23);

  
    auto l24 = convBnSilu(network, weightMap, *l23->getOutput(0), 384 / 3 * 2, 3, 2, 1, 24);
    auto l25 = convBnSilu(network, weightMap, *l24->getOutput(0), 192 / 3 * 2, 1, 1, 0, 25);
    auto l26 = l24;
    auto l27 = convBnSilu(network, weightMap, *l24->getOutput(0), 192 / 3 * 2, 1, 1, 0, 27);
    auto l28 = convBnSilu(network, weightMap, *l27->getOutput(0), 192 / 3 * 2, 1, 1, 0, 28);
    auto l29 = convBnSilu(network, weightMap, *l28->getOutput(0), 192 / 3 * 2, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*l29->getOutput(0), *l27->getOutput(0), ElementWiseOperation::kSUM);
    auto l31 = convBnSilu(network, weightMap, *ew30->getOutput(0), 192 / 3 * 2, 1, 1, 0, 31);
    auto l32 = convBnSilu(network, weightMap, *l31->getOutput(0), 192 / 3 * 2, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*l32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto l34 = convBnSilu(network, weightMap, *ew33->getOutput(0), 192 / 3 * 2, 1, 1, 0, 34);
    auto l35 = convBnSilu(network, weightMap, *l34->getOutput(0), 192 / 3 * 2, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*l35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    ITensor* inputTensors38[] = {ew36->getOutput(0), l25->getOutput(0)};
    auto cat37 = network->addConcatenation(inputTensors38, 2);
    auto l38 = convBnSilu(network, weightMap, *cat37->getOutput(0), 384 / 3 * 2, 1, 1, 0, 38);


    auto l39 = convBnSilu(network, weightMap, *l38->getOutput(0), 768 / 3 * 2, 3, 2, 1, 39);

    auto l40 = convBnSilu(network, weightMap, *l39->getOutput(0), 384 / 3 * 2, 1, 1, 0, 40);
    auto pool41 = network->addPoolingNd(*l40->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
    pool41->setPaddingNd(DimsHW{2, 2});
    pool41->setStrideNd(DimsHW{1, 1}); 
    auto l42 = l40; 
    auto pool43 = network->addPoolingNd(*l40->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
    pool43->setPaddingNd(DimsHW{4, 4});
    pool43->setStrideNd(DimsHW{1, 1}); 
    auto l44 = l42; 
    auto pool45 = network->addPoolingNd(*l40->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
    pool45->setPaddingNd(DimsHW{6, 6});
    pool45->setStrideNd(DimsHW{1, 1}); 
    ITensor* inputTensors46[] = {pool45->getOutput(0), pool43->getOutput(0), pool41->getOutput(0), l40->getOutput(0)};
    auto cat46 = network->addConcatenation(inputTensors46, 4);
    auto l47 = convBnSilu(network, weightMap, *cat46->getOutput(0), 768 / 3 * 2, 1, 1, 0, 47);

    auto l48 = convBnSilu(network, weightMap, *l47->getOutput(0), 384 / 3 * 2, 1, 1, 0, 48);
    auto l49 = l47;
    auto l50 = convBnSilu(network, weightMap, *l47->getOutput(0), 384 / 3 * 2, 1, 1, 0, 50);
    auto l51 = convBnSilu(network, weightMap, *l50->getOutput(0), 384 / 3 * 2, 1, 1, 0, 51);
    auto l52 = convBnSilu(network, weightMap, *l51->getOutput(0), 384 / 3 * 2, 3, 1, 1, 52);
    ITensor* inputTensorsl53[] = {l52->getOutput(0), l48->getOutput(0)};
    auto cat53 = network->addConcatenation(inputTensorsl53, 2);
    auto l54 = convBnSilu(network, weightMap, *cat53->getOutput(0), 768 / 3 * 2, 1, 1, 0, 54);

    auto l55 = convBnSilu(network, weightMap, *l54->getOutput(0), 384 / 3 * 2, 1, 1, 0, 55);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 384 / 3 * 2 * 2 * 2));
    for (int i = 0; i < 384 / 3 * 2 * 2 * 2; i++) {   deval[i] = 1.0;  }
    Weights deconvwts56{DataType::kFLOAT, deval, 384 / 3 * 2 * 2 * 2};
    IDeconvolutionLayer* deconv56 = network->addDeconvolutionNd(*l55->getOutput(0), 384 / 3 * 2, DimsHW{2, 2}, deconvwts56, emptywts);
    assert(deconv56);
    deconv56->setStrideNd(DimsHW{2, 2});
    deconv56->setNbGroups(384 / 3 * 2);
    ITensor* inputTensors57[] = {l38->getOutput(0), deconv56->getOutput(0)};
    auto cat57 = network->addConcatenation(inputTensors57, 2);
    auto l58 = convBnSilu(network, weightMap, *cat57->getOutput(0), 192 / 3 * 2, 1, 1, 0, 58);
    auto l59 = cat57;
    auto l60 = convBnSilu(network, weightMap, *l59->getOutput(0), 192 / 3 * 2, 1, 1, 0, 60);
    auto l61 = convBnSilu(network, weightMap, *l60->getOutput(0), 192 / 3 * 2, 1, 1, 0, 61);
    auto l62 = convBnSilu(network, weightMap, *l61->getOutput(0), 192 / 3 * 2, 3, 1, 1, 62);
    ITensor* inputTensors63[] = {l62->getOutput(0), l58->getOutput(0)};
    auto cat63 = network->addConcatenation(inputTensors63, 2);
    auto l64 = convBnSilu(network, weightMap, *cat63->getOutput(0), 384 / 3 * 2, 1, 1, 0, 64);

    auto l65 = convBnSilu(network, weightMap, *l64->getOutput(0), 192 / 3 * 2, 1, 1, 0, 65);
    
    Weights deconvwts66{DataType::kFLOAT, deval, 192 / 3 * 2 * 2 * 2};
    IDeconvolutionLayer* deconv66 = network->addDeconvolutionNd(*l65->getOutput(0), 192 / 3 * 2, DimsHW{2, 2}, deconvwts66, emptywts);
    assert(deconv66);
    deconv66->setStrideNd(DimsHW{2, 2});
    deconv66->setNbGroups(192 / 3 * 2);
    ITensor* inputTensors66[] = {l23->getOutput(0), deconv66->getOutput(0)};
    auto cat67 = network->addConcatenation(inputTensors66, 2);
    auto l68 = convBnSilu(network, weightMap, *cat67->getOutput(0), 96 / 3 * 2, 1, 1, 0, 68);
    auto l69 = cat67;
    auto l70 = convBnSilu(network, weightMap, *l69->getOutput(0), 96 / 3 * 2, 1, 1, 0, 70);
    auto l71 = convBnSilu(network, weightMap, *l70->getOutput(0), 96 / 3 * 2, 1, 1, 0, 71);
    auto l72 = convBnSilu(network, weightMap, *l71->getOutput(0), 96 / 3 * 2, 3, 1, 1, 72);
    ITensor* inputTensors73[] = {l72->getOutput(0), l68->getOutput(0)};
    auto cat73 = network->addConcatenation(inputTensors73, 2);
    auto l74 = convBnSilu(network, weightMap, *cat73->getOutput(0), 192 / 3 * 2, 1, 1, 0, 74);
    auto convl75 = network->addConvolutionNd(*l74->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.75.Conv2d.weight"], weightMap["module_list.75.Conv2d.bias"]);
    assert(convl75); 
    // 76 is yolo layer


    auto l77 = l74;
    auto l78 = convBnSilu(network, weightMap, *l77->getOutput(0), 192 / 3 * 2, 3, 2, 1, 78);

    ITensor* inputTensors179[] = {l78->getOutput(0), l65->getOutput(0)};
    auto cat79 = network->addConcatenation(inputTensors179, 2);
    auto l80 = convBnSilu(network, weightMap, *cat79->getOutput(0), 192 / 3 * 2, 1, 1, 0, 80);
    auto l81 = cat79;
    auto l82 = convBnSilu(network, weightMap, *l81->getOutput(0), 192 / 3 * 2, 1, 1, 0, 82);
    auto l83 = convBnSilu(network, weightMap, *l82->getOutput(0), 192 / 3 * 2, 1, 1, 0, 83);
    auto l84 = convBnSilu(network, weightMap, *l83->getOutput(0), 192 / 3 * 2, 3, 1, 1, 84);
    ITensor* inputTensors85[] = {l84->getOutput(0), l80->getOutput(0)};
    auto cat85 = network->addConcatenation(inputTensors85, 2);
    auto l86 = convBnSilu(network, weightMap, *cat85->getOutput(0), 384 / 3 * 2, 1, 1, 0, 86);  
    IConvolutionLayer* convl87 = network->addConvolutionNd(*l86->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.87.Conv2d.weight"], weightMap["module_list.87.Conv2d.bias"]);
    assert(convl87);

    // 88 is yolo Layer

    auto l89 = l86;
    auto l90 = convBnSilu(network, weightMap, *l89->getOutput(0), 384 / 3 * 2, 3, 2, 1, 90);

    ITensor* inputTensors91[] = {l90->getOutput(0), l55->getOutput(0)};
    auto cat91 = network->addConcatenation(inputTensors91, 2);
    auto l92 = convBnSilu(network, weightMap, *cat91->getOutput(0), 384 / 3 * 2, 1, 1, 0, 92);
    auto l93 = cat91;
    auto l94 = convBnSilu(network, weightMap, *l93->getOutput(0), 384 / 3 * 2, 1, 1, 0, 94);
    auto l95 = convBnSilu(network, weightMap, *l94->getOutput(0), 384 / 3 * 2, 1, 1, 0, 95);
    auto l96 = convBnSilu(network, weightMap, *l95->getOutput(0), 384 / 3 * 2, 3, 1, 1, 96);
    ITensor* inputTensors97[] = {l96->getOutput(0), l92->getOutput(0)};
    auto cat97 = network->addConcatenation(inputTensors97, 2);
    auto l98 = convBnSilu(network, weightMap, *cat97->getOutput(0), 768 / 3 * 2, 1, 1, 0, 98);  
    IConvolutionLayer* convl99 = network->addConvolutionNd(*l98->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.99.Conv2d.weight"], weightMap["module_list.99.Conv2d.bias"]);
    assert(convl99);
    // 99 is yolo layer

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {convl75->getOutput(0), convl87->getOutput(0), convl99->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
    std::cout << "Building tensorrt engine by FP16, please wait for a while..." << std::endl;
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, PATH_CALIBRATOR_IMGS , "./int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
    std::cout << "Building tensorrt engine by INT8, please wait for a while..." << std::endl;
#else
    std::cout << "Building tensorrt engine by FLOAT32, please wait for a while..." << std::endl;
#endif
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

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
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
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
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
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
        std::ofstream p( PATH_ENGINE , std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file( PATH_ENGINE , std::ios::binary);
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
        std::cerr << "./yolov4 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov4 -d ../samples  // deserialize plan file and run inference" << std::endl;
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
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    int fcount = 0;
    float tt = 0;
    int ii = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;

        auto start = std::chrono::system_clock::now(); 
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img, INPUT_W , INPUT_H);
            // std::cout << pr_img.size() << std::endl;
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
            }
        }
        // Run inference
        doInference(*context, data, prob, BATCH_SIZE);
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE]);
        }
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        if ( f > (int)file_names.size() / 4 ) tt +=  (end - start).count() + 0 * ii++ ;  ; 
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            // std::cout << img.size() << std::endl;
            for (size_t j = 0; j < res.size(); j++) {
                //float *p = (float*)&res[j];
                //for (size_t k = 0; k < 7; k++) {
                //    std::cout << p[k] << ", ";
                //}
                //std::cout << std::endl;
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("detected/_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }
    std::cout  << "Average D2D time: " << (tt / ii) / 1000000 << "ms !"  << std::endl;
    std::cout  << "Average D2D time: " << 1000 / (tt / ii) << "fps !"  << std::endl;

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
