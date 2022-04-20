# pragma once

#include <memory>
#include <iostream>
#include <memory>
#include <chrono>
#include <string>

#include "torch/script.h"
#include "torch/torch.h"

// #include <c10/cuda/CUDAStream.h>
// #include <ATen/cuda/CUDAEvent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace TorchScpt{
enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};

static const int INPUT_H = 768 ;
static const int INPUT_W = 1536 ;

class Detector {
public:
    Detector(const std::string& model_path, std::string class_path , int gpuid );
    virtual ~Detector();

    std::vector<std::vector<Detection>> 
        Run(const cv::Mat& img, float conf_threshold, float iou_threshold);
    cv::Mat& draw_result(cv::Mat& img,
                const std::vector<std::vector<Detection>>& detections );
    cv::Mat& pic2pic(cv::Mat& img,
            float conf_threshold, float iou_threshold );
    cv::Mat& path2pic(const std::string img_path,
                       float conf_threshold, float iou_threshold );  
    float predict_cv(cv::Mat& img, float conf_thres, float iou_thres);
    
    std::vector<std::vector<int>> predict(cv::Mat img, float conf_thres,float iou_thres);

    void help(){
        std::cout << "Usage:\n    auto detector = new TorchScpt::Detector( pt ,"" ,0);\n    detector->predict_cv( img, 0.25, 0.45 );\n";
    }
private: 
    void ChangeNames(const std::vector<std::string>& names  );
    std::vector<std::string> LoadNames(const std::string& path); 
    std::vector<std::string> getNames();

    static std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(INPUT_H, INPUT_W));
 
    static std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor& detections,
                                                              float pad_w, float pad_h, float scale, const cv::Size& img_shape,
                                                              float conf_thres = 0.4, float iou_thres = 0.6);
 
    static void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
                                 float scale, const cv::Size& img_shape);
 
    static torch::Tensor xywh2xyxy(const torch::Tensor& x);
 
    static void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                 const at::TensorAccessor<float, 2>& det,
                                 std::vector<cv::Rect>& offset_box_vec,
                                 std::vector<float>& score_vec);

    bool half_;  int GCpuId_ ;
    std::vector<std::string>  class_names;
    torch::Device device_;
    torch::jit::script::Module module_;
}; 
}
