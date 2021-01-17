#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "cuda_runtime_api.h"

#include <cmath>

#include "logging.h"
#include "yolo.h"
#include "trt_utils.h"
#include "yololayer.h"
#include "mish.h"

#include "opencv2/opencv.hpp"

using namespace nvinfer1;

Logger gLogger;
REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

cv::Mat preprocess_img(cv::Mat& img,int input_w,int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h* img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat& img, float bbox[4],int input_w,int input_h) {
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (input_h - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (input_h - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (input_w - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (input_w - r_h * img.cols) / 2;
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

bool cmp(Detection& a, Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Detection>& res, float *output, float ignore_thresh=0.4,float nms_thresh = 0.4) {
    std::map<float, std::vector<Detection>> m;
//    std::cout << "output[0] "<< output[0]<<std::endl;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + 7 * i + 4] <= ignore_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
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

int main(int argc,char* argv[])
{
    cudaSetDevice(0);
    char *trtModelStream{nullptr};
    size_t size{0};

    NetworkInfo networkInfo;

    networkInfo.networkType     = "yolov4";
    networkInfo.configFilePath  = "";
    networkInfo.wtsFilePath     = "";
    networkInfo.deviceType      = "kGPU";
    networkInfo.inputBlobName   = "data";


    IBuilder* builder = createInferBuilder(gLogger);
    if (argc == 2 && std::string(argv[1]) == "-s") {

        std::cout << "\n  cfg  : ";
        std::cin >> networkInfo.configFilePath ;
        std::cout << "\n  weights  : ";
        std::cin >> networkInfo.wtsFilePath ;
        
        std::string modelname = networkInfo.configFilePath.substr(0, (networkInfo.configFilePath.find(".cfg")) + ".engine"))); // networkInfo.networkType + ".engine";
        std::cout << "engine will be build : \n"<< modelname << std::endl;

        IHostMemory* modelStream{nullptr};
        Yolo yolo(networkInfo);
        ICudaEngine *cudaEngine = yolo.createEngine (builder);
        modelStream = cudaEngine->serialize();
        assert(modelStream != nullptr);
        std::ofstream p(modelname, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::string modelname;
        std::cout << "\n  engine  : ";
        std::cin >> modelname ;
        std::ifstream file(modelname, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov3 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov3 -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    int numbindings=engine->getNbBindings();
    std::cout<< "getNbBindings: " << numbindings<<std::endl;

    const char* layername = engine->getBindingName(1);
    std::cout<< "getBindingName:1 " << layername<<std::endl;
    Dims out = engine->getBindingDimensions(1);
    std::cout<< "out dims: " << out.d[0]<<" "<<out.d[1]<<" "<<out.d[2]<<" "<<out.d[3]<<std::endl;

    Dims in = engine->getBindingDimensions(0);
    std::cout<< "out dims: " << in.d[0]<<" "<<in.d[1]<<" "<<in.d[2]<<" "<<in.d[3]<<std::endl;

    int input_h =  in.d[1];
    int input_w =  in.d[2];
    int OUTPUT_SIZE = out.d[0];

    void* buffers[2];
    int batchSize = 1;

    cudaMalloc(&buffers[0], batchSize * 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffers[1], batchSize * OUTPUT_SIZE * sizeof(float));

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    std::string source;
    uint const source_type=0;
    std::cout << "\n Type The path of source .\n  camera for camera \n  video\n  images\n  Folder\n  Txt contain the path of images.\n";
    std::cin >> source;

    std::string const basename  = filename.substr(filename.find_last_of("/") + 1);

    if ( source.find(".mp4") ||source.find(".avi")||source.find(".mp4")||source.find(".mp4")||source.find(".mp4") ){
        cv::VideoCapture cap("source");
        source_type = 2;

    }else if(source.find("camera") ){
        cv::VideoCapture cap(0);
        source_type = 1;



    }else if(source.find(".jpg") ){
        source_type = 3;




    }else if(source.find(".txt") ){
        source_type = 5;



    }else if(source.find(".") ){
        source_type = 4;
    // std::string folder;
    // std::cin >> folder;
    // const char *folder = folder.c_str();
    // std::vector<std::string> file_names;  //const_cast<char*>
    // if (read_files_in_dir(folder, file_names) < 0) {
    //     std::cout << "read_files_in_dir failed." << std::endl;
    //     return -1;
    // }
    }

    cv::Mat img;
    cv::Mat pr_img;
    int fp = 0;

    bool detect = false;

    float data[3 * input_h * input_w];
    float prob[OUTPUT_SIZE];

    std::cout<<"start detect"<<std::endl;


    while (true){
        if(!detect){detect=true; continue;};

        switch(source_type){
            case 1:
            case 2:
                cap >> img;
                break;
            case 3:
                img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);

            case 4:
                
            case 5:
                
            default :
                std::cout << "";
                break;]
        };

        cv::Mat pr_img = preprocess_img(img,input_w,input_h);
        for (int i = 0; i < input_h * input_w; i++) {
            data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[i + input_h * input_w] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + 2 * input_h * input_w] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }


//        // Run inference
        auto start = std::chrono::system_clock::now();
        cudaMemcpyAsync(buffers[0], data, batchSize * 3 * input_w * input_h * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueue(batchSize, buffers, stream, nullptr);
        cudaMemcpyAsync(prob, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<Detection> res;
        nms(res, prob);
        for (size_t j = 0; j < res.size(); j++) {
            float *p = (float*)&res[j];
            cv::Rect r = get_rect(img, res[j].bbox,input_w,input_h);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            std::string text = std::to_string((int)res[j].class_id) + " "+
                    std::to_string((float)res[j].det_confidence)+" "+
                    std::to_string((float)res[j].class_confidence);
            cv::putText(img, text, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        cv::imshow("_", img);
        cv::imwrite(img,"../infernce/"+ file_names[f - fcount + 1 + b])
        if(cv::waitKey(1)==27){break;}
    }









    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}







