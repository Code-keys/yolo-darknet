#ifndef RT_CLASS_H
#define RT_CLASS_H

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp> 
#include <string>
#include <fstream>
#include <chrono>
#include "NvInfer.h"  
#include "logging.h"


#include "yololayer.h"
#include "mish.h"
 
namespace TensorRt { 

#define DEVICE 0
#define BATCH_SIZE 1
using namespace nvinfer1; 
// #define NMS_THRESH 0.4
// #define BBOX_CONF_THRESH 0.5

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static char* INPUT_BLOB_NAME = "data";
static char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

class Detector  {

    void* buffers[2];
    float *data, *prob;    
    int inputIndex, outputIndex;
    float BBOX_CONF_THRESH, NMS_THRESH;
    std::vector< std::string > classnames;

    std::string class_namefile;
    std::string engine_name;

    IRuntime* runtime ;
    ICudaEngine* engine;
    IExecutionContext *context; 
    cudaStream_t stream;

public:
    Detector(std::string wtsOrengine ): engine_name(wtsOrengine)
    {    
        init_engine(); 
        BBOX_CONF_THRESH = 0.25; 
        NMS_THRESH = 0.45;
        infer_once();
    }

    ~Detector(){
        delete[] data;
        delete[] prob;
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }

    void help(){
        std::cout << " usage:\n      auto model = new Tensorrt::Detector( \" YOLO.engine\" );\n      model->predict_cv( img=img, /*conf=*/0.25, /*nms=*/ 0.45 );  " << std::endl ;
    }
    void init_engine(){
        cudaSetDevice(DEVICE);

        data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        //    data[i] = 1.0;
        prob = new float[BATCH_SIZE * OUTPUT_SIZE];

        std::ifstream file(engine_name, std::ios::binary);
        assert(file.good());
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);

        char *trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);

        PluginFactory pf;        /*   PluginFactory pf;  */

        engine = runtime->deserializeCudaEngine(trtModelStream,  size, &pf);
        assert(engine != nullptr);

        context = engine->createExecutionContext();
        assert(context != nullptr);

        delete[] trtModelStream;

        assert(engine->getNbBindings() == 2); 
        inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);

        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        
        // Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    void infer_once(){
        cv::Mat m( cv::Size(400, 400), CV_8UC3 , cv::Scalar(0, 255, 255 ) );
        predict_cv(m, BBOX_CONF_THRESH, NMS_THRESH);
    }

    float predict_cv( cv::Mat& img, float conf, float nms){
        BBOX_CONF_THRESH = conf, NMS_THRESH = nms;
        if (img.empty()) return .0f;
        auto start = std::chrono::system_clock::now();
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
        int b=0;
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        postprocess_img(img, 0);  
        auto end = std::chrono::system_clock::now();
        auto ret = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() ; 
        return ret;
    }

    std::vector< std::vector<int> >  predict( cv::Mat& img, float conf_thr, float nms_thr){
        BBOX_CONF_THRESH = conf_thr, NMS_THRESH = nms_thr;
        if (img.empty()) return {};
        auto start = std::chrono::system_clock::now();
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
        int b=0;
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);

        std::vector<std::vector<Yolo::Detection>> batch_res(BATCH_SIZE); 
        auto& res = batch_res[0];
        nms(res, &prob[ 0 * OUTPUT_SIZE], NMS_THRESH );
        std::vector< std::vector<int> > rreett;
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect( img, res[j].bbox );
            std::vector<int> temp;
            temp.push_back( (int)res[j].class_id );
            temp.push_back( (int)(res[j].det_confidence*1000) );
            temp.push_back( r.x ); /* 方形的左上角的x-坐标 */ 
            temp.push_back( r.y );
            temp.push_back( r.width ); /* 宽 pix */
            temp.push_back( r.height );
            rreett.push_back( temp ); 
        } 
        auto end = std::chrono::system_clock::now();
        auto ret = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() ; 
        std::cout <<"time wasted :  "<< ret << "  ms \n" ;
        return rreett;
    }


    static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
        int w, h, x, y;
        float r_w = input_w / (img.cols*1.0);
        float r_h = input_h / (img.rows*1.0);
        if (r_h > r_w) {
            w = input_w;
            h = r_w * img.rows;
            x = 0;
            y = (input_h - h) / 2;
        } else {
            w = r_h * img.cols;
            h = input_h;
            x = (input_w - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        return out;
    };

    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
    } 

    inline void postprocess_img(cv::Mat& retimg, int image_id){  
        std::vector<std::vector<Yolo::Detection>> batch_res(BATCH_SIZE); 
        auto& res = batch_res[image_id];
        nms(res, &prob[ image_id * OUTPUT_SIZE], NMS_THRESH ); 
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(retimg, res[j].bbox);
            cv::rectangle(retimg, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(retimg, std::to_string((int)res[j].class_id), 
                cv::Point(r.x, r.y - 1), 
                cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
    }

private:

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

    void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh) {
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
            std::sort(dets.begin(), dets.end(), 
                [](const Yolo::Detection& a, const Yolo::Detection& b){return a.det_confidence > b.det_confidence;});
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

};//rt_detector

} //namespace  



namespace TensorRT{

using namespace nvinfer1; 

#define USE_FP16  // comment out this if want to use FP32 

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 320;
static const int INPUT_W = 320;
static const int OUTPUT_SIZE = 6300 * 7;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob"; 

static Logger gLogger;




}



#endif