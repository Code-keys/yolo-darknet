#ifndef NCNN_CLASS_hpp
#define NCNN_CLASS_hpp


#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/videoio/videoio.hpp>
#endif

#include <vector>
#include <string>
#include <chrono>
#include <stdio.h>

namespace YOLONCNN{


#define NCNN_GPU
#undef NCNN_GPU



struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class Detector {

    static int detect_yolov4(const cv::Mat& bgr, std::vector<Object>& objects, std::vector<int>& target_size, ncnn::Net* yolov4)
    {
        int img_w = bgr.cols;
        int img_h = bgr.rows;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size[0], target_size[1] );

        const float mean_vals[3] = {0, 0, 0};
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = yolov4->create_extractor();

        ex.input("data", in);

        ncnn::Mat out;
        ex.extract("output", out);

        objects.clear();
        for (int i = 0; i < out.h; i++)
        {
            const float* values = out.row(i);

            Object object;
            object.label = values[0];
            object.prob = values[1];
            object.rect.x = values[2] * img_w;
            object.rect.y = values[3] * img_h;
            object.rect.width = values[4] * img_w - object.rect.x;
            object.rect.height = values[5] * img_h - object.rect.y;

            objects.push_back(object);
        }

        return 0;
    };

    static int draw_objects(cv::Mat& image, const std::vector<Object>& objects )
    {
        static const char* class_names[] = {"background", "person", "bicycle",
                                            "car", "motorbike", "aeroplane", "bus", "train", "truck",
                                            "boat", "traffic light", "fire hydrant", "stop sign",
                                            "parking meter", "bench", "bird", "cat", "dog", "horse",
                                            "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                            "backpack", "umbrella", "handbag", "tie", "suitcase",
                                            "frisbee", "skis", "snowboard", "sports ball", "kite",
                                            "baseball bat", "baseball glove", "skateboard", "surfboard",
                                            "tennis racket", "bottle", "wine glass", "cup", "fork",
                                            "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                                            "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
                                            "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                                            "cell phone", "microwave", "oven", "toaster", "sink",
                                            "refrigerator", "book", "clock", "vase", "scissors",
                                            "teddy bear", "hair drier", "toothbrush"
                                           };

        // cv::Mat image = bgr.clone();

        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

            // char text[256];
            // sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            // int baseLine = 0;
            // cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            // int x = obj.rect.x;
            // int y = obj.rect.y - label_size.height - baseLine;
            // if (y < 0)
            //     y = 0;
            // if (x + label_size.width > image.cols)
            //     x = image.cols - label_size.width;

            // cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            //               cv::Scalar(255, 255, 255), -1);

            // cv::putText(image, text, cv::Point(x, y + label_size.height),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        return 0;
    };


public:

    Detector( std::string _param, std::string _bin, std::vector<int> w_h, int gpu ) :
        gpu_id(gpu), bin(_bin), param(_param), target_size(w_h) {
        /* --> Set the params you need for the ncnn inference <-- */

        yolov4 = new ncnn::Net();

        yolov4->opt.num_threads = 1; //You need to compile with libgomp for multi thread support

        if( gpu_id > 0){

            yolov4->opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support
            // initialize when app starts
#ifdef NCNN_GPU
            ncnn::create_gpu_instance();// line1
            // deinitialize when app exits
            // get gpu count
            int gpu_count = ncnn::get_gpu_count();

            // ncnn::VulkanDevice vkdev; // use default gpu
            ncnn::VulkanDevice vkdev( gpu_id ); // use device -0、1、2-

            net.set_vulkan_device(&vkdev);
#endif
        }
        yolov4->opt.use_winograd_convolution = true;
        yolov4->opt.use_sgemm_convolution = true;
        yolov4->opt.use_fp16_packed = true;
        yolov4->opt.use_fp16_storage = true;
        yolov4->opt.use_fp16_arithmetic = true;
        yolov4->opt.use_packing_layout = true;
        yolov4->opt.use_shader_pack8 = false;
        yolov4->opt.use_image_storage = false;

        /* --> End of setting params <-- */
        int ret = 0;

        ret += yolov4->load_param( param.data() );
        if (ret != 0)
        {
            fprintf(stderr, "Failed to load model or param, error %d", ret);
        }

        ret += yolov4->load_model( bin.data() );
        if (ret != 0)
        {
            fprintf(stderr, "Failed to load model or param, error %d", ret);
        }
    };

    ~Detector(){
#ifdef NCNN_GPU
        ncnn::destroy_gpu_instance();// line3
#endif
    delete yolov4;
    };


    float predict_cv(cv::Mat& frame,  float conf=0.25, float iou=0.45)
    {
        if (frame.empty())
        {
            fprintf(stderr, "Failed to read image \n");
            return -1.f;
        }
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<Object> objects;

        detect_yolov4(frame, objects, target_size , yolov4); //Create an extractor and run detection
        draw_objects(frame, objects );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)  ;
        return duration.count();
    }


private:

    ncnn::Net* yolov4;
    int gpu_id;
    std::string bin, param;
    std::vector<int> target_size  /* 416  640 */ ;
};  // END class

} // END namespace


#endif
