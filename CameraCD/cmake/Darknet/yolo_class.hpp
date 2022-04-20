#ifndef DARKNET_CLASS_hpp
#define DARKNET_CLASS_hpp

#ifndef LIB_API
#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define LIB_API
#else
#define LIB_API
#endif
#endif
#endif

// #include "ABCDetector.hpp"

#ifdef __cplusplus
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono> // time infer

#define OPENCV
#ifdef OPENCV
#include <opencv2/opencv.hpp>          // C++
#include <opencv2/highgui/highgui_c.h> // C
#include <opencv2/imgproc/imgproc_c.h> // C
#endif

#define C_SHARP_MAX_OBJECTS 1000

struct bbox_t
{
    unsigned int x, y, w, h;     // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                  // confidence - probability that the object was found correctly
    unsigned int obj_id;         // class of object - from range [0, classes-1]
    unsigned int track_id;       // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter; // counter of frames on which the object was detected
    float x_3d, y_3d, z_3d;      // center of object (in Meters) if ZED 3D Camera is used
};

struct image_t
{
    int h;       // height
    int w;       // width
    int c;       // number of chanels (3 - for RGB)
    float *data; // pointer to the image data
};

struct bbox_t_container
{
    bbox_t candidates[C_SHARP_MAX_OBJECTS];
};

extern "C" LIB_API int init(const char *configurationFilename, const char *weightsFilename, int gpu);
extern "C" LIB_API int detect_image(const char *filename, bbox_t_container &container);
extern "C" LIB_API int detect_mat(const uint8_t *data, const size_t data_length, bbox_t_container &container);
extern "C" LIB_API int dispose();
extern "C" LIB_API int get_device_count();
extern "C" LIB_API int get_device_name(int gpu, char *deviceName);
extern "C" LIB_API bool built_with_cuda();
extern "C" LIB_API bool built_with_cudnn();
extern "C" LIB_API bool built_with_opencv();

// class ABCDetector{
// public:
//     virtual float predict_cv(cv::Mat& im, float confthresh, float nmsthresh){};
// };

class Detector // :ABCDetector   -> ::Bug::  Bus error(core dumped) -> namespace fixed
{
    std::shared_ptr<void> detector_gpu_ptr;
    std::deque<std::vector<bbox_t>> prev_bbox_vec_deque;

    std::string _cfg_filename, _weight_filename;
    std::vector<std::string> _obj_names;

public:
    const int cur_gpu_id;
    float nms = .45;
    bool wait_stream;

    LIB_API Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
    LIB_API ~Detector();

    LIB_API std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
    LIB_API std::vector<bbox_t> detect(image_t img, float thresh = 0.2, bool use_mean = false);
    static LIB_API image_t load_image(std::string image_filename);
    static LIB_API void free_image(image_t m);
    LIB_API int get_net_width() const;
    LIB_API int get_net_height() const;
    LIB_API int get_net_color_depth() const;

    LIB_API void *get_cuda_context();

    std::vector<bbox_t> detect_resized(image_t img, int init_w, int init_h, float thresh = 0.2, bool use_mean = false)
    {
        if (img.data == NULL)
            throw std::runtime_error("Image is empty");
        auto detection_boxes = detect(img, thresh, use_mean);
        float wk = (float)init_w / img.w, hk = (float)init_h / img.h;
        for (auto &i : detection_boxes)
            i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
        return detection_boxes;
    };

    std::vector<std::string> loadclasses(std::string f)
    {
        std::ifstream file(f);
        if (!file.is_open())
            return _obj_names;
        _obj_names.clear();
        for (std::string line; getline(file, line);)
            _obj_names.push_back(line);
        return _obj_names;
    };
    std::vector<std::string> getclasses()
    {
        return _obj_names;
    };

#ifdef OPENCV
    float predict_cv(cv::Mat &im, float confthresh, float nmsthresh)
    {
        auto start = std::chrono::high_resolution_clock::now();
        nms = nmsthresh;
        std::vector<bbox_t> bbxes = this->detect_mat(im, confthresh);
        this->draw_boxes(im, bbxes);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return duration.count();
    };

    void draw_boxes(cv::Mat &mat_img, std::vector<bbox_t> &result_vec)
    {
        for (auto &i : result_vec)
        {
            cv::Scalar color = obj_id_to_color(i.obj_id);
            cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        }
    };

    std::vector<bbox_t> detect_mat(cv::Mat mat, float thresh = 0.3, bool use_mean = false)
    {
        if (mat.data == NULL)
            throw std::runtime_error("Image is empty");
        auto image_ptr = mat_to_image_resize(mat);
        return detect_resized(*image_ptr, mat.cols, mat.rows, thresh, use_mean);
    };

    std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat) const
    {
        if (mat.data == NULL)
            return std::shared_ptr<image_t>(NULL);

        cv::Size network_size = cv::Size(get_net_width(), get_net_height());
        cv::Mat det_mat;
        if (mat.size() != network_size)
            cv::resize(mat, det_mat, network_size);
        else
            det_mat = mat; // only reference is copied

        return mat_to_image(det_mat);
    };

    static std::shared_ptr<image_t> mat_to_image(cv::Mat img_src)
    {
        cv::Mat img;
        if (img_src.channels() == 4)
            cv::cvtColor(img_src, img, cv::COLOR_RGBA2BGR);
        else if (img_src.channels() == 3)
            cv::cvtColor(img_src, img, cv::COLOR_RGB2BGR);
        else if (img_src.channels() == 1)
            cv::cvtColor(img_src, img, cv::COLOR_GRAY2BGR);
        else
            std::cerr << " Warning: img_src.channels() is not 1, 3 or 4. It is = " << img_src.channels() << std::endl;
        std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img)
                                           {
                                               free_image(*img);
                                               delete img;
                                           });
        *image_ptr = mat_to_image_custom(img);
        return image_ptr;
    };

    static cv::Scalar obj_id_to_color(int obj_id)
    {
        int const colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
        int const offset = obj_id * 123457 % 6;
        int const color_scale = 150 + (obj_id * 123457) % 100;
        cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
        color *= color_scale;
        return color;
    };

private:
    static image_t mat_to_image_custom(cv::Mat mat)
    {
        int w = mat.cols;
        int h = mat.rows;
        int c = mat.channels();
        image_t im = make_image_custom(w, h, c);
        unsigned char *data = (unsigned char *)mat.data;
        int step = mat.step;
        for (int y = 0; y < h; ++y)
        {
            for (int k = 0; k < c; ++k)
            {
                for (int x = 0; x < w; ++x)
                {
                    im.data[k * w * h + y * w + x] = data[y * step + x * c + k] / 255.0f;
                }
            }
        }
        return im;
    }

    static image_t make_empty_image(int w, int h, int c)
    {
        image_t out;
        out.data = 0;
        out.h = h;
        out.w = w;
        out.c = c;
        return out;
    }

    static image_t make_image_custom(int w, int h, int c)
    {
        image_t out = make_empty_image(w, h, c);
        out.data = (float *)calloc(h * w * c, sizeof(float));
        return out;
    }

#endif // opencv
};

#endif // __cplusplus

#endif // DARKNET_HPP
