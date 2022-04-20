#ifndef ENTROPY_CALIBRATOR_H
#define ENTROPY_CALIBRATOR_H

#include "NvInfer.h"
#include <string>
#include <vector>

#include <dirent.h>
#include <opencv2/opencv.hpp> 


#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache = true);

    virtual ~Int8EntropyCalibrator2();
    int getBatchSize() const override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* cache, size_t length) override;

private:
    int batchsize_;
    int input_w_;
    int input_h_;
    int img_idx_;
    std::string img_dir_;
    std::vector<std::string> img_files_;
    size_t input_count_;
    std::string calib_table_name_;
    const char* input_blob_name_;
    bool read_cache_;
    void* device_input_;
    std::vector<char> calib_cache_;

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
    }

    static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
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
}; 



#endif // ENTROPY_CALIBRATOR_H
