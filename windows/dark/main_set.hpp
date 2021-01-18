#include <iostream>

#if CUDA
#include "Kernel_Factory.hpp"
#else
#include "Factory.hpp"
#endif

#if OPENCV
#pragma comment(lib, "extra_lib\\opencv\\opencv_world320d.lib")
#pragma comment(lib, "extra_lib\\opencv\\opencv_world320.lib")
#include <opencv2/opencv.hpp>
#endif

#if TENSORRT
#include "include\NvInferRuntime.h"
#pragma comment(lib, "extra_lib\\nvinfer_plugin.lib")
#pragma comment(lib, "extra_lib\\nvinfer.lib")
#endif

template<typename T>
class service {
public:
	service() {};
	~service() {};
private:
	std::string file;
};