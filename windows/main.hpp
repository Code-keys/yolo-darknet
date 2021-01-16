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
#include "extra_lib\tensorrt\include\NvInferRuntime.h"
#pragma comment(lib, "extra_lib\\tensorrt\\lib\\nvinfer_plugin.lib")
#pragma comment(lib, "extra_lib\\tensorrt\\lib\\nvinfer.lib")
#endif

template<typename T>
class service {
public:
	service() {};
	~service() {};
private:
	std::string file;
};