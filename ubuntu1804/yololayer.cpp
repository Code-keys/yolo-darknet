#include "yololayer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

using namespace YoloLayer;
using namespace nvinfer1;

YoloLayerPlugin::YoloLayerPlugin()
{
    mClassCount = CLASS_NUM;
    mYoloKernel.clear();
    mYoloKernel.push_back(yolo1);
    mYoloKernel.push_back(yolo2);
    mYoloKernel.push_back(yolo3);

    mKernelCount = mYoloKernel.size();
}

YoloLayerPlugin::~YoloLayerPlugin()
{
}

// create the plugin at runtime from a byte stream
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
{

    const char *d = reinterpret_cast<const char *>(data), *a = d;
    read(d, mClassCount);
    read(d, mThreadCount);
    read(d, mKernelCount);
    mYoloKernel.resize(mKernelCount);
    auto kernelSize = mKernelCount*sizeof(YoloKernel);
    memcpy(mYoloKernel.data(),d,kernelSize);
    d += kernelSize;

    assert(d == a + length);
}

void YoloLayerPlugin::serialize(void* buffer) const
{
    std::cout<<"start getSerializationSize"<<std::endl;
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, mClassCount);
    write(d, mThreadCount);
    write(d, mKernelCount);
    auto kernelSize = mKernelCount*sizeof(YoloKernel);
    memcpy(d,mYoloKernel.data(),kernelSize);
    d += kernelSize;

    assert(d == a + getSerializationSize());
}

size_t YoloLayerPlugin::getSerializationSize() const
{
    std::cout<<"start getSerializationSize"<<std::endl;
    return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(YoloLayer::YoloKernel) * mYoloKernel.size();
}

int YoloLayerPlugin::initialize()
{
    return 0;
}

Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    //output the result to channel
    int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

    return Dims3(totalsize + 1, 1, 1);
}

// Set plugin namespace
void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

bool YoloLayerPlugin::supportsFormat (
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
    return (type == nvinfer1::DataType::kFLOAT &&
            format == nvinfer1::PluginFormat::kNCHW);
}

void YoloLayerPlugin::configureWithFormat (
    const nvinfer1::Dims* inputDims, int nbInputs,
    const nvinfer1::Dims* outputDims, int nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    assert(nbInputs == 3);
    assert (format == nvinfer1::PluginFormat::kNCHW);
    assert(inputDims != nullptr);
}

const char* YoloLayerPlugin::getPluginType() const
{
    return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const
{
    return "1";
}

void YoloLayerPlugin::destroy()
{
    delete this;
}

// Clone the plugin
IPluginV2* YoloLayerPlugin::clone() const
{
    return new YoloLayerPlugin();
}

int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize, mYoloKernel, mThreadCount);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
