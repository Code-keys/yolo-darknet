/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _YOLO_H_
#define _YOLO_H_

#include <stdint.h>
#include <string>
#include <vector>
#include <memory>

#include "NvInfer.h"
#include "trt_utils.h"
#include "yololayer.h"
#include "mish.h"

typedef unsigned int uint;

typedef enum {
    /** NvDsInferContext operation succeeded. */
    NVDSINFER_SUCCESS = 0,
    /** Failed to configure the NvDsInferContext instance possibly due to an
     *  erroneous initialization property. */
    NVDSINFER_CONFIG_FAILED,
    /** Custom Library interface implementation failed. */
    NVDSINFER_CUSTOM_LIB_FAILED,
    /** Invalid parameters were supplied. */
    NVDSINFER_INVALID_PARAMS,
    /** Output parsing failed. */
    NVDSINFER_OUTPUT_PARSING_FAILED,
    /** CUDA error was encountered. */
    NVDSINFER_CUDA_ERROR,
    /** TensorRT interface failed. */
    NVDSINFER_TENSORRT_ERROR,
    /** Resource error was encountered. */
    NVDSINFER_RESOURCE_ERROR,
    /** TRT-IS error was encountered. */
    NVDSINFER_TRTIS_ERROR,
    /** Unknown error was encountered. */
    NVDSINFER_UNKNOWN_ERROR
} NvDsInferStatus;

class IModelParser
{
public:
    IModelParser() = default;
    /**
     * Destructor, make sure all external resource would be released here. */
    virtual ~IModelParser() = default;

    /**
     * Function interface for parsing custom model and building tensorrt
     * network.
     *
     * @param[in, out] network NvDsInfer will create the @a network and
     *                 implementation can setup this network layer by layer.
     * @return NvDsInferStatus indicating if model parsing was sucessful.
     */
    virtual NvDsInferStatus parseModel(
        nvinfer1::INetworkDefinition& network) = 0;

    /**
     * Function interface to check if parser can support full-dimensions.
     */
    virtual bool hasFullDimsSupported() const = 0;

    /**
     * Function interface to get the new model name which is to be used for
     * constructing the serialized engine file path.
     */
    virtual const char* getModelName() const = 0;
};


/**
 * Holds all the file paths required to build a network.
 */
struct NetworkInfo
{
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string deviceType;
    std::string inputBlobName;
};

/**
 * Holds information about an output tensor of the yolo network.
 */
struct TensorInfo
{
    std::string blobName;
    uint stride{0};
    uint gridSize{0};
    uint numClasses{0};
    uint numBBoxes{0};
    uint64_t volume{0};
    std::vector<uint> masks;
    std::vector<float> anchors;
    int bindingIndex{-1};
    float* hostBuffer{nullptr};
};

class Yolo : public IModelParser {
public:
    Yolo(const NetworkInfo& networkInfo);
    ~Yolo() override;
    bool hasFullDimsSupported() const override { return false; }
    const char* getModelName() const override {
        return m_ConfigFilePath.empty() ? m_NetworkType.c_str()
                                        : m_ConfigFilePath.c_str();
    }
    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition& network) override;

    nvinfer1::ICudaEngine *createEngine (nvinfer1::IBuilder* builder);

protected:
    const std::string m_NetworkType;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_DeviceType;
    const std::string m_InputBlobName;
    const std::string m_OutputBlobName;
    std::vector<TensorInfo> m_OutputTensors;
    std::vector<std::map<std::string, std::string>> m_ConfigBlocks;
    uint m_InputH;
    uint m_InputW;
    uint m_InputC;
    uint64_t m_InputSize;

    // TRT specific members
    std::vector<nvinfer1::Weights> m_TrtWeights;
    std::vector<nvinfer1::ITensor*> m_YoloTensor;

    std::vector<YoloKernel> m_YoloKernel;


private:
    NvDsInferStatus buildYoloNetwork(
        std::vector<float>& weights, nvinfer1::INetworkDefinition& network);
    std::vector<std::map<std::string, std::string>> parseConfigFile(
        const std::string cfgFilePath);
    void parseConfigBlocks();
    void destroyNetworkUtils();
};

#endif // _YOLO_H_
