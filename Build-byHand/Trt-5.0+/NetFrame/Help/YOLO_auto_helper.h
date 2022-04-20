#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__

#include <set>
#include <map>
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#define UNUSED(expr) (void)(expr)
#define DIVUP(n, d) ((n) + (d)-1) / (d)

std::string trim(std::string s);
float clamp(const float val, const float minVal, const float maxVal);
bool fileExists(const std::string fileName, bool verbose = true);
std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType);
std::string dimsToString(const nvinfer1::Dims d);
void displayDimType(const nvinfer1::Dims d);
int getNumChannels(nvinfer1::ITensor* t);
uint64_t get3DTensorVolume(nvinfer1::Dims inputDims);

nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
	nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
	int& inputChannels, nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvBNActive(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
	int& inputChannels, nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
	nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddBatchNorm(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
	int& inputChannels, nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddActivation(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
	int& inputChannels, nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network);
void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr);

#endif
