#include <experimental/filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <math.h>
#include "NvInferPlugin.h"

#include "API/YOLO_auto_helper.h"


static void leftTrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType)
{
    assert(fileExists(weightsFilePath));
    std::cout << "Loading pre-trained weights..." << std::endl;
    std::ifstream file(weightsFilePath, std::ios_base::binary);
    assert(file.good());
    std::string line;

    if (networkType == "yolov2")
    {
        file.ignore(4 * 4);
    }
    else if ((networkType == "yolov3") || (networkType == "yolov3-tiny")
             || (networkType == "yolov4") || (networkType == "yolov4-tiny"))
    {
        file.ignore(4 * 5);
    }
    else
    {
        std::cout << "Invalid network type" << std::endl;
        assert(0);
    }

    std::vector<float> weights;
    char floatWeight[4];
    while (!file.eof())
    {
        file.read(floatWeight, 4);
        assert(file.gcount() == 4);
        weights.push_back(*reinterpret_cast<float*>(floatWeight));
        if (file.peek() == std::istream::traits_type::eof()) break;
    }
    std::cout << "Loading weights of " << networkType << " complete!"
              << std::endl;
    std::cout << "Total Number of weights read : " << weights.size() << std::endl;
    return weights;
}

std::string dimsToString(const nvinfer1::Dims d)
{
    std::stringstream s;
    assert(d.nbDims >= 1);
    for (int i = 0; i < d.nbDims - 1; ++i)
    {
        s << std::setw(4) << d.d[i] << " x";
    }
    s << std::setw(4) << d.d[d.nbDims - 1];

    return s.str();
}

void displayDimType(const nvinfer1::Dims d)
{
    std::cout << "(" << d.nbDims << ") ";
    for (int i = 0; i < d.nbDims; ++i)
    {
        switch (d.type[i])
        {
        case nvinfer1::DimensionType::kSPATIAL: std::cout << "kSPATIAL "; break;
        case nvinfer1::DimensionType::kCHANNEL: std::cout << "kCHANNEL "; break;
        case nvinfer1::DimensionType::kINDEX: std::cout << "kINDEX "; break;
        case nvinfer1::DimensionType::kSEQUENCE: std::cout << "kSEQUENCE "; break;
        }
    }
    std::cout << std::endl;
}

int getNumChannels(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[0];
}

uint64_t get3DTensorVolume(nvinfer1::Dims inputDims)
{
    assert(inputDims.nbDims == 3);
    return inputDims.d[0] * inputDims.d[1] * inputDims.d[2];
}

nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "maxpool");
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int size = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));

    nvinfer1::IPoolingLayer* pool
        = network->addPooling(*input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{size, size});
    assert(pool);
    std::string maxpoolLayerName = "maxpool_" + std::to_string(layerIdx);
    pool->setStride(nvinfer1::DimsHW{stride, stride});
    pool->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    pool->setName(maxpoolLayerName.c_str());

    return pool;
}

nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "convolutional");
    assert(block.at("activation") == "linear");
    assert(block.find("filters") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;
  
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, filters};
    float* val = new float[filters];
    for (int i = 0; i < filters; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convBias.values = val;
    trtWeights.push_back(convBias);

	int groups = 1;
	if (block.find("groups") != block.end())
		groups = std::stoi(block.at("groups"));

    int size = filters * inputChannels * kernelSize * kernelSize / groups;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    val = new float[size];
    for (int i = 0; i < size; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(
        *input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{pad, pad});
	conv->setNbGroups(groups);

    return conv;
}

nvinfer1::ILayer* netAddConvBNActive(int layerIdx, std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                    int& inputChannels, nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") != block.end());
    assert(block.at("batch_normalize") == "1");
    assert(block.find("filters") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());
	
    bool batchNormalize, bias;
    if (block.find("batch_normalize") != block.end())
    {
        batchNormalize = (block.at("batch_normalize") == "1");
        bias = false;
    }
    else
    {
        batchNormalize = false;
        bias = true;
    }

    assert(batchNormalize == true && bias == false);
    UNUSED(batchNormalize);
    UNUSED(bias);

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;

    std::vector<float> bnBiases;
    for (int i = 0; i < filters; ++i)
    {
        bnBiases.push_back(weights[weightPtr]);
        weightPtr++;
    }

    std::vector<float> bnWeights;
    for (int i = 0; i < filters; ++i)
    {
        bnWeights.push_back(weights[weightPtr]);
        weightPtr++;
    }
 
    std::vector<float> bnRunningMean;
    for (int i = 0; i < filters; ++i)
    {
        bnRunningMean.push_back(weights[weightPtr]);
        weightPtr++;
    }

    std::vector<float> bnRunningVar;
    for (int i = 0; i < filters; ++i)
    {
        bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5));
        weightPtr++;
    }

	int groups = 1;
	if (block.find("groups") != block.end())
		groups = std::stoi(block.at("groups"));

    int size = filters * inputChannels * kernelSize * kernelSize / groups;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* val = new float[size];

	for (int i = 0; i < size; ++i)
	{
		val[i] = weights[weightPtr];
		weightPtr++;
	}

    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    trtWeights.push_back(convBias);
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(
        *input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);

    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{pad, pad});
	conv->setNbGroups(groups);

    size = filters;
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* shiftWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        shiftWt[i] = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
    }
    shift.values = shiftWt;
    float* scaleWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
    }
    scale.values = scaleWt;
    float* powerWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        powerWt[i] = 1.0;
    }
    power.values = powerWt;
    trtWeights.push_back(shift);
    trtWeights.push_back(scale);
    trtWeights.push_back(power);

    nvinfer1::IScaleLayer* bn = network->addScale(
        *conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn != nullptr);
    std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
    bn->setName(bnLayerName.c_str());

    if(block.at("activation") == "leaky"){
        nvinfer1::ITensor* bnOutput = bn->getOutput(0);
        nvinfer1::IActivationLayer* leaky = network->addActivation(
                    *bnOutput, nvinfer1::ActivationType::kLEAKY_RELU);
        leaky->setAlpha(0.1);
        assert(leaky != nullptr);
        std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
        leaky->setName(leakyLayerName.c_str());
        return leaky;
    }else if(block.at("activation") == "mish")
    {
        auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
        const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
        nvinfer1::IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(layerIdx)).c_str(), pluginData);
        nvinfer1::ITensor* inputTensors[] = {bn->getOutput(0)};
        auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
        return mish;
    }
    else if (block.at("activation") == "silu")
	{
		nvinfer1::ITensor* bnOutput = bn->getOutput(0);
        nvinfer1::IActivationLayer* sig = network->addActivation(
                    *bnOutput, nvinfer1::ActivationType::kSIGMOID);
        assert(sig != nullptr);
        auto silu = network->addElementWise( *bnOutput, *sig->getOutput(0), nvinfer1::ElementWiseOperation::kPROD );
        assert(silu != nullptr);
        std::string sigLayerName = "silu_" + std::to_string(layerIdx);
        silu->setName(sigLayerName.c_str());
        return silu;
	}
	else if (block.at("activation") == "linear")
	{
		return bn;
	}
}

nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights, std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
	nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "upsample");
	nvinfer1::Dims inpDims = input->getDimensions();
	assert(inpDims.nbDims == 3);
	assert(inpDims.d[1] == inpDims.d[2]);
	int h = inpDims.d[1];
	int w = inpDims.d[2];
	int stride = std::stoi(block.at("stride"));

	float *deval = new float[inputChannels * stride * stride];
	for (int i = 0; i < inputChannels * stride * stride; i++) {
		deval[i] = 1.0;
	}
	nvinfer1::Weights emptywts{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
	nvinfer1::Weights deconvwts{ nvinfer1::DataType::kFLOAT, nullptr, inputChannels * stride * stride };
	trtWeights.push_back(emptywts);
	trtWeights.push_back(deconvwts);
	deconvwts.values = deval;
	nvinfer1::IDeconvolutionLayer* deconv = network->addDeconvolutionNd(*input, inputChannels, nvinfer1::DimsHW{ stride, stride }, deconvwts, emptywts);
	deconv->setStrideNd(nvinfer1::DimsHW{ stride, stride });
	deconv->setNbGroups(inputChannels);

	std::string deconvLayerName = "upsample_" + std::to_string(layerIdx);
	deconv->setName(deconvLayerName.c_str());

	return deconv;
}

nvinfer1::ILayer* netAddBatchNorm(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
	int& inputChannels, nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "batchnorm");
	assert(block.find("filter") != block.end());
	int filters = std::stoi(block.at("filter"));

	std::vector<float> bnBiases;
	for (int i = 0; i < filters; ++i)
	{
		bnBiases.push_back(weights[weightPtr]);
		weightPtr++;
	}

	std::vector<float> bnWeights;
	for (int i = 0; i < filters; ++i)
	{
		bnWeights.push_back(weights[weightPtr]);
		weightPtr++;
	}

	std::vector<float> bnRunningMean;
	for (int i = 0; i < filters; ++i)
	{
		bnRunningMean.push_back(weights[weightPtr]);
		weightPtr++;
	}

	std::vector<float> bnRunningVar;
	for (int i = 0; i < filters; ++i)
	{
		bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5));
		weightPtr++;
	}
	int size = filters;
	nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* shiftWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		shiftWt[i] = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
	}
	shift.values = shiftWt;
	float* scaleWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
	}
	scale.values = scaleWt;
	float* powerWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		powerWt[i] = 1.0;
	}
	power.values = powerWt;
	trtWeights.push_back(shift);
	trtWeights.push_back(scale);
	trtWeights.push_back(power);

	nvinfer1::IScaleLayer* bn = network->addScale(
		*input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
	assert(bn != nullptr);
	std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
	bn->setName(bnLayerName.c_str());

	return bn;
}

nvinfer1::ILayer* netAddActivation(int layerIdx, std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
	int& inputChannels, nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "activation");
	if (block.at("activation") == "leaky") {
		nvinfer1::ITensor* output = input;
		nvinfer1::IActivationLayer* leaky = network->addActivation(
			*output, nvinfer1::ActivationType::kLEAKY_RELU);
		leaky->setAlpha(0.1);
		assert(leaky != nullptr);
		std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
		leaky->setName(leakyLayerName.c_str());
		return leaky;
	}
	else if (block.at("activation") == "mish")
	{
		auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
		const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
		nvinfer1::IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(layerIdx)).c_str(), pluginData);
		nvinfer1::ITensor* inputTensors[] = { input };
		auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
		return mish;
	}
	else if (block.at("activation") == "linear")
	{
		nvinfer1::IIdentityLayer* linear = network->addIdentity(*input);
		return linear;
	}
}


void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr)
{
    std::cout << std::setw(6) << std::left << layerIndex << std::setw(15) << std::left << layerName;
    std::cout << std::setw(20) << std::left << layerInput << std::setw(20) << std::left
              << layerOutput;
    std::cout << std::setw(6) << std::left << weightPtr << std::endl;
}


