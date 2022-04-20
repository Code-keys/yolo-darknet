#include "YOLO_auto.h"


#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "utils.h"
#include "logging.h"
#include "yololayer.h"
#include "mish.h"
#include "calibrator.h"
#include "API/YOLO_auto_helper.h"

// set the typeof YOLOv5

#define YOLOv5_cfg_ratio  1  //s :1 , m: 2//3 , l : 2, x : 5//4 
//(YOLOv5_cfg_ratio 不可加括号)  and Not Forget modify the layername before the YOLO_Layer

using namespace nvinfer1;
#define USE_INT8  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25
#define BATCH_SIZE 1
 
const char* PATH_CALIBRATOR_IMGS = "/home/nvidia/CX/VisDrone/test/";
const char* PATH_CALIBRATOR = "./int8.calib";


// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

/**************************************//**************************************//**************************************/
bool fileExists(const std::string fileName, bool verbose) {
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
} 

std::vector<std::map<std::string, std::string>>
parseConfigFile(const std::string cfgFilePath)
{
    assert(fileExists(cfgFilePath));
    std::ifstream file(cfgFilePath);
    assert(file.good());
    std::string line;
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;

    while (getline(file, line))
    {
		line = trim(line);
        if (line.size() == 0) continue;
        if (line.front() == '#') continue;
        if (line.front() == '[')
        {
            if (block.size() > 0)
            {
                blocks.push_back(block);
                block.clear();
            }
            std::string key = "type";
            std::string value = trim(line.substr(1, line.size() - 2));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
        else
        {
            int cpos = line.find('=');
            std::string key = trim(line.substr(0, cpos));
            std::string value = trim(line.substr(cpos + 1));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
    }
    blocks.push_back(block);
    return blocks;
} 
void parseConfigBlocks()
{
    for (auto block : m_ConfigBlocks) {
        if (block.at("type") == "net")
        {
            assert((block.find("height") != block.end())
                   && "Missing 'height' param in network cfg");
            assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
            assert((block.find("channels") != block.end())
                   && "Missing 'channels' param in network cfg");

            m_InputH = std::stoul(block.at("height"));
            m_InputW = std::stoul(block.at("width"));
            m_InputC = std::stoul(block.at("channels"));
            assert(m_InputW == m_InputH);
            m_InputSize = m_InputC * m_InputH * m_InputW;
        }
        else if ((block.at("type") == "region") || (block.at("type") == "yolo"))
        {
            assert((block.find("num") != block.end())
                   && std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
            assert((block.find("classes") != block.end())
                   && std::string("Missing 'classes' param in " + block.at("type") + " layer")
                          .c_str());
            assert((block.find("anchors") != block.end())
                   && std::string("Missing 'anchors' param in " + block.at("type") + " layer")
                          .c_str());

            TensorInfo outputTensor;
            std::string anchorString = block.at("anchors");
            while (!anchorString.empty())
            {
                int npos = anchorString.find_first_of(',');
                if (npos != -1)
                {
                    float anchor = std::stof(trim(anchorString.substr(0, npos)));
                    outputTensor.anchors.push_back(anchor);
                    anchorString.erase(0, npos + 1);
                }
                else
                {
                    float anchor = std::stof(trim(anchorString));
                    outputTensor.anchors.push_back(anchor);
                    break;
                }
            }
			
			for(size_t i = 0 ; i < outputTensor.anchors.size(); i ++)
				std::cout << outputTensor.anchors[i]<< " ";
			std::cout << std::endl;
			
            if ((m_NetworkType == "yolov3") || (m_NetworkType == "yolov3-tiny") || (m_NetworkType == "yolov4-tiny") || (m_NetworkType == "yolov4"))
            {
                assert((block.find("mask") != block.end())
                       && std::string("Missing 'mask' param in " + block.at("type") + " layer")
                              .c_str());

                std::string maskString = block.at("mask");
                while (!maskString.empty())
                {
                    int npos = maskString.find_first_of(',');
                    if (npos != -1)
                    {
                        uint mask = std::stoul(trim(maskString.substr(0, npos)));
                        outputTensor.masks.push_back(mask);
                        maskString.erase(0, npos + 1);
                    }
                    else
                    {
                        uint mask = std::stoul(trim(maskString));
                        outputTensor.masks.push_back(mask);
                        break;
                    }
                }
            }

            outputTensor.numBBoxes = outputTensor.masks.size() > 0
                ? outputTensor.masks.size()
                : std::stoul(trim(block.at("num")));
            outputTensor.numClasses = std::stoul(block.at("classes"));
            m_OutputTensors.push_back(outputTensor);
        }
    }
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

// to do
bool AutoBuildNetwork( 
    std::vector<std::map<std::string, std::string>> m_ConfigBlocks,
    std::vector<float>& weights, nvinfer1::INetworkDefinition& network) { 

    int weightPtr = 0;
    int channels = m_InputC = 3;
    auto m_InputW{INPUT_W},  m_InputH{INPUT_H};  

    nvinfer1::ITensor* data =
        network.addInput( INPUT_BLOB_NAME , nvinfer1::DataType::kFLOAT,
            nvinfer1::DimsCHW{static_cast<int>(m_InputC),
                static_cast<int>(m_InputH), static_cast<int>(m_InputW)});
    assert(data != nullptr && data->getDimensions().nbDims > 0);

    nvinfer1::ITensor* previous = data;
    std::vector<nvinfer1::ITensor*> tensorOutputs;
    uint outputTensorCount = 0;
    for (uint i = 0; i < m_ConfigBlocks.size(); ++i) {
        assert(getNumChannels(previous) == channels);
        std::string layerIndex = "(" + std::to_string(tensorOutputs.size()) + ")";
        if (m_ConfigBlocks.at(i).at("type") == "net") {
            printLayerInfo("", "layer", "     inp_size", "     out_size", "weightPtr");
        } 
		else if (m_ConfigBlocks.at(i).at("type") == "convolutional") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out;
            std::string layerType;
            if (m_ConfigBlocks.at(i).find("batch_normalize") != m_ConfigBlocks.at(i).end()) {
				if (m_ConfigBlocks.at(i).at("batch_normalize") == "1")
				{
					out = netAddConvBNActive(i, m_ConfigBlocks.at(i), weights,
						m_TrtWeights, weightPtr, channels, previous, &network);
					layerType = "conv-bn-Active";
				}
				else
				{
					out = netAddConvLinear(i, m_ConfigBlocks.at(i), weights,
						m_TrtWeights, weightPtr, channels, previous, &network);
					layerType = "conv-linear";
				}
            }
			else{
                out = netAddConvLinear(i, m_ConfigBlocks.at(i), weights,
                    m_TrtWeights, weightPtr, channels, previous, &network);
                layerType = "conv-linear";
            }
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = getNumChannels(previous); // update
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
        } 
		else if (m_ConfigBlocks.at(i).at("type") == "shortcut") {
            assert(m_ConfigBlocks.at(i).at("activation") == "linear");
            assert(m_ConfigBlocks.at(i).find("from") !=
                   m_ConfigBlocks.at(i).end());
            int from = stoi(m_ConfigBlocks.at(i).at("from"));

            std::string inputVol = dimsToString(previous->getDimensions());
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);
            nvinfer1::IElementWiseLayer* ew = network.addElementWise(
                *tensorOutputs[i - 2], *tensorOutputs[i + from - 1],
                nvinfer1::ElementWiseOperation::kSUM);
            assert(ew != nullptr);
            std::string ewLayerName = "shortcut_" + std::to_string(i);
            ew->setName(ewLayerName.c_str());
            previous = ew->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(ew->getOutput(0));
            printLayerInfo(layerIndex, "skip", inputVol, outputVol, "    -");
        } 
		else if (m_ConfigBlocks.at(i).at("type") == "yolo") {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curYoloTensor = m_OutputTensors.at(outputTensorCount);
            curYoloTensor.gridSize = prevTensorDims.d[1];
            curYoloTensor.stride = m_InputW / curYoloTensor.gridSize;
            m_OutputTensors.at(outputTensorCount).volume = curYoloTensor.gridSize
                * curYoloTensor.gridSize
                * (curYoloTensor.numBBoxes * (5 + curYoloTensor.numClasses));
            std::string layerName = "yolo_" + std::to_string(i);
            curYoloTensor.blobName = layerName;

            m_YoloTensor.push_back(previous);
            tensorOutputs.push_back(previous);

            // Dims inputdims = previous->getDimensions();
            // YoloKernel tmpYolokernel;
            // tmpYolokernel.height= inputdims.d[1];
            // tmpYolokernel.width= inputdims.d[2];

            // int masksize = m_OutputTensors.at(outputTensorCount).masks.size();
            // tmpYolokernel.everyYoloAnchors = masksize;

            // for(int i=0;i<masksize;i++)
            // {
            //     int index = (int)m_OutputTensors.at(outputTensorCount).masks[i] * 2;
            //     tmpYolokernel.anchors[2*i] = m_OutputTensors.at(outputTensorCount).anchors[index];
            //     tmpYolokernel.anchors[2*i+1] = m_OutputTensors.at(outputTensorCount).anchors[index+1];
            // }

            // m_YoloKernel.push_back(tmpYolokernel);

            std::string inputVol = dimsToString(inputdims);
            printLayerInfo(layerIndex, "yolo", inputVol, inputVol, std::to_string(weightPtr));

            ++outputTensorCount;
        } 
		else if (m_ConfigBlocks.at(i).at("type") == "region") {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curRegionTensor = m_OutputTensors.at(outputTensorCount);
            curRegionTensor.gridSize = prevTensorDims.d[1];
            curRegionTensor.stride = m_InputW / curRegionTensor.gridSize;
            m_OutputTensors.at(outputTensorCount).volume = curRegionTensor.gridSize
                * curRegionTensor.gridSize
                * (curRegionTensor.numBBoxes * (5 + curRegionTensor.numClasses));
            std::string layerName = "region_" + std::to_string(i);
            curRegionTensor.blobName = layerName;
            nvinfer1::plugin::RegionParameters RegionParameters{
                static_cast<int>(curRegionTensor.numBBoxes), 4,
                static_cast<int>(curRegionTensor.numClasses), nullptr};
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::IPluginV2* regionPlugin
                = createRegionPlugin(RegionParameters);
            assert(regionPlugin != nullptr);
            nvinfer1::IPluginV2Layer* region =
                network.addPluginV2(&previous, 1, *regionPlugin);
            assert(region != nullptr);
            region->setName(layerName.c_str());
            previous = region->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            network.markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(region->getOutput(0));
            printLayerInfo(layerIndex, "region", inputVol, outputVol, std::to_string(weightPtr));
            std::cout << "Anchors are being converted to network input resolution i.e. Anchors x "
                      << curRegionTensor.stride << " (stride)" << std::endl;
            for (auto& anchor : curRegionTensor.anchors) anchor *= curRegionTensor.stride;
            ++outputTensorCount;
        } 
		else if (m_ConfigBlocks.at(i).at("type") == "reorg") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::IPluginV2* reorgPlugin = createReorgPlugin(2);
            assert(reorgPlugin != nullptr);
            nvinfer1::IPluginV2Layer* reorg =
                network.addPluginV2(&previous, 1, *reorgPlugin);
            assert(reorg != nullptr);

            std::string layerName = "reorg_" + std::to_string(i);
            reorg->setName(layerName.c_str());
            previous = reorg->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            channels = getNumChannels(previous);
            tensorOutputs.push_back(reorg->getOutput(0));
            printLayerInfo(layerIndex, "reorg", inputVol, outputVol, std::to_string(weightPtr));
        }
        else if (m_ConfigBlocks.at(i).at("type") == "route") {
            std::string strLayers = m_ConfigBlocks.at(i).at("layers");
            std::vector<int> idxLayers;
            size_t lastPos = 0, pos = 0;
            while ((pos = strLayers.find(',', lastPos)) != std::string::npos) {
                int vL = std::stoi(trim(strLayers.substr(lastPos, pos - lastPos)));
                idxLayers.push_back (vL);
                lastPos = pos + 1;
            }
            if (lastPos < strLayers.length()) {
                std::string lastV = trim(strLayers.substr(lastPos));
                if (!lastV.empty()) {
                    idxLayers.push_back (std::stoi(lastV));
                }
            }
            assert (!idxLayers.empty());
            std::vector<nvinfer1::ITensor*> concatInputs;
            for (int idxLayer : idxLayers) {
                if (idxLayer < 0) {
                    idxLayer = tensorOutputs.size() + idxLayer;
                }
                assert (idxLayer >= 0 && idxLayer < (int)tensorOutputs.size());
                concatInputs.push_back (tensorOutputs[idxLayer]);
            }
            nvinfer1::IConcatenationLayer* concat;
            if(m_ConfigBlocks.at(i).find("groups") != m_ConfigBlocks.at(i).end())
            {
                assert(m_ConfigBlocks.at(i).find("group_id") != m_ConfigBlocks.at(i).end());
                int groups =  std::stoi(m_ConfigBlocks.at(i).at("groups"));
                int group_id = std::stoi(m_ConfigBlocks.at(i).at("group_id"));
                std::vector<nvinfer1::ITensor*> group_concatInputs;
                for(auto concatInput : concatInputs)
                {
                    Dims out_shape = concatInput->getDimensions();
                    ISliceLayer* tmp= network.addSlice(*concatInput, Dims3{group_id*out_shape.d[0]/groups, 0, 0}, Dims3{out_shape.d[0]/groups, out_shape.d[1],out_shape.d[2]}, Dims3{1, 1, 1});
                    group_concatInputs.push_back(tmp->getOutput(0));
                }
                concat=network.addConcatenation(group_concatInputs.data(), group_concatInputs.size());
            }else {
                concat=network.addConcatenation(concatInputs.data(), concatInputs.size());
            }

            assert(concat != nullptr);
            std::string concatLayerName = "route_" + std::to_string(i - 1);
            concat->setName(concatLayerName.c_str());

            concat->setAxis(0);
            previous = concat->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());

            channels = getNumChannels(previous);
            tensorOutputs.push_back(concat->getOutput(0));
            printLayerInfo(layerIndex, "route", "        -", outputVol,
                           std::to_string(weightPtr));
        } 
		else if (m_ConfigBlocks.at(i).at("type") == "upsample") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddUpsample(i - 1, m_ConfigBlocks[i],
                weights, m_TrtWeights, channels, previous, &network);
            previous = out->getOutput(0);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
        } 
		else if (m_ConfigBlocks.at(i).at("type") == "maxpool") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out =
                netAddMaxpool(i, m_ConfigBlocks.at(i), previous, &network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "maxpool", inputVol, outputVol, std::to_string(weightPtr));
        }
		else if (m_ConfigBlocks.at(i).at("type") == "batchnorm") {
			std::string inputVol = dimsToString(previous->getDimensions());
			nvinfer1::ILayer* out;
			std::string layerType = "batchnorm";
			out = netAddBatchNorm(i, m_ConfigBlocks.at(i), weights,
				m_TrtWeights, weightPtr, channels, previous, &network);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
		}
		else if (m_ConfigBlocks.at(i).at("type") == "activation") {
			std::string inputVol = dimsToString(previous->getDimensions());
			nvinfer1::ILayer* out;
			std::string layerType = "activation";
			out = netAddActivation(i, m_ConfigBlocks.at(i), weights,
				m_TrtWeights, weightPtr, channels, previous, &network);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
		}
        else
        {
            std::cout << "Unsupported layer type --> \""
                      << m_ConfigBlocks.at(i).at("type") << "\"" << std::endl;
            assert(0);
        }
    }

    ITensor** inputTensors_yolo = new ITensor*;
    for(int i=0;i<m_YoloTensor.size();i++) inputTensors_yolo[i]=m_YoloTensor[i]; 

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    
    yolo->getOutput(0)->setName( OUTPUT_BLOB_NAME );
    network->markOutput(*yolo->getOutput(0));

 
    if ((int)weights.size() != weightPtr)
    {
        std::cout << "Number of unused weights left : " << (int)weights.size() - weightPtr << std::endl;
        assert(0);
    }

    std::cout << "Output yolo blob names :" << std::endl;
    for (auto& tensor : m_OutputTensors) {
        std::cout << tensor.blobName << std::endl;
    }

    int nbLayers = network.getNbLayers();
    std::cout << "Total number of yolo layers: " << nbLayers << std::endl;

    return true;
}
 
// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(
    unsigned int maxBatchSize, IBuilder* builder, 
    IBuilderConfig* config, DataType dt ) {
    
    std::vector<std::map<std::string, std::string> >  
        ConfigBlocks = parseConfigFile( PATH_CFG );  
    std::vector<float> weights = loadWeights( PATH_WTS, "yolov4" );

    std::cout << "Building Yolo network ..." << std::endl;
    INetworkDefinition* network = builder->createNetworkV2(0U);
 
    auto i = AutoBuildNetwork(  weights, , network ); 

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
    std::cout << "Building tensorrt engine by FP16, please wait for a while..." << std::endl;
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, PATH_CALIBRATOR_IMGS , "./int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
    std::cout << "Building tensorrt engine by INT8, please wait for a while..." << std::endl;
#else
    std::cout << "Building tensorrt engine by FLOAT32, please wait for a while..." << std::endl;
#endif
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

int main(){
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    const char* PATH_CFG = argv[2] ;
    const char* PATH_WTS = argv[3] ; 
    const char* PATH_ENGINE = argv[4] ;

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};

        // Create builder
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network, then set the outputs and create an engine
        ICudaEngine* engine = AutoBuildNetwork( BATCH_SIZE, builder, config, DataType::kFLOAT);
         
        assert(engine != nullptr); 
        // Serialize the engine
        modelStream = engine->serialize();
 
        assert(modelStream != nullptr);
        std::ofstream p( PATH_ENGINE , std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
         
        // Close everything down
        engine->destroy();
        builder->destroy();
        config->destroy();
        modelStream->destroy();
        return 0;
    }else std::cout <<"\nUsage:\n   run.exe -s XX.cfg XX.weights XX.engine \n";  
}