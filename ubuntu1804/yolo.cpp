
#include "yolo.h"

#include <fstream>
#include <iomanip>
#include <iterator>

using namespace nvinfer1;

REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

Yolo::Yolo(const NetworkInfo& networkInfo)
    : m_NetworkType(networkInfo.networkType),           // yolov3
      m_ConfigFilePath(networkInfo.configFilePath),     // yolov3.cfg
      m_WtsFilePath(networkInfo.wtsFilePath),           // yolov3.weights
      m_DeviceType(networkInfo.deviceType),             // kDLA, kGPU
      m_InputBlobName(networkInfo.inputBlobName),       // data
      m_InputH(0),
      m_InputW(0),
      m_InputC(0),
      m_InputSize(0)
{}

Yolo::~Yolo()
{
    destroyNetworkUtils();
}

nvinfer1::ICudaEngine *Yolo::createEngine (nvinfer1::IBuilder* builder)
{
    assert (builder);

//    std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
//    std::vector<nvinfer1::Weights> trtWeights;

    nvinfer1::INetworkDefinition *network = builder->createNetwork();
    if (parseModel(*network) != NVDSINFER_SUCCESS) {
        network->destroy();
        return nullptr;
    }

    // Build the engine
    std::cout << "Building the TensorRT Engine..." << std::endl;
    nvinfer1::ICudaEngine * engine = builder->buildCudaEngine(*network);
    if (engine) {
        std::cout << "Building complete!" << std::endl;
    } else {
        std::cerr << "Building engine failed!" << std::endl;
    }

    // destroy
    network->destroy();
    return engine;
}

NvDsInferStatus Yolo::parseModel(nvinfer1::INetworkDefinition& network) {
    destroyNetworkUtils();

    m_ConfigBlocks = parseConfigFile(m_ConfigFilePath);
    parseConfigBlocks();

    std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
    // build yolo network
    std::cout << "Building Yolo network..." << std::endl;
    NvDsInferStatus status = buildYoloNetwork(weights, network);

    if (status == NVDSINFER_SUCCESS) {
        std::cout << "Building yolo network complete!" << std::endl;
    } else {
        std::cerr << "Building yolo network failed!" << std::endl;
    }

    return status;
}

NvDsInferStatus Yolo::buildYoloNetwork(
    std::vector<float>& weights, nvinfer1::INetworkDefinition& network) {

    // 清理yolo层
    m_YoloKernel.clear();

    int weightPtr = 0;
    int channels = m_InputC;

    nvinfer1::ITensor* data =
        network.addInput(m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
            nvinfer1::DimsCHW{static_cast<int>(m_InputC),
                static_cast<int>(m_InputH), static_cast<int>(m_InputW)});
    assert(data != nullptr && data->getDimensions().nbDims > 0);

    nvinfer1::ITensor* previous = data;
    std::vector<nvinfer1::ITensor*> tensorOutputs;
    uint outputTensorCount = 0;

    // build the network using the network API
    for (uint i = 0; i < m_ConfigBlocks.size(); ++i) {
        // check if num. of channels is correct
        assert(getNumChannels(previous) == channels);
        std::string layerIndex = "(" + std::to_string(tensorOutputs.size()) + ")";

        if (m_ConfigBlocks.at(i).at("type") == "net") {
            printLayerInfo("", "layer", "     inp_size", "     out_size", "weightPtr");
        } else if (m_ConfigBlocks.at(i).at("type") == "convolutional") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out;
            std::string layerType;
            // check if batch_norm enabled
            if (m_ConfigBlocks.at(i).find("batch_normalize") != m_ConfigBlocks.at(i).end()) {

                out = netAddConvBNActive(i, m_ConfigBlocks.at(i), weights,
                                         m_TrtWeights, weightPtr, channels, previous, &network);
                layerType = "conv-bn-Active";
            }else{
                out = netAddConvLinear(i, m_ConfigBlocks.at(i), weights,
                    m_TrtWeights, weightPtr, channels, previous, &network);
                layerType = "conv-linear";
            }
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = getNumChannels(previous);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
        } else if (m_ConfigBlocks.at(i).at("type") == "shortcut") {
            assert(m_ConfigBlocks.at(i).at("activation") == "linear");
            assert(m_ConfigBlocks.at(i).find("from") !=
                   m_ConfigBlocks.at(i).end());
            int from = stoi(m_ConfigBlocks.at(i).at("from"));

            std::string inputVol = dimsToString(previous->getDimensions());
            // check if indexes are correct
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
        } else if (m_ConfigBlocks.at(i).at("type") == "yolo") {
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

            // 添加yolo层
            m_YoloTensor.push_back(previous);
            tensorOutputs.push_back(previous);

            // 调整 yolo层的信息
            Dims inputdims = previous->getDimensions();
            YoloKernel tmpYolokernel;
            tmpYolokernel.height= inputdims.d[1];
            tmpYolokernel.width= inputdims.d[2];
            // 添加yolo anchors
            int masksize = m_OutputTensors.at(outputTensorCount).masks.size();
            tmpYolokernel.everyYoloAnchors = masksize;

            for(int i=0;i<masksize;i++)
            {
                int index = (int)m_OutputTensors.at(outputTensorCount).masks[i] * 2;
                tmpYolokernel.anchors[2*i] = m_OutputTensors.at(outputTensorCount).anchors[index];
                tmpYolokernel.anchors[2*i+1] = m_OutputTensors.at(outputTensorCount).anchors[index+1];
            }

            // 全局
            m_YoloKernel.push_back(tmpYolokernel);

            std::string inputVol = dimsToString(inputdims);
            printLayerInfo(layerIndex, "yolo", inputVol, inputVol, std::to_string(weightPtr));

            ++outputTensorCount;
        } else if (m_ConfigBlocks.at(i).at("type") == "region") {
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
        } else if (m_ConfigBlocks.at(i).at("type") == "reorg") {
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
        // route layers (single or concat)
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
                int gorups =  std::stoi(m_ConfigBlocks.at(i).at("groups"));
                int group_id = std::stoi(m_ConfigBlocks.at(i).at("group_id"));
                std::vector<nvinfer1::ITensor*> group_concatInputs;
                for(auto concatInput : concatInputs)
                {
                    Dims out_shape = concatInput->getDimensions();
                    ISliceLayer* tmp= network.addSlice(*concatInput,Dims3{out_shape.d[0]/2,0,0},Dims3{out_shape.d[0]/2,out_shape.d[1],out_shape.d[2]},Dims3{1,1,1});
                    group_concatInputs.push_back(tmp->getOutput(0));
                }
                concat=network.addConcatenation(group_concatInputs.data(), group_concatInputs.size());
            }else {
                concat=network.addConcatenation(concatInputs.data(), concatInputs.size());
            }

            assert(concat != nullptr);
            std::string concatLayerName = "route_" + std::to_string(i - 1);
            concat->setName(concatLayerName.c_str());
            // concatenate along the channel dimension
            concat->setAxis(0);
            previous = concat->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            // set the output volume depth
            channels
                = getNumChannels(previous);
            tensorOutputs.push_back(concat->getOutput(0));
            printLayerInfo(layerIndex, "route", "        -", outputVol,
                           std::to_string(weightPtr));
        } else if (m_ConfigBlocks.at(i).at("type") == "upsample") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddUpsample(i - 1, m_ConfigBlocks[i],
                weights, m_TrtWeights, channels, previous, &network);
            previous = out->getOutput(0);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
        } else if (m_ConfigBlocks.at(i).at("type") == "maxpool") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out =
                netAddMaxpool(i, m_ConfigBlocks.at(i), previous, &network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "maxpool", inputVol, outputVol, std::to_string(weightPtr));
        }
        else
        {
            std::cout << "Unsupported layer type --> \""
                      << m_ConfigBlocks.at(i).at("type") << "\"" << std::endl;
            assert(0);
        }
    }

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    assert(m_YoloKernel.size() == outputTensorCount);

    // plugin filed 数量
    int numyololayers = m_YoloKernel.size();

    // 假定每个yolo输出层class相等
    int numclass = m_OutputTensors[0].numClasses;
    int input_w = m_InputW;
    int input_h = m_InputH;

    std::vector<PluginField> mPluginAttributes1 = {
        PluginField("numclass", &numclass, PluginFieldType::kINT32, 1),
        PluginField("input_w", &input_w, PluginFieldType::kINT32, 1),
        PluginField("input_h", &input_h, PluginFieldType::kINT32, 1),
        PluginField("numyololayers", &numyololayers, PluginFieldType::kINT32, 1),
        PluginField("m_YoloKernel", &m_YoloKernel, PluginFieldType::kUNKNOWN, numyololayers),
    };
    PluginFieldCollection mFC1;
    mFC1.nbFields = mPluginAttributes1.size();
    mFC1.fields = mPluginAttributes1.data();
    IPluginV2 * yoloplugin = creator->createPlugin(creator->getPluginName(), &mFC1);

    ITensor** inputTensors_yolo = new ITensor*;
    for (int i = 0; i<m_YoloTensor.size();i++)
    {
        inputTensors_yolo[i] = m_YoloTensor[i];
    }

    auto yolo = network.addPluginV2(inputTensors_yolo, 3, *yoloplugin);

    previous = yolo->getOutput(0);
    assert(previous != nullptr);
    previous->setName("prob");
    std::string outputVol = dimsToString(previous->getDimensions());
    network.markOutput(*previous);

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

    return NVDSINFER_SUCCESS;
}

std::vector<std::map<std::string, std::string>>
Yolo::parseConfigFile (const std::string cfgFilePath)
{
    assert(fileExists(cfgFilePath));
    std::ifstream file(cfgFilePath);
    assert(file.good());
    std::string line;
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;

    while (getline(file, line))
    {
        if (line.size() == 0) continue;
        if (line.front() == '#') continue;
        line = trim(line);
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

void Yolo::parseConfigBlocks()
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
//            assert(m_InputW == m_InputH);
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

void Yolo::destroyNetworkUtils() {
    // deallocate the weights
    for (uint i = 0; i < m_TrtWeights.size(); ++i) {
        if (m_TrtWeights[i].count > 0)
            free(const_cast<void*>(m_TrtWeights[i].values));
    }
    m_TrtWeights.clear();
}

