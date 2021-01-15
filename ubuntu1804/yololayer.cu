#include "yololayer.h"

namespace nvinfer1
{
    // build network 阶段，创建模型阶段，调用的接口
    YoloLayerPlugin::YoloLayerPlugin(const PluginFieldCollection& fc)
    {
        void* tmpvoid;
        const PluginField* fields = fc.fields;
        for (int i = 0; i < fc.nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "numclass")){
                mClassCount = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "input_w")){
                mInput_w= *(static_cast<const int*>(fields[i].data));
            }else if(!strcmp(attrName, "input_h")){
                mInput_h = *(static_cast<const int*>(fields[i].data));
            }else if(!strcmp(attrName, "numyololayers")){
                mNumYoloLayers = *(static_cast<const int*>(fields[i].data));
            }else if(!strcmp(attrName, "m_YoloKernel")){
                assert(fields[i].type == PluginFieldType::kUNKNOWN);
                tmpvoid = const_cast<void*>(fields[i].data);
            }
        }
        // 解析 yolo层
        mYoloKernel = *(std::vector<YoloKernel> *)tmpvoid;
        std::cout<<"mYoloKernel.size()"<<mYoloKernel.size()<<std::endl;
    }
    
    YoloLayerPlugin::~YoloLayerPlugin()
    {}
    // create the plugin at runtime from a byte stream，反序列化，调用的接口，生成模型
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mNumYoloLayers);
        read(d, mInput_h);
        read(d, mInput_w);
        mYoloKernel.resize(mNumYoloLayers);
        auto kernelSize = mNumYoloLayers*sizeof(YoloKernel);
        memcpy(mYoloKernel.data(),d,kernelSize);
        d += kernelSize;
        assert(d == a + length);
    }
    // 序列化模型，即保存模型，将插件内用到的参数保存到模型中
    void YoloLayerPlugin::serialize(void* buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mNumYoloLayers);
        write(d, mInput_h);
        write(d, mInput_w);
        auto kernelSize = mNumYoloLayers*sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(),kernelSize);
        d += kernelSize;
        assert(d == a + getSerializationSize());
    }
    // 保存模型，序列化阶段，计算插件需要保存的数据长度
    size_t YoloLayerPlugin::getSerializationSize() const
    {  
        int size  = sizeof(mInput_w) +sizeof(mInput_h)+
                sizeof(mClassCount) + sizeof(mThreadCount) +
                sizeof(mNumYoloLayers)  + sizeof(YoloKernel) * mYoloKernel.size();
        return size;
    }

    int YoloLayerPlugin::initialize()
    { 
        return 0;
    }
    
    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = max_output_box * sizeof(Detection) / sizeof(float);
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

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() {}

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
    IPluginV2IOExt* YoloLayerPlugin::clone() const
    {

        YoloLayerPlugin *p = new YoloLayerPlugin(*this);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }
    // 核函数 sigmoid
    __device__ float Logist(float data){ return 1.0f / (1.0f + __expf(-data)); };
    // cuda 调用接口
    __global__ void CalDetection(const float *input, float *output,int noElements, 
            int yoloWidth,int yoloHeight,const float* anchors,int classes,
                                 int outputElem,int input_w,int input_h,
                                 float ignore_thresh,int every_yolo_anchors,int max_out_put_bbox_count) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * every_yolo_anchors);

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob*box_prob < ignore_thresh) continue;

            float *res_count = output + bnIdx*outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= max_out_put_bbox_count) return;
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid]))* input_w/ yoloWidth;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid]))* input_h/ yoloHeight;
            det->bbox[2] = __expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = __expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize)
    {
        // 每一层的输出大小长度，
        int outputElem = 1 + max_output_box * sizeof(Detection) / sizeof(float);
        // 根据batchsize调整输出的output 内存大小，初始化为0， 以最小内存单位字节为长度
        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        void* devAnchor;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            // yolo 每一层的参数
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            int every_yolo_anchor_num = yolo.everyYoloAnchors;
            size_t AnchorLen = sizeof(float)* yolo.everyYoloAnchors*2;
            CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));
            CUDA_CHECK(cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaFree(devAnchor));
            // 调用cuda接口，<调用的block数量，每一个block中的thread数量>
            CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, yolo.width, yolo.height,
                 (float *)devAnchor, mClassCount ,outputElem,mInput_w, mInput_w,
                 mIgnore_thresh,every_yolo_anchor_num,max_output_box);
        }
    }

    // 插件标准调用接口，enqueue
    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    YoloPluginCreator::YoloPluginCreator()
    {
    }

    const char* YoloPluginCreator::getPluginName() const
    {
            return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()
    {
            return 0;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin(*fc);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
