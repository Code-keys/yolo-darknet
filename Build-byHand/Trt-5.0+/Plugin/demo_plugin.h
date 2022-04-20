#ifndef _DEMO_PLUGIN_H
#define _DEMO_PLUGIN_H

#include <string>
#include <vector>
#include "NvInfer.h"
 

namespace nvinfer1
{
    /**************************** 修改 this name ****************************/
    class DemoPlugin: public IPluginV2IOExt
    {
        public:
            explicit DemoPlugin();
            DemoPlugin(const void* data, size_t length);

            ~DemoPlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            int initialize() override{return 0;};

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override{
                //assert(batchSize == 1);
                //GPU
                //CUDA_CHECK(cudaStreamSynchronize(stream));
                forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
                return 0; 
            };

            virtual size_t getSerializationSize() const override{return sizeof(input_size_);};

            virtual void serialize(void* buffer) const override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override{return "Demo";};

            const char* getPluginVersion() const override{return "1";};

            void destroy() override{ delete this; };

            IPluginV2IOExt* clone() const override;

            void setPluginNamespace(const char* pluginNamespace) override{mPluginNamespace = pluginNamespace;};

            const char* getPluginNamespace() const override{return mPluginNamespace;};
            
            // Return the DataType of the plugin output at the requested index
            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {  return DataType::kFLOAT;};
            // Return true if output tensor is broadcast across a batch.
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override{ return false; };
            // Return true if plugin can use input that is broadcast across batch without replication.
            bool canBroadcastInputAcrossBatch(int inputIndex) const override { return false; };
            // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override{};

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override {};
            // Detach the plugin object from its execution context.
            void detachFromContext() override;

            int input_size_;
            Dims3* InputDimensions;

        private:
            void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
            int thread_count_ = 256;
            const char* mPluginNamespace;
 
            /**************************** 修改 this data ****************************/
            // to do
 


 

    };

    /**************************** 修改 this name ****************************/
    class DemoPluginCreator : public IPluginCreator
    {
        public:
            DemoPluginCreator();

            ~DemoPluginCreator() override = default;
    /**************************** 修改 this name ****************************/
            const char* getPluginName() const override{ return "Demo_Plugin"; }; 

            const char* getPluginVersion() const override{ return "1"; };

            const PluginFieldCollection* getFieldNames() override{ return &mFC; };

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

            const char* getPluginNamespace() const override { return mNamespace.c_str(); }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
    
    /**************************** 修改 this name ****************************/
    REGISTER_TENSORRT_PLUGIN(DemoPluginCreator);
};
#endif 
