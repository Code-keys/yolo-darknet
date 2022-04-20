#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>

#include "demo_plugin.h"

namespace nvinfer1
{
 
    PluginFieldCollection DemoPluginCreator::mFC{};
    std::vector<PluginField> DemoPluginCreator::mPluginAttributes; 


    __device__ float Demo_kernel_helper(float x){ return (2/(1 + expf(-2*x)) - 1); }
 
    __global__ void Demo_kernel(const float *input, int num_elem, float *output) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;
       
        /**************************** 修改 this   ****************************/
        // to do

        output[idx] = input[idx] * Demo_kernel_helper( input[idx] );
    }
 
    DemoPlugin::DemoPlugin() { }

    DemoPlugin::~DemoPlugin()  {
        delete this->InputDimensions;
     }

    // create the plugin at runtime from a byte stream
    DemoPlugin::DemoPlugin(const void* data, size_t length)
    {
        assert(length == sizeof(input_size_));
        input_size_ = *reinterpret_cast<const int*>(data);
    }

    void DemoPlugin::serialize(void* buffer) const
    {
        *reinterpret_cast<int*>(buffer) = input_size_;
    }
  
    Dims DemoPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        // Input dimensions
        input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        this->InputDimensions = new Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
        
        // Output dimensions
        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    } 

    // Clone the plugin
    IPluginV2IOExt* DemoPlugin::clone() const
    {
        DemoPlugin *p = new DemoPlugin();
        p->input_size_ = input_size_;
        p->InputDimensions = InputDimensions;
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }
 
    void DemoPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
       
        /**************************** 修改 this   ****************************/
        // to do
        Demo_kernel<<<grid_size, block_size>>>(
            inputs[0], input_size_ * batchSize, output );
    }




    DemoPluginCreator::DemoPluginCreator() {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    } 
    IPluginV2IOExt* DemoPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) {
        DemoPlugin* obj = new DemoPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    } 
    IPluginV2IOExt* DemoPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) {
        // This object will be deleted when the network is destroyed, which will
        // call DemoPlugin::destroy()
        DemoPlugin* obj = new DemoPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    } 
};