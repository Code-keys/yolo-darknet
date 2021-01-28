# if this help to you,please star star star
# tensorrt7-yolo
Now, just support Ubuntu, windows will be soon.

Support yolov3 yolov3-tiny yolov4 yolov4-tiny，even your custom cfg networ 

if you train from darknet(AB), it usually can support.

Don't need onnx, directly transport .cfg and .weights to Tensorrt engine

this project borrow from [Deepstream](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/restructure) and [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

## Excute:
```
1. clone.
2.set CMakeList.txt tensorrt path, opencv path.
3.main.cpp, set diffenrt cfg and weights
4.set .cfg input_w and input_h,due to tensorrt upsample , input_w shuld equal input_h
5.copy .cfg and .weights file to folder 
6.mkdir build.  
7.cd build && cmake .. && make 
7.run ./yolo -s to build yolo engine
7.run ./yolo -d to start detect
```
## set FP16 or FP32
- FP16/FP32 can be selected by the macro `USE_FP16` 
