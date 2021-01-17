# tensorrt 7.0.0.11-yolo

Now, just support Ubuntu

Support your custom cfg network
Support carema mp4 txt folder

Don't need onnx, directly transport .cfg and .weights to Tensorrt engine

updating !

Computations supported now :
    conv
    reg
    maxpool
    route:  group_id    groups  
    mish
    yolo   

to do :

    add int8_calibrator to this project.



## Excute:
```
1. clone.
2.set CMakeList.txt tensorrt path, opencv path.
4.set .cfg input_w and input_h,due to tensorrt upsample , input_w shuld equal input_h
6.mkdir build. && cd build && cmake .. && make 

7.run ./yolo -s to build yolo engine

    arg option :
        input : cfg and weight files

    output : the engine same to the cfg path

7.run ./yolo -d to start detect

    arg option as following
        0 : using camera;
        1 : using txt include abs_path
        2 : using folder contain images
        3 : using .mp4 

    output : the result of detection at __file__ path 

```
## set FP16 or FP32
- FP16/FP32 can be selected by the macro `USE_FP16` 
