Here is how to use tensorrt to accelerate roma:

1.transfer onnx mode to engine:
(you need to change the path to yours)
trtexec --onnx=/home/percy/workspace/roma_trt/onnx_ori/roma_core.onnx  --saveEngine=/home/percy/workspace/roma_trt/roma_core_ori.engine --minShapes=image_a:1x3x504x504,image_b:1x3x504x504  --optShapes=image_a:1x3x504x504,image_b:1x3x504x504  --maxShapes=image_a:1x3x504x504,image_b:1x3x504x504  --fp16 --verbose

2.testtrt.py:use the engine file to Inference 

3.roma_cpp: a cpp version using tensorrt to Inference