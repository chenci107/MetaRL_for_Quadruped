## 1. 环境配置

```
conda create -n metarl python=3.8
pip install torch
pip install numpy
pip install pybullet
pip install onnx
pip install onnxruntime
pip install gym
```

## 2. 运行训练好的网络

```
cd infer_policy
python sim_policy_onnx.py
```
