## 1. 环境配置

```
conda create -n metarl python=3.8
conda activate metarl
pip install torch
pip install numpy
pip install pybullet
pip install onnx
pip install onnxruntime
pip install gym
pip install attrs --upgrade
pip install scipy
```

## 2. 运行训练好的网络

```
cd infer_policy
python sim_policy_onnx.py
```
