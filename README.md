![image](https://github.com/chenci107/MetaRL_for_Quadruped/assets/48233618/d50c302c-ec7f-47a3-aa11-2aeb9b6321c7)
## Meta Reinforcement Learning of Locomotion Policy for Quadruped Robots with Motor Stuck

---

This is the official repository for the paper: Meta Reinforcement Learning of Locomotion Policy for Quadruped Robots with Motor Stuck

### Architecture
---

![framework](https://github.com/chenci107/MetaRL_for_Quadruped/assets/48233618/6a499416-594c-47af-a3ee-0a93a0a10de6)


### Requirements
---

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

### Running 

```
cd infer_policy
python sim_policy_onnx.py
```
