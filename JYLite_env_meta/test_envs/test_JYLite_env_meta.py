from JYLite_env_meta import *
from JYLite_env_meta.robots import robot_config
import numpy as np
import os
import inspect
import time
#
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)


SENSOR_MODE = {'dis':1, 'motor':1, 'imu':1, 'contact':1, 'footpose':0, 'dynamic_vec':0, 'force_vec':0, 'noise':0,'add_bezier':1}

env = JYLiteGymEnv(task="ground",render=True,gait=3,enable_disabled=False,random_dynamic=False)
joint_index = np.array([2])

for j in joint_index:
    env.reset_task(idx=j)
    observation,info = env.reset()  # 1,2,4,5,7,8,10,11
    for j in range(1000):
        action = np.zeros(12)
        next_obs,reward,done,info = env.step(action)
        print('The observation dim is:',next_obs.shape)
    print("========== DONE =========")

    # print('The next_obs is:',next_obs)
    # print('The reward is:',reward)
    # print('The done is:',done)
    # print('The info is:',info)