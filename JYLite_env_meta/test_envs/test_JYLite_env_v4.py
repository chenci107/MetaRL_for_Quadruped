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


SENSOR_MODE = {'dis':1, 'motor':1, 'imu':1, 'contact':1, 'ETG':1, 'ETG_obs':0, 'footpose':0, 'dynamic_vec':0, 'force_vec':0, 'noise':0}

env = JYLiteGymEnv(task="ground",render=True,gait=3,enable_disabled=True)
joint_index = np.array([1,2,4,5,7,8,10,11])

for j in joint_index:
    observation,info = env.reset(dis_joint=j)  # 1,2,4,5,7,8,10,11
    # print('The observation is:',observation)
    # observation_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # print('The observation_dim is:',observation_dim)
    # print('The action_dim is:',action_dim)
    for j in range(1000):
        # action = env.action_space.sample()
        # print('The action is:',action)
        action = np.zeros(12)
        next_obs,reward,done,info = env.step(action)
        # time.sleep(1/500.)

    # print('The next_obs is:',next_obs)
    # print('The reward is:',reward)
    # print('The done is:',done)
    # print('The info is:',info)
