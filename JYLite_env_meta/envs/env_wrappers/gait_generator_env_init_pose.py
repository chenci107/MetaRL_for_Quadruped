from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from gym import spaces
from JYLite_env_meta.robots.JYLite import INIT_MOTOR_ANGLES


class GaitGeneratorWrapperEnv_InitPose(object):
    def __init__(self,gym_env):
        self._gym_env = gym_env
        self.timesteps = 0

        action_high = np.array([0.2,0.7,0.7] * 4)
        action_low = np.array([-0.2,-0.7,-0.7] * 4)
        self.action_space = spaces.Box(action_low,action_high,dtype=np.float32)

        self._pose = np.array(
            [INIT_MOTOR_ANGLES[0],INIT_MOTOR_ANGLES[1],INIT_MOTOR_ANGLES[2],
             INIT_MOTOR_ANGLES[3],INIT_MOTOR_ANGLES[4],INIT_MOTOR_ANGLES[5],
             INIT_MOTOR_ANGLES[6],INIT_MOTOR_ANGLES[7],INIT_MOTOR_ANGLES[8],
             INIT_MOTOR_ANGLES[9],INIT_MOTOR_ANGLES[10],INIT_MOTOR_ANGLES[11]])

    def __getattr__(self, attr):
        return getattr(self._gym_env,attr)

    def reset(self,**kwargs):
        self.timesteps = 0
        self.obs,info = self._gym_env.reset(**kwargs)
        return self.obs[0],info

    def step(self,action):
        self.timesteps += 1
        new_action = action + self._pose

        self.obs,reward,done,info = self._gym_env.step(new_action)

        return self.obs[0],reward,done,info
