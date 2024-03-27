from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class NODisabledJointWrapper(object):
    def __init__(self,gym_env,dis_joint=None):
        self._gym_env = gym_env
        self.dis_joint = None
        self.is_infer = False

    # important
    def __getattr__(self, attr):
        return getattr(self._gym_env,attr)

    def reset(self,**kwargs):
        obs,info = self._gym_env.reset(**kwargs)
        return obs,info

    def step(self,action):
        obs,reward,done,info = self._gym_env.step(action)
        return obs,reward,done,info

    def reset_task(self,**kwargs):
        obs,info = self._gym_env.reset(**kwargs)
        return obs,info

