import gym
import numpy as np

from JYLite_env_meta.envs import env_builder
from JYLite_env_meta.robots import JYLite
from JYLite_env_meta.robots import robot_config
from JYLite_env_meta.envs.env_wrappers.MonitorEnv import EnvWrapper,Param_Dict,Random_Param_Dict

SENSOR_MODE = {"dis":1,"motor":1,"imu":1,"contact":1,"footpose":0,"linear_acc":0,"add_bezier":0}

class JYLiteGymEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,
                 render = False,
                 on_rack = False,
                 sensor_mode = SENSOR_MODE,
                 gait = "straight",
                 normal = 0,
                 random_dynamic = False,
                 reward_param = Param_Dict,
                 vel_d = 0.5,
                 reward_p = 1.0,
                 motor_control_mode = robot_config.MotorControlMode.POSITION,
                 dynamic_param={},
                 enable_disabled=True,
                 filter = False,
                 **kwargs
                ):
        self._env = env_builder.build_regular_env(
            JYLite.JYLITE,
            motor_control_mode=motor_control_mode,
            gait=gait,
            normal=normal,
            enable_rendering=render,
            sensor_mode=sensor_mode,
            random=random_dynamic,
            on_rack=on_rack,
            param=dynamic_param,
            enable_disabled=enable_disabled)

        self._env = EnvWrapper(
            env=self._env,
            param=reward_param,
            sensor_mode=sensor_mode,
            normal=normal,
            enable_action_filter=filter,
            vel_d=vel_d,
            reward_p=reward_p)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.sensor_mode = sensor_mode
        self._max_episode_steps = 1000

    def step(self,action,**kwargs):
        return self._env.step(action,**kwargs)

    def reset(self,**kwargs):
        return self._env.reset(**kwargs)

    def close(self):
        self._env.close()

    def render(self, mode="human"):
        if (mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        return self._env.render(mode)

    def rest_task(self,**kwargs):
        return self._env.reset_task(**kwargs)

    def get_all_task_idx(self):
        return np.array([1,2,4,5,7,8,10,11])

    def __getattr__(self, attr):
        return getattr(self._env, attr)


