from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import copy
from JYLite_env_meta.envs.GaitGenerator.Bezier import BezierGait
from JYLite_env_meta.envs.OpenLoopSM.SpotOL import BezierStepper
from gym import spaces
from JYLite_env_meta.robots.JYLite import INIT_MOTOR_ANGLES

class GaitGeneratorWrapperEnv(object):
    def __init__(self,gym_env,vel=0.5,gait_mode=3,add_bezier=False):
        self._gym_env = gym_env
        self.bz_step = BezierStepper(dt=-self._gym_env.env_time_step,StepVelocity=vel)
        self.bzg = BezierGait(dt=self._gym_env.env_time_step)

        self.timesteps = 0
        self.vel = vel
        self.gait_mode = gait_mode

        if self.gait_mode == "straight":
            action_high = np.array([0.1, 0.5, 0.4] * 4)
            action_low = np.array([-0.1, -0.3, -0.6] * 4)
            self.action_space = spaces.Box(action_low,action_high,dtype=np.float32)

        self.add_bezier = add_bezier
        if self.add_bezier == True:
            sensor_shape = self._gym_env.observation_space.high.shape[0]
            obs_h = np.array([1] * (sensor_shape + 12))
            obs_l = np.array([0] * (sensor_shape + 12))
            self.observation_space = spaces.Box(obs_l, obs_h, dtype=np.float32)
        else:
            self.observation_space = self._gym_env.observation_space

    def __getattr__(self, attr):
        return getattr(self._gym_env,attr)

    def reset(self,**kwargs):
        self.timesteps = 0
        self.obs,info = self._gym_env.reset(**kwargs)
        self.bz_step = BezierStepper(dt=self._gym_env.env_time_step,StepVelocity=self.vel)
        self.bzg = BezierGait(dt=self._gym_env.env_time_step)
        T_b0_ = copy.deepcopy(self._gym_env.robot.GetFootPositionsInBaseFrame())
        Tb_d = {}
        Tb_d["FL"] = T_b0_[0, :]
        Tb_d["FR"] = T_b0_[1, :]
        Tb_d["BL"] = T_b0_[2, :]
        Tb_d["BR"] = T_b0_[3, :]
        self.T_b0 = Tb_d
        _, _, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, _ = self.bz_step.return_bezier_params()

        if self.add_bezier == False:
            return self.obs[0], info

        else:
            action_ref = np.array([INIT_MOTOR_ANGLES[0],INIT_MOTOR_ANGLES[1],INIT_MOTOR_ANGLES[2]] * 4)
            obs = np.concatenate([self.obs[0],action_ref],axis=0)
            return obs, info


    def step(self,action):
        self.timesteps += 1

        if action is None:
            raise ValueError('action cannot be None')
        pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = self.bz_step.StateMachine()
        ClearanceHeight = 0.05
        StepLength = np.clip(StepLength, self.bz_step.StepLength_LIMITS[0],self.bz_step.StepLength_LIMITS[1])           # StepLength:0.04
        StepVelocity = np.clip(StepVelocity,self.bz_step.StepVelocity_LIMITS[0],self.bz_step.StepVelocity_LIMITS[1])    # StepVelocity:0.5
        LateralFraction = np.clip(LateralFraction, self.bz_step.LateralFraction_LIMITS[0],self.bz_step.LateralFraction_LIMITS[1])  # LateralFraction:0.0
        YawRate = np.clip(YawRate, self.bz_step.YawRate_LIMITS[0],self.bz_step.YawRate_LIMITS[1])                                  # YawRate: 0.0
        ClearanceHeight = np.clip(ClearanceHeight,self.bz_step.ClearanceHeight_LIMITS[0],self.bz_step.ClearanceHeight_LIMITS[1])   # ClearanceHeight: 0.05
        PenetrationDepth = np.clip(PenetrationDepth,self.bz_step.PenetrationDepth_LIMITS[0],self.bz_step.PenetrationDepth_LIMITS[1]) # PenetrationDepth: 0.003
        contacts = copy.deepcopy(self.obs[1]["FootContactSensor"])

        if self.timesteps > 0:
            if self._gym_env.is_infer == False:
                if self._gym_env.dis_joint == 1 or self._gym_env.dis_joint == 2:
                    T_bf = self.bzg.GenerateTrajectoryX_FR(StepLength,LateralFraction,YawRate,StepVelocity,self.T_b0,ClearanceHeight,PenetrationDepth,contacts)
                else:
                    T_bf = self.bzg.GenerateTrajectoryX(StepLength,LateralFraction,YawRate,StepVelocity,self.T_b0,ClearanceHeight,PenetrationDepth,contacts)
            elif self._gym_env.is_infer == True:
                T_bf = self.bzg.GenerateTrajectoryX(StepLength, LateralFraction, YawRate, StepVelocity, self.T_b0,ClearanceHeight, PenetrationDepth, contacts)
        else:
            T_bf = self.T_b0

        leg_id = 0
        action_ref = np.zeros(12)
        for key in T_bf:
            leg_pos = T_bf[key]
            index, angle = self._gym_env.robot.ComputeMotorAnglesFromFootLocalPosition(leg_id,leg_pos)
            action_ref[index] = np.asarray(angle)
            leg_id += 1

        new_action = action_ref + action
        self.obs,reward,done,info = self._gym_env.step(new_action)

        info['ref_action'] = action_ref
        info['ref_leg'] = T_bf
        info['real_action'] = new_action

        '''Don't return the action_ref'''
        if self.add_bezier == False:
            return self.obs[0], reward, done, info
        else:
            obs = np.concatenate([self.obs[0], action_ref], axis=0)
            return obs, reward, done, info






