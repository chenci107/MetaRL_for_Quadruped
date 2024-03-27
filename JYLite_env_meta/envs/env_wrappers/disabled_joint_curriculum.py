from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np


NUM_ITERATIONS = 1000

class DisabledJointWrapper(object):
    def __init__(self,gym_env,dis_joint=None):
        self._gym_env = gym_env

        self.dis_joint = dis_joint
        self.epochs = 0
        self.curriculum_scale = 0
        self.is_infer = False
        self.specify_pos = np.array([0.9,-1.8])
        self.stuck_angle = 0.0

        self.FL_HipX = 0
        self.FL_HipY = 1
        self.FL_Knee = 2
        self.FR_HipX = 3
        self.FR_HipY = 4
        self.FR_Knee = 5
        self.HL_HipX = 6
        self.HL_HipY = 7
        self.HL_Knee = 8
        self.HR_HipX = 9
        self.HR_HipY = 10
        self.HR_Knee = 11

        self.HipY_and_Knee = np.array([
            self.FL_HipY,self.FL_Knee,
            self.FR_HipY,self.FR_Knee,
            self.HL_HipY,self.HL_Knee,
            self.HR_HipY,self.HR_Knee,])

    # important
    def __getattr__(self, attr):
        return getattr(self._gym_env,attr)

    def SpecifyDisabledIndex(self,index=None):
        self.dis_joint = index

    def _DisabledMotorCurriculum(self,motor_commands):
        new_joint_angles = copy.deepcopy(motor_commands)
        new_joint_angles[self.dis_joint] = self.stuck_angle
        return new_joint_angles

    def reset(self,**kwargs):
        obs, info = self._gym_env.reset(**kwargs)
        return obs, info

    def step(self,action):
        new_action = self._DisabledMotorCurriculum(motor_commands=action)
        obs,reward,done,info = self._gym_env.step(new_action)

        return obs,reward,done,info

    def reset_task(self,**kwargs):
        self._robot.RecoverColor()
        '''step 1: specify the self.dis_joint'''
        if "idx" in kwargs.keys():
            self.dis_joint = kwargs["idx"]
        else:
            self.dis_joint = np.random.choice(a=self.HipY_and_Knee, size=1, replace=True, p=None)[0]

        '''step 2: specify the self.curriculum_scale'''
        if "iterations" in kwargs.keys():
            self.epochs = kwargs["iterations"]
        else:
            self.epochs = 0

        if 0 <= self.epochs < int(NUM_ITERATIONS / 5):
            self.curriculum_scale = 0
        elif 1 * int(NUM_ITERATIONS / 5) <= self.epochs < 2 * int(NUM_ITERATIONS / 5):
            self.curriculum_scale = 1
        elif 2 * int(NUM_ITERATIONS / 5) <= self.epochs < 3 * int(NUM_ITERATIONS / 5):
            self.curriculum_scale = 2
        elif 3 * int(NUM_ITERATIONS / 5) <= self.epochs < 4 * int(NUM_ITERATIONS / 5):
            self.curriculum_scale = 3
        elif 4 * int(NUM_ITERATIONS / 5) <= self.epochs < NUM_ITERATIONS:
            self.curriculum_scale = 4
        else:
            pass

        '''step 3: specify the specify_pos for testing'''
        if "specify_pos" in kwargs.keys():
            self.specify_pos = kwargs["specify_pos"]
            self.curriculum_scale = 5
        else:
            pass

        '''step 4: specify the self.stuck angle'''
        if self.dis_joint == self.FL_HipX or self.dis_joint == self.FR_HipX or self.dis_joint == self.HL_HipX or self.dis_joint == self.HR_HipX:
            raise NotImplementedError
        elif self.dis_joint == self.FL_HipY or self.dis_joint == self.FR_HipY or self.dis_joint == self.HL_HipY or self.dis_joint == self.HR_HipY:
            if self.curriculum_scale == 0:
                self.stuck_angle = -0.9
            elif self.curriculum_scale == 1:
                self.stuck_angle = np.random.uniform(-1.0,-0.8)
            elif self.curriculum_scale == 2:
                self.stuck_angle = np.random.uniform(-1.1,-0.7)
            elif self.curriculum_scale == 3:
                self.stuck_angle = np.random.uniform(-1.2,-0.6)
            elif self.curriculum_scale == 4:
                self.stuck_angle = np.random.uniform(-1.4,-0.6)
            elif self.curriculum_scale == 5:
                self.stuck_angle = self.specify_pos[0]
            else:
                print('Please specify the curriculum scales!')
                raise NotImplementedError
        elif self.dis_joint == self.FL_Knee or self.dis_joint == self.FR_Knee or self.dis_joint == self.HL_Knee or self.dis_joint == self.HR_Knee:
            if self.curriculum_scale == 0:
                self.stuck_angle = 1.8
            elif self.curriculum_scale == 1:
                self.stuck_angle = np.random.uniform(1.7,1.9)
            elif self.curriculum_scale == 2:
                self.stuck_angle = np.random.uniform(1.7,2.0)
            elif self.curriculum_scale == 3:
                self.stuck_angle = np.random.uniform(1.7,2.2)
            elif self.curriculum_scale == 4:
                self.stuck_angle = np.random.uniform(1.7,2.4)
            elif self.curriculum_scale == 5:
                self.stuck_angle = self.specify_pos[1]
            else:
                print('Please specify the curriculum scales!')
                raise NotImplementedError
        else:
            raise NotImplementedError

        print('The self.dis_joint is: {}, self.curriculum_scale is: {}, self.stuck_angle is: {}'.format(self.dis_joint,self.curriculum_scale,self.stuck_angle))

        obs, info = self._gym_env.reset(**kwargs)
        self._robot.ChangeColor(dis_joint=self.dis_joint)

        return obs,info



