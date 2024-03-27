#!/usr/bin/env python

import numpy as np


'''
    # URDF             Lite_2            Lite_3
[-0.523,0.523]    [-0.4363,0.4363]   [-0.4887,0.4887]

[-2.67,0.314]     [-3.3336,0.2269]   [-3.4907,0.3491]

[0.524,2.792]     [0.6632,2.7576]    [0.4363,2.7576]

'''

class LegIK():
    def __init__(self,
                 legtype="RIGHT",
                 shoulder_length=0.102,
                 elbow_length=0.2,
                 wrist_length=0.21,
                 hip_lim=[-0.523,0.523],    # hip rotationange
                 shoulder_lim=[-2.67,0.314], # shoulder rotation range
                 leg_lim=[0.524,2.792]):      # leg rotation range
        self.legtype=legtype
        self.shoulder_length = shoulder_length
        self.elbow_length = elbow_length
        self.wrist_length = wrist_length
        self.hip_lim = hip_lim
        self.shoulder_lim = shoulder_lim
        self.leg_lim = leg_lim

    def get_domain(self,x,y,z):
        '''
        Calculate the leg's Domain and caps it in case of breach
        :param x: hip-to-foot distances in each dimension
        :param y:
        :param z:
        :return: Leg Domain D
        '''
        D = (y**2 + (-z)**2 - self.shoulder_length**2 + (-x)**2 - self.elbow_length**2 - self.wrist_length**2) / (2 * self.wrist_length * self.elbow_length) # e.q.(17)
        if D > 1 or D < -1:
            D = np.clip(D,-1.0,1.0)
            return D
        else:
            return D

    def solve(self,xyz_coord):
        '''
        Generic Leg Inverse Kinematics Solver
        :param xyz_coord: hip-to-foot distances in each dimensions
        :return: Joint angles required for desired position
        '''
        x = xyz_coord[0]
        y = xyz_coord[1]
        z = xyz_coord[2]
        D = self.get_domain(x,y,z)
        if self.legtype == 'RIGHT':
            return self.RightIK(x,y,z,D)
        else:
            return self.LeftIK(x,y,z,D)

    def RightIK(self,x,y,z,D):
        '''
        Right Leg Inverse Kinematics Solver
        :param x: hip-to-foot distances in each dimensions
        :param y:
        :param z:
        :param D: leg domain
        :return: Joint Angles required for desired position
        '''
        ### wrist angle, e.q.(17) ###
        wrist_angle = np.arctan2(-np.sqrt(1 - D**2),D)
        sqrt_component = y**2 + (-z)**2 - self.shoulder_length**2
        if sqrt_component < 0.0:
            sqrt_component = 0.0

        ### shoulder angle, e.q.(15) ###
        shoulder_angle = -np.arctan2(z,y) - np.arctan2(np.sqrt(sqrt_component),-self.shoulder_length)

        ### elbow angle, e.q.(16) ###
        elbow_angle = np.arctan2(-x, np.sqrt(sqrt_component)) - np.arctan2(
            self.wrist_length * np.sin(wrist_angle),
            self.elbow_length + self.wrist_length * np.cos(wrist_angle))
        joint_angles = np.array([-shoulder_angle,-elbow_angle,-wrist_angle])
        return joint_angles

    def LeftIK(self,x,y,z,D):
        '''
        Left Leg Inverse Kinematics Solver
        :param x: hip-to-feet distances in each dimension
        :param y:
        :param z:
        :param D: leg domain
        :return: Joint Angles required for desired position
        '''
        ### wrist angle, e.q.(17) ###
        wrist_angle = np.arctan2(-np.sqrt(1 - D**2),D)
        sqrt_component = y**2 + (-z)**2 - self.shoulder_length**2
        if sqrt_component < 0.0:
            sqrt_component = 0.0

        ### shoulder angle, e.q.(15), different from RightIK ###
        shoulder_angle = -np.arctan2(z, y) - np.arctan2(np.sqrt(sqrt_component), self.shoulder_length)

        ### elbow angle, e.q.(16) ###
        elbow_angle = np.arctan2(-x, np.sqrt(sqrt_component)) - np.arctan2(
            self.wrist_length * np.sin(wrist_angle),
            self.elbow_length + self.wrist_length * np.cos(wrist_angle))
        joint_angles = np.array([-shoulder_angle,-elbow_angle,-wrist_angle])
        return joint_angles
