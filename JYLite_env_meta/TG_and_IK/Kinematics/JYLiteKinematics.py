#!/usr/bin/env python

import numpy as np
from JYLite_env_meta.TG_and_IK.Kinematics.LegKinematics_JYLite import LegIK
from JYLite_env_meta.TG_and_IK.Kinematics.LieAlgebra import RpToTrans,TransToRp,TransInv,RPY

from collections import OrderedDict

class JYLiteModel:
    def __init__(self,
                 shoulder_length=0.102,
                 elbow_length=0.2,
                 wrist_length=0.21,
                 hip_x=0.349,
                 hip_y=0.124,
                 foot_x=0.349,
                 foot_y=0.328,
                 height = 0.26,
                 com_offset = 0,  # need to modify
                 shoulder_lim = [-0.523,0.523],
                 elbow_lim = [-2.67,0.314],
                 wrist_lim = [0.524,2.792]):

       """Spot Micro Kinematics"""
       # COM offset in x direction
       self.com_offset = com_offset

       # Leg Parameters
       self.shoulder_length = shoulder_length
       self.elbow_length = elbow_length
       self.wrist_length = wrist_length

       # Leg Vector desired_positions

       # Distance Between Hips
       # Length
       self.hip_x = hip_x
       # Width
       self.hip_y = hip_y

       # Distance Between Feet
       # Length
       self.foot_x = foot_x
       # Width
       self.foot_y = foot_y

       # Body Height
       self.height = height

       # Joint Parameters
       self.shoulder_lim = shoulder_lim
       self.elbow_lim = elbow_lim
       self.wrist_lim = wrist_lim

       # Directionary to store Leg IK Solvers
       self.Legs = OrderedDict()
       self.Legs["FL"] = LegIK("LEFT",self.shoulder_length,self.elbow_length,self.wrist_length, self.shoulder_lim,self.elbow_lim,self.wrist_lim)
       self.Legs["FR"] = LegIK("RIGHT",self.shoulder_length,self.elbow_length,self.wrist_length, self.shoulder_lim,self.elbow_lim,self.wrist_lim)
       self.Legs["BL"] = LegIK("LEFT",self.shoulder_length,self.elbow_length,self.wrist_length, self.shoulder_lim,self.elbow_lim,self.wrist_lim)
       self.Legs["BR"] = LegIK("RIGHT",self.shoulder_length,self.elbow_length,self.wrist_length,self.shoulder_lim,self.elbow_lim,self.wrist_lim)

       # Directionary to store Hip and Foot Transforms

       # Transfrom of Hip relative to world frame with body centroid also in world frame
       Rwb = np.eye(3)
       self.WorldToHip = OrderedDict()

       self.ph_FL = np.array([self.hip_x / 2.0, self.hip_y / 2.0, 0])
       self.WorldToHip["FL"] = RpToTrans(Rwb,self.ph_FL)

       self.ph_FR = np.array([self.hip_x / 2.0, -self.hip_y / 2.0, 0])
       self.WorldToHip["FR"] = RpToTrans(Rwb, self.ph_FR)

       self.ph_BL = np.array([-self.hip_x / 2.0, self.hip_y / 2.0, 0])
       self.WorldToHip["BL"] = RpToTrans(Rwb, self.ph_BL)

       self.ph_BR = np.array([-self.hip_x / 2.0, -self.hip_y / 2.0, 0])
       self.WorldToHip["BR"] = RpToTrans(Rwb, self.ph_BR)


       # Transform of Foot relative to world frame with body centroid also in world frame
       self.WorldToFoot = OrderedDict()
       self.pf_FL = np.array([self.foot_x / 2.0, self.foot_y / 2.0, -self.height])
       self.WorldToFoot["FL"] = RpToTrans(Rwb,self.pf_FL)

       self.pf_FR = np.array([self.foot_x / 2.0, -self.foot_y / 2.0, -self.height])
       self.WorldToFoot["FR"] = RpToTrans(Rwb, self.pf_FR)

       self.pf_BL = np.array([-self.foot_x / 2.0, self.foot_y / 2.0, -self.height])
       self.WorldToFoot["BL"] = RpToTrans(Rwb, self.pf_BL)

       self.pf_BR = np.array([-self.foot_x / 2.0, -self.foot_y / 2.0, -self.height])
       self.WorldToFoot["BR"] = RpToTrans(Rwb, self.pf_BR)


    def HipToFoot(self,orn,pos,T_bf):
        '''
        Converts a desired position and orientation wrt Spot's home position, with a desired body-to-foot Transform into a
        body-to-hip Transform, which is used to extract and return the Hip to Foot Vector.

        :param orn: A 3x1 np.array([]) with Spot's Roll, Pitch, Yaw angles
        :param pos: A 3x1 np.array([]) with Spot's X, Y, Z coordinates
        :param T_bf: Dictionary of desired body-to-foot Transforms.
        :return: Hip To Foot Vector for each of Spot's Legs.
        '''

        # Only get Rot component
        Rb, _ = TransToRp(RPY(orn[0], orn[1], orn[2]))
        pb = pos
        T_wb = RpToTrans(Rb,pb)  # world-to-body

        # Dictionary to store vectors
        HipToFoot_List = OrderedDict()

        for i, (key, T_wh) in enumerate(self.WorldToHip.items()):
            # ORDER: FL, FR, BL, BR

            # Extract vector component
            _, p_bf = TransToRp(T_bf[key])

            # Step 1, get T_bh for each leg
            T_bh = np.dot(TransInv(T_wb), T_wh)  # body-to-world, world-to-hip ==> body-to-hip (not fixed)

            # Step 2, get T_hf for each leg

            # VECTOR ADDITION METHOD
            _, p_bh = TransToRp(T_bh)
            p_hf0 = p_bf - p_bh                # body-to-foot - body-to-hip ==> hip-to-foot

            # TRANSFORM METHOD
            T_hf = np.dot(TransInv(T_bh),T_bf[key])  # hip-to-body, body-to-foot ==> hip-to-foot
            _, p_hf1 = TransToRp(T_hf)

            # They should yield the same result
            if p_hf1.all() != p_hf0.all():
                print("NOT EQUAL!")

            p_hf = p_hf1

            HipToFoot_List[key] = p_hf

        return HipToFoot_List

    def IK(self,orn,pos,T_bf):
        '''
        Uses HipToFoot() to convert a desired position and orientation wrt Spot's home position into a Hip to Foot Vector, which is fed into the LegIK solver.
        Finally, the resultant joint angles are returned from the LegIK solver for each leg.
        :param orn:  A 3x1 np.array([]) with Spot's Roll, Pitch, Yaw angles
        :param pos:  A 3x1 np.array([]) with Spot's X, Y, Z coordinates
        :param T_bf: Dictionary of desired body-to-foot Transforms.
        :return: Joint angles for each of Spot's joints.
        '''

        # modify x by com offset
        pos[0] += self.com_offset

        # 4 legs, 3 joints per leg
        joint_angles = np.zeros((4,3))

        # Steps 1 and 2 of pipeline here
        HipToFoot = self.HipToFoot(orn,pos,T_bf)

        # Step 3, compute joint angles from T_hf for each leg
        for i, (key,p_hf) in enumerate (HipToFoot.items()):
            joint_angles[i,:] = self.Legs[key].solve(p_hf)

        return joint_angles
