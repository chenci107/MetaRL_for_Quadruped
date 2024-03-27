import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import math
from JYLite_env_meta.TG_and_IK.Kinematics.JYLiteKinematics import JYLiteModel

class JYLite(gym.Env):
    def __init__(self,render=True):
        self.render = render
        # self.path = os.getcwd() + '/pybullet_data/assets/urdf/spot.urdf'
        self.path = os.getcwd() + '/Lite3_urdf/urdf/Lite3.urdf'
        # self.plane_path = os.getcwd() + '/pybullet_data/plane100.urdf'
        self.action_lowest = np.array([-0.523, -2.67, 0.524] * 4)
        self.action_highest = np.array([0.523, 0.314, 2.792] * 4)
        self.action_dim = 12
        self.action_space = spaces.Box(low=self.action_lowest, high=self.action_highest, dtype=np.float32)
        self.observation_dim = 36
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self.observation_dim], dtype=np.float32)
        self._physical_client_id = p.connect(p.GUI if self.render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        '''field_1'''
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._physical_client_id)
        '''load the robot'''
        base_position = [0, 0, 0.26]
        self.robot = p.loadURDF(self.path, physicsClientId=self._physical_client_id, basePosition=base_position)
        '''
        FL_HipX:1, FL_HipY:2, FL_Knee:3,
        FR_HipX:4, FR_HipY:5, FR_Knee:6,
        HL_HipX:7, HL_HipY:8, HL_Knee:9,
        HR_HipX:10,HR_HipY:11,HR_Knee:12
        '''
        self.available_joints_indexes = [i for i in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, i)[2] != p.JOINT_FIXED]
        print('The available joints are:',self.available_joints_indexes)

        self.init_joint_pos = np.array([0.0, -0.6283, 1.3614] * 4)
        # self.init_joint_pos = np.array([0.0, 0.0, 0.0] * 4)
        for j, pos in zip(self.available_joints_indexes, self.init_joint_pos):
            p.resetJointState(self.robot, j, targetValue=pos, targetVelocity=0.0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        '''hyperparameters'''
        self.step_num = 0
        '''----------'''
        self._time_step = 0.01
        self.bodyIK = JYLiteModel()


    def _apply_action(self, action):
        assert isinstance(action, list) or isinstance(action, np.ndarray)
        if not hasattr(self, 'robot'):
            assert Exception("robot hasn't been loaded in!")
        # action_1 = np.array(action) - self.init_joint_pos
        # action_2 = np.minimum(action_1,self.action_lowest)
        # self.action = np.maximum(action_2,self.action_highest)
        self.action = action
        for n, j in enumerate(self.available_joints_indexes):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=self.action[n])

    def wholeIK_test(self,orn=None,pos=None):
        T_bf0 = self.bodyIK.WorldToFoot
        joint_angles = self.bodyIK.IK(orn=orn,pos=pos,T_bf=T_bf0)
        return joint_angles

    def _get_observation(self):
        joint_pos = []
        joint_vel = []
        for i in self.available_joints_indexes:
            lower_limit = p.getJointInfo(self.robot, i)[8]
            upper_limit = p.getJointInfo(self.robot, i)[9]
            pos, vel, _, _ = p.getJointState(self.robot, i)
            pos_mid = 0.5 * (lower_limit + upper_limit)
            pos_res = 2 * (pos - pos_mid) / (upper_limit - lower_limit)
            vel_res = 0.1 * vel
            joint_pos.append(pos_res)  # 12
            joint_vel.append(vel_res)  # 12
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot)
        pos_x, pos_y, pos_z = basePos  # 3
        ori_r, ori_p, ori_y = p.getEulerFromQuaternion(baseOri)  # 3
        (vx, vy, vz), (wr, wp, wy) = p.getBaseVelocity(self.robot)  # 6
        state = np.array([pos_x, pos_y, pos_z, ori_r, ori_p, ori_y, vx, vy, vz, wr, wp, wy])
        state = np.append(state, joint_pos)
        state = np.append(state, joint_vel)
        return state

    def reset(self):
        noise = np.random.uniform(low=-0.01, high=0.01, size=(12,))
        for j, pos in zip(self.available_joints_indexes, noise):
            p.setJointMotorControl2(self.robot, j, controlMode=p.POSITION_CONTROL, targetPosition=pos,
                                    targetvelocity=0, positionGain=0.1, velocity=0.1, force=0)
        state = self._get_observation()
        return state

    def step(self, action):
        basePos_before, _ = p.getBasePositionAndOrientation(self.robot)
        xposbefore, _, _ = basePos_before

        self.step_num += 1
        self._apply_action(action)
        p.stepSimulation()

        basePos_after, _ = p.getBasePositionAndOrientation(self.robot)
        xposafter, _, zposafter = basePos_after
        '''get observation'''
        state = self._get_observation()
        '''calculate reward'''
        forward_reward = xposafter - xposbefore
        reward = forward_reward / 0.05
        done = False
        info = {}

        time.sleep(1 / 120)
        return state, reward, done, info

    def disconnect(self):
        p.disconnect()

    def get_z_pos(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        base_x, base_y, base_z = pos
        return base_z


if __name__ == '__main__':
    env = JYLite()
    while True:
        # orn = np.array([0, 0, 0]) * math.pi / 180
        # pos = np.mat([0, 0, 0]).T
        #
        # joints = env.wholeIK_test(orn=orn, pos=pos).flatten()
        # print('---------------------')
        # print('The joints are:',joints)
        #
        # ob, reward, done, info = env.step(joints)
        # z = env.get_z_pos()
        # print('The z position is:',z)

        joints = np.array([0.0,-0.9,1.8] * 4)
        obs,reward,done,info = env.step(joints)
        z = env.get_z_pos()
        print('The z position is:',z)




