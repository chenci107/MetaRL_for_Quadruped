import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from Kinematics.LegKinematics_JYLite import LegIK

MOTOR_NAMES = [
    "FL_HipX", "FL_HipY", "FL_Knee",
    "FR_HipX", "FR_HipY", "FR_Knee",
    "HL_HipX", "HL_HipX", "HL_HipX",
    "HR_HipX", "HR_HipX", "HR_HipX",]


MOTOR_LIMITS_BY_NAME = {}
for name in MOTOR_NAMES:
    if "HipX" in name:                 # URDF             Lite_2            Lite_3
        MOTOR_LIMITS_BY_NAME[name] = [-0.523,0.523]   # [-0.4363,0.4363]   [-0.4887,0.4887]
    elif "HFE" in name:
        MOTOR_LIMITS_BY_NAME[name] = [-2.67,0.314]    # [-3.3336,0.2269]   [-3.4907,0.3491]
    elif "KFE" in name:
        MOTOR_LIMITS_BY_NAME[name] = [0.524,2.792]    # [0.6632,2.7576]    [0.4363,2.7576]


class JYLite(gym.Env):
    def __init__(self,render=True):
        self.render = render
        # self.path = os.getcwd() + '/Lite3_urdf/urdf/Lite3_flip.urdf'
        self.path = os.getcwd() + '/a1_urdf/urdf/a1.urdf'
        self.action_lowest = np.array([-0.523,-2.67,0.524] * 4)
        self.action_highest = np.array([0.523, 0.314, 2.792] * 4)
        self.action_dim = 12
        self.action_space = spaces.Box(low=self.action_lowest,high=self.action_highest,dtype=np.float32)
        self.observation_dim = 36
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self.observation_dim], dtype=np.float32)
        self._physical_client_id = p.connect(p.GUI if self.render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        '''field_1'''
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._physical_client_id)
        '''load the robot'''
        base_position = [0, 0, 0.23]
        self.robot = p.loadURDF(self.path, physicsClientId=self._physical_client_id, basePosition=base_position)
        self.available_joints_indexes = [i for i in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, i)[2] != p.JOINT_FIXED]
        '''
        FL_HipX:1, FL_HipY:2, FL_Knee:3,
        FR_HipX:4, FR_HipY:5, FR_Knee:6,
        HL_HipX:7, HL_HipY:8, HL_Knee:9,
        HR_HipX:10,HR_HipY:11,HR_Knee:12
        
        '''
        self.init_joint_pos = np.array([0.0, 0.9, -1.8] * 4)
        for j, pos in zip(self.available_joints_indexes, self.init_joint_pos):
            p.resetJointState(self.robot, j, targetValue=pos, targetVelocity=0.0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        '''Other parameters'''
        self._time_step = 0.01
        self._step_counter = 0
        self.num_motors = 12
        self._motor_direction = np.ones(self.num_motors)
        # self._BuildJointNameToIdDict()
        # self._BuildMotorIdList()
        # self.get_joint_info()
        self.LegIK = LegIK()

    def get_joint_info(self):
        for joint_index in self.available_joints_indexes:
            info_tuple = p.getJointInfo(self.robot,joint_index)
            print(f"关节序号：{info_tuple[0]}\n\
                    关节名称：{info_tuple[1]}\n\
                    关节类型：{info_tuple[2]}\n\
                    机器人第一个位置的变量索引：{info_tuple[3]}\n\
                    机器人第一个速度的变量索引：{info_tuple[4]}\n\
                    保留参数：{info_tuple[5]}\n\
                    关节的阻尼大小：{info_tuple[6]}\n\
                    关节的摩擦系数：{info_tuple[7]}\n\
                    slider和revolute(hinge)类型的位移最小值：{info_tuple[8]}\n\
                    slider和revolute(hinge)类型的位移最大值：{info_tuple[9]}\n\
                    关节驱动的最大值：{info_tuple[10]}\n\
                    关节的最大速度：{info_tuple[11]}\n\
                    节点名称：{info_tuple[12]}\n\
                    局部框架中的关节轴系：{info_tuple[13]}\n\
                    父节点frame的关节位置：{info_tuple[14]}\n\
                    父节点frame的关节方向：{info_tuple[15]}\n\
                    父节点的索引，若是基座返回-1：{info_tuple[16]}\n\n")



    def _apply_action(self,action):
        assert isinstance(action,list) or isinstance(action,np.ndarray)
        if not hasattr(self,'robot'):
            assert Exception("robot hasn't been loaded in!")
        self.action = action
        for n, j in enumerate(self.available_joints_indexes):
            p.setJointMotorControl2(self.robot,j,p.POSITION_CONTROL,targetPosition=self.action[n])

    def _get_observation(self):
        joint_pos = []
        joint_vel = []
        for i in self.available_joints_indexes:
            lower_limit = p.getJointInfo(self.robot,i)[8]
            upper_limit = p.getJointInfo(self.robot,i)[9]
            pos,vel,_,_ = p.getJointState(self.robot,i)
            pos_mid = 0.5 * (lower_limit + upper_limit)
            pos_res = 2 * (pos - pos_mid) / (upper_limit - lower_limit)
            vel_res = 0.1 * vel
            joint_pos.append(pos_res)  # 12
            joint_vel.append(vel_res)  # 12
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot)
        pos_x,pos_y, pos_z = basePos   # 3
        ori_r, ori_p, ori_y = p.getEulerFromQuaternion(baseOri) # 3
        (vx,vy,vz),(wr,wp,wy) = p.getBaseVelocity(self.robot)   # 6
        state = np.array([pos_x,pos_y,pos_z,ori_r,ori_p,ori_y,vx,vy,vz,wr,wp,wy])
        state = np.append(state,joint_pos)
        state = np.append(state,joint_vel)
        return state

    def reset(self):
        noise = np.random.uniform(low=-0.001,high=0.001,size=(12,))
        for j, pos in zip(self.available_joints_indexes,noise):
            p.setJointMotorControl2(self.robot,j,controlMode=p.POSITION_CONTROL)
        state = self._get_observation()
        return state

    def disconnect(self):
        p.disconnect()

    def step(self,action):
        basePos_before, _ = p.getBasePositionAndOrientation(self.robot)
        xposbefore, _, _ = basePos_before
        self._step_counter += 1

        # self.ApplyAction(action)
        self._apply_action(action)
        p.stepSimulation()

        basePos_after, _ = p.getBasePositionAndOrientation(self.robot)
        xposafter, _, _ = basePos_after
        state = self._get_observation()

        forward_reward = xposafter - xposbefore
        reward = forward_reward / 0.05
        done = False
        info = {}

        time.sleep(1 / 120)
        return state, reward, done, info

    def _BuildMotorIdList(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

    def _BuildJointNameToIdDict(self):
        num_joints = p.getNumJoints(self.robot)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot,i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def ApplyAction(self,motor_commands):
        motor_commands = self.ApplyMotorLimits(motor_commands)
        motor_commands_with_direction = np.multiply(motor_commands,self._motor_direction)
        for motor_id, motor_commands_with_direction in zip(self._motor_id_list,motor_commands_with_direction):
            self._SetDesiredMotorAngleById(motor_id,motor_commands_with_direction)

    def ApplyMotorLimits(self,joint_angles):
        eps = 0.001
        for i in range(len(joint_angles)):
            LIM = MOTOR_LIMITS_BY_NAME[MOTOR_NAMES[i]]
            joint_angles[i] = np.clip(joint_angles[i], LIM[0]+eps, LIM[1]-eps)
        return joint_angles

    def _SetDesiredMotorAngleById(self,motor_id,desired_angle):
        p.setJointMotorControl2(bodyIndex=self.robot,
                                jointIndex=motor_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=desired_angle,)

    def _apply_action(self,action):
        assert isinstance(action,list) or isinstance(action,np.ndarray)
        if not hasattr(self,'robot'):
            assert Exception("robot hasn't been loaded in!")
        self.action = action
        for n, j in enumerate(self.available_joints_indexes):
            p.setJointMotorControl2(self.robot,j,p.POSITION_CONTROL,targetPosition=self.action[n])

    def LegIK_test(self,xyz_coord,foot_name=None):
        other_angles = np.array([0.0, -0.6283, 1.3614])
        if foot_name == "LF":
            self.LegIK.legtype = "LEFT"
            joint_angles = self.LegIK.solve(xyz_coord)
            all_joint_angles = np.concatenate((joint_angles, other_angles, other_angles, other_angles), axis=0)
            return all_joint_angles
        elif foot_name == "RF":
            self.LegIK.legtype = "RIGHT"
            joint_angles = self.LegIK.solve(xyz_coord)
            all_joint_angles = np.concatenate((other_angles,joint_angles,other_angles,other_angles),axis=0)
            return all_joint_angles
        elif foot_name == "LH":
            self.LegIK.legtype = "LEFT"
            joint_angles = self.LegIK.solve(xyz_coord)
            all_joint_angles = np.concatenate((other_angles,other_angles,joint_angles,other_angles),axis=0)
            return all_joint_angles
        elif foot_name == "RH":
            self.LegIK.legtype = "RIGHT"
            joint_angles = self.LegIK.solve(xyz_coord)
            all_joint_angles = np.concatenate((other_angles,other_angles,other_angles,joint_angles),axis=0)
            return all_joint_angles
        else:
            raise ValueError ("Please verify the foot_name")


def foot_position_in_hip_frame_to_joint_angle(foot_position,l_hip_sign=1):
    l_up = 0.2
    l_low = 0.21
    l_hip = 0.102 * l_hip_sign
    x,y,z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) / (2 * l_low * l_up))
    l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.asarray([theta_ab,theta_hip,theta_knee])

def leg_test_2(joint_angles,leg_idx):
    other_angles = np.array([0.0, -0.9, 1.8])
    if leg_idx == 0:
        all_joint_angles = np.concatenate((joint_angles,other_angles,other_angles,other_angles),axis=0)
        return all_joint_angles
    elif leg_idx == 1:
        all_joint_angles = np.concatenate((other_angles,joint_angles,other_angles,other_angles),axis=0)
        return all_joint_angles
    elif leg_idx == 2:
        all_joint_angles = np.concatenate((other_angles,other_angles,joint_angles,other_angles),axis=0)
        return all_joint_angles
    elif leg_idx == 3:
        all_joint_angles = np.concatenate((other_angles,other_angles,other_angles,joint_angles),axis=0)
        return all_joint_angles
    else:
        raise ValueError("Please verify the foot_name")

def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.21
    l_hip = 0.102 * l_hip_sign
    leg_distance = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])



if __name__ == '__main__':
    env = JYLite()
    while True:
        '''------------------------- TEST IK -------------------------------------------'''
        '''Mod 1: The original method'''
        # xyz_coord = np.array([0.1,0.102,-0.323])
        # # xyz_coord = np.array([0.0, 0.102, -0.323])
        # joints = env.LegIK_test(xyz_coord, foot_name="LF")
        # # print('The joints are:', joints)
        # ob, reward, done, info = env.step(joints)
        '''Method 2: The method copied from A1'''
        # foot_position_FR = np.array([0.009731, -0.100814, -0.229485])
        # joints_FR = foot_position_in_hip_frame_to_joint_angle(foot_position_FR, l_hip_sign=-1)
        #
        # foot_position_FL = np.array([0.009731,0.103186,-0.229485])
        # joints_FL = foot_position_in_hip_frame_to_joint_angle(foot_position_FL,l_hip_sign=1)
        #
        # foot_position_HR = np.array([0.015731, -0.090814, -0.229485])
        # joints_HR = foot_position_in_hip_frame_to_joint_angle(foot_position_HR, l_hip_sign=-1)
        #
        # foot_position_HL = np.array([0.015731, 0.090186, -0.229485])
        # joints_HL = foot_position_in_hip_frame_to_joint_angle(foot_position_HL,l_hip_sign=1)
        #
        # all_joints = np.concatenate((joints_FR,joints_FL,joints_HR,joints_HL),axis=0)
        #
        # ob,reward,done,info = env.step(all_joints)
        '''---------------------------- TEST INVERSE IK --------------------------------'''
        ### FR and HR
        joint_angles = np.array([0.0,-0.9,1.8])
        xyz = foot_position_in_hip_frame(joint_angles,l_hip_sign=-1)
        print('The xyz position is:',xyz)
        ### FL and HL
        joint_angles = np.array([0.0, -0.9, 1.8])
        xyz = foot_position_in_hip_frame(joint_angles, l_hip_sign=1)
        print('The xyz position is:', xyz)




