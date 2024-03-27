import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time


MOTOR_NAMES = [
    "FL_HipX", "FL_HipY", "FL_Knee",
    "FR_HipX", "FR_HipY", "FR_Knee",
    "HL_HipX", "HL_HipX", "HL_HipX",
    "HR_HipX", "HR_HipX", "HR_HipX",]


MOTOR_LIMITS_BY_NAME = {}
for name in MOTOR_NAMES:
    if "HipX" in name:                 # URDF             Lite_2            Lite_3
        MOTOR_LIMITS_BY_NAME[name] = [-0.523,0.523]   # [-0.4363,0.4363]   [-0.4887,0.4887]
    elif "HipY" in name:
        MOTOR_LIMITS_BY_NAME[name] = [-2.67,0.314]    # [-3.3336,0.2269]   [-3.4907,0.3491]
    elif "Knee" in name:
        MOTOR_LIMITS_BY_NAME[name] = [0.524,2.792]    # [0.6632,2.7576]    [0.4363,2.7576]


class JYLite(gym.Env):
    def __init__(self,render=True):
        self.render = render
        # self.path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/Lite3_urdf/urdf/Lite3+foot.urdf' # For training
        self.path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/TG_and_IK/Lite3_urdf/urdf/Lite3+foot.urdf'  # For training
        print('-------The urdf path is:',self.path)
        # self.path = os.getcwd() + '/Lite3_urdf/urdf/Lite3+foot.urdf'                                    # For check the environment
        '''Define the state space and action space'''
        self.action_lowest = np.array([-0.523,-2.67,0.524] * 4)
        self.action_highest = np.array([0.523, 0.314, 2.792] * 4)
        self.action_dim = 12
        self.action_space = spaces.Box(low=self.action_lowest,high=self.action_highest,dtype=np.float32)
        self.observation_dim = 40
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self.observation_dim], dtype=np.float32)
        self._physical_client_id = p.connect(p.GUI if self.render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        '''load the field'''
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._physical_client_id)
        '''load the robot'''
        self.init_position = [0, 0, 0.33]
        self.init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.robot = p.loadURDF(self.path, physicsClientId=self._physical_client_id, basePosition=self.init_position,baseOrientation=self.init_orientation)
        '''set the default joint angles'''
        self.available_joints_indexes = [i for i in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, i)[2] != p.JOINT_FIXED]
        '''
        FL_HipX:1, FL_HipY:2, FL_Knee:3,
        FR_HipX:4, FR_HipY:5, FR_Knee:6,
        HL_HipX:7, HL_HipY:8, HL_Knee:9,
        HR_HipX:10,HR_HipY:11,HR_Knee:12
        
        '''
        self.init_joint_pos = np.array([0.0, -0.6283, 1.3614] * 4)
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
        # self.get_link_info()

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

    def get_link_info(self):
        self.link_indexes = [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        for link_index in self.link_indexes:
            info_tuple = p.getDynamicsInfo(self.robot,link_index)
            print(info_tuple)
            print('-----------')
        info_tuple = p.getDynamicsInfo(1,-1)
        print('The plane info is:',info_tuple)
        print('--------------')
        info_tuple = p.getBodyInfo(1)
        print('The body info is:',info_tuple)



    # def get_observation(self):
    #     joint_pos = []
    #     joint_vel = []
    #     for i in self.available_joints_indexes:
    #         lower_limit = p.getJointInfo(self.robot,i)[8]
    #         upper_limit = p.getJointInfo(self.robot,i)[9]
    #         pos,vel,_,_ = p.getJointState(self.robot,i)
    #         pos_mid = 0.5 * (lower_limit + upper_limit)
    #         pos_res = 2 * (pos - pos_mid) / (upper_limit - lower_limit)
    #         vel_res = 0.1 * vel
    #         joint_pos.append(pos_res)  # 12
    #         joint_vel.append(vel_res)  # 12
    #     basePos, baseOri = p.getBasePositionAndOrientation(self.robot)
    #     pos_x,pos_y, pos_z = basePos   # 3
    #     ori_r, ori_p, ori_y = p.getEulerFromQuaternion(baseOri) # 3
    #     (vx,vy,vz),(wr,wp,wy) = p.getBaseVelocity(self.robot)   # 6
    #     state = np.array([pos_x,pos_y,pos_z,ori_r,ori_p,ori_y,vx,vy,vz,wr,wp,wy])
    #     state = np.append(state,joint_pos)
    #     state = np.append(state,joint_vel)
    #     return state

    def GetFootContacts(self):
        contacts = []
        self.foot_link_ids = [4,8,12,16]
        for idx in self.foot_link_ids:
            contact = bool(p.getContactPoints(bodyA=0,
                                         bodyB=self.robot,
                                         linkIndexA=-1,
                                         linkIndexB=idx))
            # contact = p.getContactPoints(bodyA=0,
            #                              bodyB=self.robot,
            #                              linkIndexA=-1,
            #                              linkIndexB=idx)
            # print('The contact are:',contact)
            # print('-------------------------')
            contacts.append(contact)
        return contacts


    def GetFootContacts_v2(self):
        all_contacts = p.getContactPoints(bodyA=self.robot)
        contacts = [False,False,False,False]
        for contact in all_contacts:
            if contact[2] == self.robot:
                continue
            try:
                toe_link_index = self.foot_link_ids.index(contact[3])
                contacts[toe_link_index] = True
            except ValueError:
                continue
        return contacts







    def get_observation(self):
        joint_pos = []
        joint_vel = []
        for i in self.available_joints_indexes:
            pos,vel,_,_ = p.getJointState(self.robot,i)
            joint_pos.append(pos)
            joint_vel.append(vel)
        (vx,vy,vz),(wr,wp,wy) = p.getBaseVelocity(self.robot)
        (pos_x,pos_y,pos_z),baseOri = p.getBasePositionAndOrientation(self.robot)
        ori_r, ori_p, ori_y = p.getEulerFromQuaternion(baseOri)
        contacts = self.GetFootContacts()
        state = np.array([pos_x,pos_y,pos_z,vx,vy,vz,ori_r,ori_p,ori_y,wr,wp,wy])  # 12
        state = np.append(state,joint_pos)  # 12+12
        state = np.append(state,joint_vel)  # 12+12+12
        state = np.append(state,contacts)   # 12+12+12+4 = 40
        # state = np.array(state).astype(float)

        return state


    def reset(self):
        p.resetSimulation(physicsClientId=self._physical_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._physical_client_id)
        self.robot = p.loadURDF(self.path, physicsClientId=self._physical_client_id, basePosition=self.init_position,baseOrientation=self.init_orientation)
        p.resetBasePositionAndOrientation(self.robot, self.init_position, self.init_orientation)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        self.init_joint_pos = np.array([0.0, -0.6283, 1.3614] * 4)
        for j, pos in zip(self.available_joints_indexes, self.init_joint_pos):
            p.resetJointState(self.robot, j, targetValue=pos, targetVelocity=0.0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self._step_counter = 0
        state = self.get_observation()
        return state

    def disconnect(self):
        p.disconnect()

    def get_pos(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        base_x, base_y, base_z = pos
        return base_x,base_y,base_z

    def get_orientation(self):
        _,baseOri = p.getBasePositionAndOrientation(self.robot)
        ori_r, ori_p, ori_y = p.getEulerFromQuaternion(baseOri)
        return ori_r, ori_p, ori_y



    def step(self,action):
        basePos_before, _ = p.getBasePositionAndOrientation(self.robot)
        xposbefore, _, _ = basePos_before
        self._step_counter += 1

        # self.ApplyAction(action)
        self.apply_action(action)
        p.stepSimulation()

        basePos_after, _ = p.getBasePositionAndOrientation(self.robot)
        xposafter, _, _ = basePos_after
        state = self.get_observation()

        forward_reward = xposafter - xposbefore
        reward = forward_reward / 0.05
        info = {}

        if self._step_counter >= 1000:
            done = True
        else:
            done = False

        # time.sleep(1 / 120)
        return state, reward, done, info

    def _BuildMotorIdList(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

    def _BuildJointNameToIdDict(self):
        num_joints = p.getNumJoints(self.robot)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot,i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

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

    def apply_action(self,action):
        assert isinstance(action,list) or isinstance(action,np.ndarray)
        if not hasattr(self,'robot'):
            assert Exception("robot hasn't been loaded in!")
        self.action = action
        for n, j in enumerate(self.available_joints_indexes):
            p.setJointMotorControl2(self.robot,j,p.POSITION_CONTROL,targetPosition=self.action[n])
        p.stepSimulation()





if __name__ == '__main__':
    env = JYLite()
    t = 0
    ep_rewards = 0
    for t in range(1000):
        random = np.random.standard_normal(12) * 0.5
        action = np.array([0.0, -0.6283, 1.3614] * 4) + random   # 0.3236
        # action = np.array([0.0, -0.8427, 1.6334] * 4)  # 0.3236
        ob, reward, done, info = env.step(action)
        # z_pos = env.get_z_pos()
        # print('z_pos is:',z_pos)
        ep_rewards += reward
        time.sleep(1/5)
        contacts_1 = env.GetFootContacts()
        print('The contacts_1 are:',contacts_1)
        contacts_2 = env.GetFootContacts_v2()
        print('The contacts_2 are:', contacts_2)
        print('-----------------------')


