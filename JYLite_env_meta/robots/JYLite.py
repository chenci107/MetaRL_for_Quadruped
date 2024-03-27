import os
import inspect
import xmltodict
import tempfile

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import math
import re
import numpy as np
import pybullet as pyb

from JYLite_env_meta.robots import laikago_constants
from JYLite_env_meta.robots import laikago_motor
from JYLite_env_meta.robots import minitaur
from JYLite_env_meta.robots import robot_config
from JYLite_env_meta.envs import locomotion_gym_config



NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = ["FL_HipX","FL_HipY","FL_Knee",
               "FR_HipX","FR_HipY","FR_Knee",
               "HL_HipX","HL_HipY","HL_Knee",
               "HR_HipX","HR_HipY","HR_Knee",]
INIT_RACK_POSITION = [0, 0, 1]

JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array([HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
_DEFAULT_HIP_POSITIONS = (       # the hip joint position in baselink frame
    (0.1745, 0.062, 0),
    (0.1745, -0.062, 0),
    (-0.1745, 0.062, 0),
    (-0.1745, -0.062, 0)
)

COM_OFFSET = np.array([0,0,0])
HIP_OFFSETS = np.array([[0.1745, 0.062, 0], [0.1745, -0.062, 0],
                        [-0.1745, 0.062, 0],[-0.1745, -0.062, 0]]) + COM_OFFSET

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

BASE_LINK_ID = [0]
LEG_LINK_ID = [1,2,3, 5,6,7, 9,10,11, 13,14,15]
FOOT_LINK_ID = [4,8,12,16]

'''---------- The initial position of JYLite ----------'''
INIT_POSITION = [0, 0, 0.26]

'''---------- The initial angles of JYLite ----------'''
INIT_MOTOR_ANGLES = np.array([0.0, -0.9, 1.8] * NUM_LEGS)        # ([0.0, -0.9, 1.8] * 4) ==> 0.2618


'''---------- The P gain and D gain for PD controller'''
# FL
FL_ABDUCTION_P_GAIN = 60.0
FL_ABDUCTION_D_GAIN = 1.2
FL_HIP_P_GAIN = 60.0
FL_HIP_D_GAIN = 1.2
FL_KNEE_P_GAIN = 60.0
FL_KNEE_D_GAIN = 1.2
# FR
FR_ABDUCTION_P_GAIN = 60.0
FR_ABDUCTION_D_GAIN = 1.2
FR_HIP_P_GAIN = 60.0
FR_HIP_D_GAIN = 1.2
FR_KNEE_P_GAIN = 60.0
FR_KNEE_D_GAIN = 1.2
# HL
HL_ABDUCTION_P_GAIN = 100.0
HL_ABDUCTION_D_GAIN = 1.2
HL_HIP_P_GAIN = 100.0
HL_HIP_D_GAIN = 1.2
HL_KNEE_P_GAIN = 100.0
HL_KNEE_D_GAIN = 1.2
# HR
HR_ABDUCTION_P_GAIN = 100.0
HR_ABDUCTION_D_GAIN = 1.2
HR_HIP_P_GAIN = 100.0
HR_HIP_D_GAIN = 1.2
HR_KNEE_P_GAIN = 100.0
HR_KNEE_D_GAIN = 1.2


'''---------- The URDF model path -----------'''
# URDF_FILENAME = os.path.abspath(os.path.join(os.getcwd())) + "/JYLite_env_meta/TG_and_IK/Lite3_urdf/urdf/Lite3+foot_v2.urdf"
URDF_FILENAME = os.path.abspath(os.path.join(os.getcwd(),"../")) + '/JYLite_env_meta/TG_and_IK/Lite3_urdf/urdf/Lite3+foot_v2.urdf'
print('The URDF_FILENAME is:',URDF_FILENAME)


def foot_position_in_hip_frame_to_joint_angle(foot_position,l_hip_sign=1):
    '''the rotation FL_HipY/FL_upper_joint (also include FR,HL,HR) and FL_Knee/FL_lower_joint (also include FR,HL,HR) are opposite
    For A1: FR ==> -1; FL ==> 1; HR ==> -1; HL ==> 1.
    For JYLite: FL ==> 1; FR ==> -1, HL ==> 1, HR ==> -1.
    '''
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
    return np.asarray([theta_ab,-theta_hip,-theta_knee])

def foot_position_in_hip_frame(angles, l_hip_sign=1):
    '''the same as A1.py
    For A1: FR ==> -1; FL ==> 1; HR ==> -1; HL ==> 1.
    For JYLite: FL ==> 1; FR ==> -1, HL ==> 1, HR ==> -1.
    '''
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

def foot_position_in_base_frame(foot_angles):
    foot_angles = foot_angles.reshape((4,3))
    foot_positions = np.zeros((4,3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],l_hip_sign=(-1) ** i)

    return foot_positions + HIP_OFFSETS


class JYLITE(minitaur.Minitaur):
    ACTION_CONFIG = [
        locomotion_gym_config.ScalarField(name="FL_HipX",
                                          upper_bound=0.523,
                                          lower_bound=-0.523),
        locomotion_gym_config.ScalarField(name="FL_HipY",
                                          upper_bound=0.314,
                                          lower_bound=-2.67),
        locomotion_gym_config.ScalarField(name="FL_Knee",
                                          upper_bound=2.792,
                                          lower_bound=0.524),
        locomotion_gym_config.ScalarField(name="FR_HipX",
                                          upper_bound=0.523,
                                          lower_bound=-0.523),
        locomotion_gym_config.ScalarField(name="FR_HipY",
                                          upper_bound=0.314,
                                          lower_bound=-2.67),
        locomotion_gym_config.ScalarField(name="FR_Knee",
                                          upper_bound=2.792,
                                          lower_bound=0.524),
        locomotion_gym_config.ScalarField(name="HL_HipX",
                                          upper_bound=0.523,
                                          lower_bound=-0.523),
        locomotion_gym_config.ScalarField(name="HL_HipY",
                                          upper_bound=0.314,
                                          lower_bound=-2.67),
        locomotion_gym_config.ScalarField(name="HL_Knee",
                                          upper_bound=2.792,
                                          lower_bound=0.524),
        locomotion_gym_config.ScalarField(name="HR_HipX",
                                          upper_bound=0.523,
                                          lower_bound=-0.523),
        locomotion_gym_config.ScalarField(name="HR_HipY",
                                          upper_bound=0.314,
                                          lower_bound=-2.67),
        locomotion_gym_config.ScalarField(name="HR_Knee",
                                          upper_bound=2.792,
                                          lower_bound=0.524),
    ]
    def __init__(self,
                 pybullet_client,
                 urdf_filename=URDF_FILENAME,
                 enable_clip_motor_commands=False,
                 time_step=None,
                 action_repeat=None,
                 sensors=None,
                 control_latency=None,
                 on_rack=False,
                 enable_action_interpolation=True,
                 enable_action_filter=False,
                 motor_control_mode=None,
                 reset_time=1,
                 allow_knee_contact=False,):
        self._urdf_filename = urdf_filename
        self._allow_knee_contact = allow_knee_contact
        self._enable_clip_motor_commands = enable_clip_motor_commands

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

        motor_kp = [
            FL_ABDUCTION_P_GAIN, FL_HIP_P_GAIN, FL_KNEE_P_GAIN,
            FR_ABDUCTION_P_GAIN, FR_HIP_P_GAIN, FR_KNEE_P_GAIN,
            HL_ABDUCTION_P_GAIN, HL_HIP_P_GAIN, HL_KNEE_P_GAIN,
            HR_ABDUCTION_P_GAIN, HR_HIP_P_GAIN, HR_KNEE_P_GAIN]

        motor_kd = [
            FL_ABDUCTION_D_GAIN, FL_HIP_D_GAIN, FL_KNEE_D_GAIN,
            FR_ABDUCTION_D_GAIN, FR_HIP_D_GAIN, FR_KNEE_D_GAIN,
            HL_ABDUCTION_D_GAIN, HL_HIP_D_GAIN, HL_KNEE_D_GAIN,
            HR_ABDUCTION_D_GAIN, HR_HIP_D_GAIN, HR_KNEE_D_GAIN]


        super(JYLITE,self).__init__(
            pybullet_client=pybullet_client,
            time_step=time_step,
            action_repeat=action_repeat,
            num_motors=NUM_MOTORS,
            dofs_per_leg=DOFS_PER_LEG,
            motor_direction=JOINT_DIRECTIONS,
            motor_offset=JOINT_OFFSETS,
            motor_overheat_protection=False,
            motor_control_mode=motor_control_mode,
            motor_model_class=laikago_motor.LaikagoMotorModel,
            sensors=sensors,
            motor_kp=motor_kp,
            motor_kd=motor_kd,
            control_latency=control_latency,
            on_rack=on_rack,
            enable_action_interpolation=enable_action_interpolation,
            enable_action_filter=enable_action_filter,
            reset_time=reset_time)

    '''the same as the base class'''
    def _LoadRobotURDF(self):
        jylite_urdf_path = self.GetURDFFile()
        if self._self_collision_enabled:
            self.quadruped = self._pybullet_client.loadURDF(jylite_urdf_path,
                                                            self._GetDefaultInitPosition(),
                                                            self._GetDefaultInitOrientation(),
                                                            flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
        else:
            self.quadruped = self._pybullet_client.loadURDF(jylite_urdf_path,
                                                            self._GetDefaultInitPosition(),
                                                            self._GetDefaultInitOrientation())

    '''the same as the base class'''
    def _SettleDownForReset(self, default_motor_angles, reset_time):
        self.ReceiveObservation()
        if reset_time <= 0:
            return

        for _ in range(500):
            self._StepInternal(INIT_MOTOR_ANGLES,motor_control_mode=robot_config.MotorControlMode.POSITION)

        if default_motor_angles is not None:
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
                self._StepInternal(default_motor_angles,motor_control_mode=robot_config.MotorControlMode.POSITION)

    '''base class not implemented'''
    def GetHipPositionsInBaseFrame(self):
        return _DEFAULT_HIP_POSITIONS

    '''different implement from the base class'''
    def GetFootContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
        contacts = [False, False, False, False]
        for contact in all_contacts:
            if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[_LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue
        return contacts

    '''the following are add by ETGRL'''
    def GetBadFootContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
        bad_num = 0
        for contact in all_contacts:
            if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            elif contact[_LINK_A_FIELD_NUMBER] % 4 != 0:  # self.foot_link_ids = [4,8,12,16]
                bad_num += 1
        return bad_num

    '''the following are add by ETGRL'''
    def GetFootContactsForce(self,mode='simple'):
        '''
        not simple:
        [1(indicate contact), force_1, force_2, force_3]
        [0(not contact),0,0,0]
        [1(indicate contact), force_1, force_2, force_3]
        [0(not contact),0,0,0]
        simple:
        [1,0,1,0,np.linalg.norm(force_1,force_2,force_3),0,np.linalg.norm(force_1,force_2,force_3),0]
        :param mode:
        :return:
        '''
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
        contacts = np.zeros((4,4))
        for contact in all_contacts:
            if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[_LINK_A_FIELD_NUMBER])
                contacts[toe_link_index,0] = 1
                normalForce = contact[9] * np.asarray(contact[7])
                for i in range(3):
                    contacts[toe_link_index,i+1] += normalForce[i]
            except ValueError:
                continue
        simplecontact = np.zeros(8)
        if mode == 'simple':
            for m in range(4):
                simplecontact[m] = contacts[m,0]
                simplecontact[m + 4] = np.linalg.norm(contacts[m, 1:]) / 100.0
            return simplecontact
        else:
            return contacts.reshape(-1)

    '''different implement from the base class'''
    def ResetPose(self,add_constraint):
        del add_constraint
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                        jointIndex=(joint_id),
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        targetVelocity=0,
                                                        force=0)
        for name, i in zip(MOTOR_NAMES,range(len(MOTOR_NAMES))):
            if "HipX" in name:
                angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
            elif "HipY" in name:
                angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
            elif "Knee" in name:
                angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
            else:
                raise ValueError("The name %s is not recognized as a motor joint." % name)
            self._pybullet_client.resetJointState(self.quadruped,
                                                  self._joint_name_to_id[name],
                                                  angle,
                                                  targetVelocity=0)

    '''the same as the base class'''
    def GetURDFFile(self):
        return self._urdf_filename

    '''different implement from the base class'''
    def _BuildUrdfIds(self):
        '''need to be modified'''
        self._foot_link_ids = FOOT_LINK_ID
        self._chassis_link_ids = BASE_LINK_ID
        self._leg_link_ids = LEG_LINK_ID

    '''the same as the base class'''
    def _GetMotorNames(self):
        return MOTOR_NAMES

    '''different implement from the base class'''
    def _GetDefaultInitPosition(self):
        if self._on_rack:
            return INIT_RACK_POSITION
        else:
            return INIT_POSITION

    '''different implement from the base class'''
    def _GetDefaultInitOrientation(self):
        init_orientation = pyb.getQuaternionFromEuler([0., 0., 0.])
        return init_orientation

    '''the following are add by ETGRL'''
    def GetDefaultInitPosition(self):
        return self._GetDefaultInitPosition()

    '''the following are add by ETGRL'''
    def GetDefaultInitOrientation(self):
        return self._GetDefaultInitOrientation()

    '''the following are add by ETGRL'''
    def GetDefaultInitJointPose(self):
        joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
        return joint_pose

    '''the following are based on the base class'''
    def ApplyAction(self, motor_commands, motor_control_mode=None):
        if self._enable_clip_motor_commands:
            motor_commands = self._ClipMotorCommands(motor_commands)
        t = super(JYLITE,self).ApplyAction(motor_commands,motor_control_mode)
        return t


    '''the following are add by ETGRL'''
    def _ClipMotorCommands(self,motor_commands):
        max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angle = self.GetMotorAngles()
        motor_commands = np.clip(motor_commands,
                                 current_motor_angle - max_angle_change,
                                 current_motor_angle + max_angle_change)
        return motor_commands

    '''different implement from the base class'''
    @classmethod
    def GetConstants(cls):
        del cls
        return laikago_constants

    '''different implement from the base class'''
    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id, foot_local_position):
        assert len(self._foot_link_ids) == self.num_legs
        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = list(range(leg_id * motors_per_leg, leg_id * motors_per_leg + motors_per_leg))

        joint_angles = foot_position_in_hip_frame_to_joint_angle(foot_local_position - HIP_OFFSETS[leg_id], l_hip_sign=(-1) ** (leg_id))

        joint_angles = np.multiply(np.asarray(joint_angles) - np.asarray(self._motor_offset)[joint_position_idxs],self._motor_direction[joint_position_idxs])

        return joint_position_idxs,joint_angles

    '''different implement from the base class'''
    def GetFootPositionsInBaseFrame(self):
        motor_angles = self.GetMotorAngles()
        return foot_position_in_base_frame(motor_angles)

    '''not implement'''
    def ComputeJacobian(self, leg_id):
        raise NotImplementedError

    def ChangeColor(self,dis_joint):
        if dis_joint == self.FL_HipX:
            link_index = 1
        elif dis_joint == self.FL_HipY:
            link_index = 2
        elif dis_joint == self.FL_Knee:
            link_index = 3
        elif dis_joint == self.FR_HipX:
            link_index = 5
        elif dis_joint == self.FR_HipY:
            link_index = 6
        elif dis_joint == self.FR_Knee:
            link_index = 7
        elif dis_joint == self.HL_HipX:
            link_index = 9
        elif dis_joint == self.HL_HipY:
            link_index = 10
        elif dis_joint == self.HL_Knee:
            link_index = 11
        elif dis_joint == self.HR_HipX:
            link_index = 13
        elif dis_joint == self.HR_HipY:
            link_index = 14
        elif dis_joint == self.HR_Knee:
            link_index = 15
        else:
            raise NotImplementedError

        self._pybullet_client.changeVisualShape(objectUniqueId=1, linkIndex=link_index, rgbaColor=[1, 0, 0, 1])

    def RecoverColor(self):
        all_links_index = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        for index in all_links_index:
            self._pybullet_client.changeVisualShape(objectUniqueId=1, linkIndex=index, rgbaColor=[0.24, 0.24, 0.24, 1])




































