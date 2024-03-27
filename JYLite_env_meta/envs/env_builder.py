import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import numpy as np

from JYLite_env_meta.envs import locomotion_gym_env
from JYLite_env_meta.envs import locomotion_gym_config
from JYLite_env_meta.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from JYLite_env_meta.envs.env_wrappers import trajectory_generator_wrapper_env
from JYLite_env_meta.envs.env_wrappers import simple_openloop
from JYLite_env_meta.envs.env_wrappers import simple_forward_task
from JYLite_env_meta.envs.sensors import robot_sensors

from JYLite_env_meta.robots import JYLite
from JYLite_env_meta.robots import robot_config
from JYLite_env_meta.envs.env_wrappers.gait_generator_env import GaitGeneratorWrapperEnv
from JYLite_env_meta.envs.env_wrappers.gait_generator_env_init_pose import GaitGeneratorWrapperEnv_InitPose
from JYLite_env_meta.envs.env_wrappers.no_disabled_joint import NODisabledJointWrapper
from JYLite_env_meta.envs.env_wrappers.disabled_joint_curriculum import DisabledJointWrapper


def build_regular_env(robot_class,
                      motor_control_mode,
                      param,
                      sensor_mode,
                      gait=0,
                      normal=0,
                      enable_rendering=False,
                      on_rack=False,
                      filter=0,
                      action_space=0,
                      random=False,
                      wrap_trajectory_generator=True,
                      enable_disabled=True):
    '''step 1: specify the gym_config'''
    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = motor_control_mode
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 20
    sim_params.enable_action_interpolation = False
    if filter:
        sim_params.enable_action_filter = True
    else:
        sim_params.enable_action_filter = False
    sim_params.enable_clip_motor_commands = False
    sim_params.robot_on_rack = on_rack
    dt = sim_params.num_action_repeat * sim_params.sim_time_step_s

    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    '''step 2: specify the sensors'''
    sensors = []
    print(sensor_mode)
    # Speed, 3-dim, the speed of the robot at the global coordinate
    if sensor_mode["dis"]:
        sensors.append(robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True,normal=normal))
    # IMU, 6-dim, Yaw, Pitch, Roll, d_Yaw, d_Pitch, d_Roll
    if sensor_mode["imu"] == 1:
        sensors.append(robot_sensors.IMUSensor(channels=["R", "P", "Y", "dR", "dP", "dY"],normal=normal))
    elif sensor_mode["imu"] == 2:
        sensors.append(robot_sensors.IMUSensor(channels=["dR", "dP", "dY"]))
    # Motor Angle + Motor Angle ACC, 12+12=24-dims, The current motor angle + The accelerate of motor angle
    if sensor_mode["motor"] == 1:
        sensors.append(robot_sensors.MotorAngleAccSensor(num_motors=JYLite.NUM_MOTORS, normal=normal, dt=dt))
    elif sensor_mode["motor"] == 2:
        sensors.append(robot_sensors.MotorAngleSensor(num_motors=JYLite.NUM_MOTORS,))
    # Contact, 4-dims, Four bool variables indicating if the whether foot is touching the ground.
    if sensor_mode["contact"] == 1:
        sensors.append(robot_sensors.FootContactSensor())
    elif sensor_mode["contact"] == 2:
        sensors.append(robot_sensors.SimpleFootForceSensor())
    # footpose, 12-dims, Four feet position at the body coordinate
    if sensor_mode["footpose"]:
        sensors.append(robot_sensors.FootPoseSensor(normal=normal))

    '''step 3: Wrapped the simpleForwardTask'''
    task = simple_forward_task.SimpleForwardTask(param)

    '''step 4: gym_config + robot_class + sensors + task ==> env'''
    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                              param=param,
                                              robot_class=robot_class,
                                              robot_sensors=sensors,
                                              random=random,
                                              task=task,)
    '''step 5: flatten observation'''
    env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

    '''step 6: wrapped with disabledjoint'''
    if enable_disabled:
        env = DisabledJointWrapper(env,dis_joint=None)
    else:
        env = NODisabledJointWrapper(env)

    '''step 7: wrapped with LaikagoPoseOffsetGenerator'''
    add_bezier = True if ("add_bezier" in sensor_mode and sensor_mode["add_bezier"]) else False
    if gait == "straight" and (motor_control_mode == robot_config.MotorControlMode.POSITION):
        env = GaitGeneratorWrapperEnv(env,gait_mode=gait,add_bezier=add_bezier)
    elif gait == "initpose" and (motor_control_mode == robot_config.MotorControlMode.POSITION):
        env = GaitGeneratorWrapperEnv_InitPose(env)


    elif (motor_control_mode == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
        env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(gym_env=env,
                                                                             trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=0.65,action_space=action_space))

    return env



















