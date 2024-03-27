import collections
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
import copy
import os

from JYLite_env_meta.robots import robot_config
from JYLite_env_meta.envs.sensors import sensor
from JYLite_env_meta.envs.sensors import space_utils
from JYLite_env_meta.envs.utilities.env_utils import flatten_observations

_NUM_SIMULATION_ITERATION_STEPS = 300

PNG_PATH = os.path.abspath(os.path.join(os.getcwd())) + "/JYLite_env_meta/sources/gap_flatten_v5.png"
TEXTURE_PATH = os.path.abspath(os.path.join(os.getcwd())) + "/JYLite_env_meta/sources/wm_height_out.png"

class LocomotionGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }
    def __init__(self,
                 gym_config,
                 param,
                 robot_class=None,
                 env_sensors=None,
                 robot_sensors=None,
                 task=None,
                 random=False,
                 env_randomizers=None):
        self.seed()
        self._gym_config = gym_config
        self._robot_class = robot_class
        self._robot_sensors = robot_sensors

        self.param = param
        self.random = random

        self._sensors = env_sensors if env_sensors is not None else list()
        if self._robot_class is None:
            raise ValueError('robot_class cannot be None.')

        self._world_dict = {}
        self._task = task
        self._env_randomizers = env_randomizers if env_randomizers else []

        if isinstance(self._task,sensor.Sensor):
            self._sensors.append(self._task)

        self._num_action_repeat = gym_config.simulation_parameters.num_action_repeat
        self._on_rack = gym_config.simulation_parameters.robot_on_rack
        if self._num_action_repeat < 1:
            raise ValueError('number of action repeats should be at least 1.')
        self._sim_time_step = gym_config.simulation_parameters.sim_time_step_s
        self._env_time_step = self._num_action_repeat * self._sim_time_step
        self._env_step_counter = 0

        self._num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS / self._num_action_repeat)
        self._is_render = gym_config.simulation_parameters.enable_rendering

        self._last_frame_time = 0.0
        self._show_reference_id = -1

        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,gym_config.simulation_parameters.enable_rendering_gui)
            if hasattr(self._task, '_draw_ref_model_alpha'):
                self._show_reference_id = pybullet.addUserDebugParameter("show reference", 0, 1,self._task._draw_ref_model_alpha)
            self._delay_id = pybullet.addUserDebugParameter("delay", 0, 0.3, 0)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        if gym_config.simulation_parameters.egl_rendering:
            self._pybullet_client.loadPlugin('eglRendererPlugin')

        self._build_action_space()

        self._camera_dist = gym_config.simulation_parameters.camera_distance
        self._camera_yaw = gym_config.simulation_parameters.camera_yaw
        self._camera_pitch = gym_config.simulation_parameters.camera_pitch
        self._render_width = gym_config.simulation_parameters.render_width
        self._render_height = gym_config.simulation_parameters.render_height

        self._hard_reset = True
        self.reset()

        self._hard_reset = gym_config.simulation_parameters.enable_hard_reset

        self.observation_space = (space_utils.convert_sensors_to_gym_space_dictionary(self.all_sensors()))

    def _build_action_space(self):
        motor_mode = self._gym_config.simulation_parameters.motor_control_mode
        if motor_mode == robot_config.MotorControlMode.HYBRID:
            action_upper_bound = []
            action_lower_bound = []
            action_config = self._robot_class.ACTION_CONFIG
            for action in action_config:
                action_lower_bound.extend([6.28] * 5)
                action_lower_bound.extend([-6.28] * 5)
            self.action_space = spaces.Box(np.array(action_lower_bound),
                                           np.array(action_upper_bound),
                                           dtype=np.float32)
        elif motor_mode == robot_config.MotorControlMode.TORQUE:
            torque_limits = np.array([33.5] * len(self._robot_class.ACTION_CONFIG))
            self.action_space = spaces.Box(-torque_limits,torque_limits,dtype=np.float32)
        else:
            action_upper_bound = []
            action_lower_bound = []
            action_config = self._robot_class.ACTION_CONFIG
            for action in action_config:
                action_upper_bound.append(action.upper_bound)
                action_lower_bound.append(action.lower_bound)

            self.action_space = spaces.Box(np.array(action_lower_bound),
                                           np.array(action_upper_bound),
                                           dtype=np.float32)

    def close(self):
        if hasattr(self, '_robot') and self._robot:
            self._robot.Terminate()

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def all_sensors(self):
        """Returns all robot and environmental sensors."""
        return self._robot.GetAllSensors() + self._sensors

    def sensor_by_name(self, name):
        """Returns the sensor with the given name, or None if not exist."""
        for sensor_ in self.all_sensors():
            if sensor_.get_name() == name:
                return sensor_
        return None

    def reset(self,**kwargs):
        for sensor in self._robot_sensors:
            sensor.reset()

        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)

        Reset = False
        if "hardset" in kwargs.keys():
            if kwargs["hardset"]:
                Reset = True

        if self._hard_reset or Reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
            self._pybullet_client.setTimeStep(self._sim_time_step)
            self._pybullet_client.setGravity(0, 0, -10)

            '''=== load the plane or heightmap'''
            self._world_dict = {"ground": self._pybullet_client.loadURDF("plane_implicit.urdf")}
            self._pybullet_client.changeDynamics(self._world_dict['ground'], -1, lateralFriction=5)

            # self.terrainShape = self._pybullet_client.createCollisionShape(
            #     shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
            #     meshScale=[0.03, 0.03, 0.02],
            #     fileName=PNG_PATH)
            # textureId = self._pybullet_client.loadTexture(TEXTURE_PATH)
            # self.terrain = self._pybullet_client.createMultiBody(0, self.terrainShape)
            # self._pybullet_client.changeVisualShape(self.terrain, -1, textureUniqueId=textureId)
            # self._pybullet_client.changeDynamics(self.terrain, -1, lateralFriction=5)
            # self._pybullet_client.resetBasePositionAndOrientation(self.terrain, [6, 0, 0], self._pybullet_client.getQuaternionFromEuler(np.array([0.0,0.0,-1.57])))
            # self._pybullet_client.changeVisualShape(self.terrain, -1, rgbaColor=[1, 1, 1, 1])


            self._robot = self._robot_class(
                pybullet_client=self._pybullet_client,
                sensors=self._robot_sensors,
                on_rack=self._on_rack,
                action_repeat=self._gym_config.simulation_parameters.num_action_repeat,
                time_step=self._gym_config.simulation_parameters.sim_time_step_s,
                motor_control_mode=self._gym_config.simulation_parameters.motor_control_mode,
                reset_time=self._gym_config.simulation_parameters.reset_time,
                enable_clip_motor_commands=self._gym_config.simulation_parameters.enable_clip_motor_commands,
                enable_action_filter=self._gym_config.simulation_parameters.enable_action_filter,
                enable_action_interpolation=self._gym_config.simulation_parameters.enable_action_interpolation,
                allow_knee_contact=self._gym_config.simulation_parameters.allow_knee_contact,
                control_latency = 0.002)

        initial_motor_angles = None
        reset_duration = 0.0
        reset_visualization_camera = True

        if "yaw" in kwargs.keys():
            yaw = kwargs["yaw"]
        else:
            yaw = 0.0

        add_x = 0
        if "x_noise" in kwargs.keys() and kwargs["x_noise"]:
            add_x = np.random.uniform(-0.2,0.1)

        self._robot.Reset(reload_urdf=False,
                          default_motor_angles=initial_motor_angles,
                          reset_time=reset_duration,
                          default_pose=[0 + add_x, 0, 0.26],
                          yaw=yaw)

        '''-----The following are added by ETGRL-----'''
        footfriction = 1
        basemass = self._robot.GetBaseMassesFromURDF()[0]
        baseinertia = self._robot.GetBaseInertiasFromURDF()
        legmass = self._robot.GetLegMassesFromURDF()
        leginertia = self._robot.GetLegInertiasFromURDF()
        gravity = np.array([0, 0, -10])

        if not self.random:
            if "dynamic_param" in kwargs.keys():
                dynamic_param = kwargs["dynamic_param"]
            else:
                dynamic_param = copy.deepcopy(self.param)
            if 'control_latency' in dynamic_param.keys():
                self._robot.SetControlLatency(0.001 * dynamic_param['control_latency'])
            if 'footfriction' in dynamic_param.keys():
                footfriction = dynamic_param['footfriction']
            if 'basemass' in dynamic_param.keys():
                basemass *= dynamic_param['basemass']
            if 'baseinertia' in dynamic_param.keys():
                baseinertia_ratio = dynamic_param['baseinertia']
                baseinertia = baseinertia[0]
                baseinertia = [(baseinertia[0] * baseinertia_ratio[0], baseinertia[1] * baseinertia_ratio[1], baseinertia[2] * baseinertia_ratio[2])]
            if 'legmass' in dynamic_param.keys():
                legmass_ratio = dynamic_param['legmass']
                legmass = legmass * np.array([legmass_ratio[0], legmass_ratio[1], legmass_ratio[2]] * 4)
            if 'leginertia' in dynamic_param.keys():
                leginertia_ratio = dynamic_param['leginertia']
                leginertia_new = []
                for i in range(12):
                    leginertia_new.append((leginertia_ratio[i] * leginertia[i][0],
                                           leginertia_ratio[i] * leginertia[i][1],
                                           leginertia_ratio[i] * leginertia[i][2]))
                leginertia = copy.deepcopy(leginertia_new)
            if 'motor_kp' in dynamic_param.keys() and 'motor_kd' in dynamic_param.keys():
                motor_kp = dynamic_param['motor_kp']
                motor_kd = dynamic_param['motor_kd']
                self._robot.SetMotorGains(motor_kp, motor_kd)
            if 'gravity' in dynamic_param.keys():
                gravity = dynamic_param['gravity']
        else:
            '''footfriction'''
            footfriction = np.random.normal(loc=1,scale=0.1)
            '''basemass'''
            basemass_ratio = np.random.normal(loc=1,scale=0.05)
            basemass = basemass * basemass_ratio
            '''baseinertia'''
            baseinertia_ratio = np.random.normal(loc=1,scale=0.1,size=3)
            baseinertia = baseinertia[0]
            baseinertia = [(baseinertia[0] * baseinertia_ratio[0], baseinertia[1] * baseinertia_ratio[1], baseinertia[2] * baseinertia_ratio[2])]
            '''legmass'''
            legmass_ratio = np.random.normal(loc=1,scale=0.05,size=3)
            legmass = legmass * np.array([legmass_ratio[0], legmass_ratio[1], legmass_ratio[2]] * 4)
            '''leginertia'''
            leginertia_ratio = np.random.normal(loc=1,scale=0.1,size=12)
            leginertia_new = []
            for i in range(12):
                leginertia_new.append((leginertia_ratio[i] * leginertia[i][0],
                                       leginertia_ratio[i] * leginertia[i][1],
                                       leginertia_ratio[i] * leginertia[i][2]))
            leginertia = copy.deepcopy(leginertia_new)

        self._robot.SetFootFriction(footfriction)
        self._robot.SetBaseMasses([basemass])
        self._robot.SetBaseInertias(baseinertia)
        self._robot.SetLegMasses(legmass)
        self._robot.SetLegInertias(leginertia)
        '''----------------------------------------------------------------'''

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._env_step_counter = 0
        if reset_visualization_camera:
            self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist,
                                                             self._camera_yaw,
                                                             self._camera_pitch,
                                                             [0, 0, 0])
        self._last_action = np.zeros(self.action_space.shape)

        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)

        for s in self.all_sensors():
            s.on_reset(self)

        if self._task and hasattr(self._task,'reset'):
            self._task.reset(self)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_env(self)

        '''----- The following are added by ETGRL -----'''
        info = {}
        info["rot_quat"] = self._robot.GetBaseOrientation()
        info["rot_mat"] = self._pybullet_client.getMatrixFromQuaternion(info["rot_quat"])
        info["base"] = self._robot.GetBasePosition()
        info["footposition"] = self._robot.GetFootPositionsInBaseFrame()
        info["pose"] = self._robot.GetBaseRollPitchYaw()
        info["real_contact"] = self._robot.GetFootContacts()
        info["joint_angle"] = self._robot.GetMotorAngles()
        info["drpy"] = self._robot.GetBaseRollPitchYawRate()
        info["energy"] = self._robot.GetEnergyConsumptionPerControlStep()
        info["latency"] = self._robot.GetControlLatency()
        info["footfriction"] = footfriction
        info["basemass"] = basemass
        info["yaw_init"] = yaw
        '''---------------------------------------'''

        return self._get_observation(), info

    def GetFootPositionsInBaseFrame(self):
        pose = self._robot.GetFootPositionsInBaseFrame()
        return pose

    def step(self,action):
        self._last_base_position = self._robot.GetBasePosition()
        self._last_action = action

        '''----- The following are added by EGTRL -----'''
        if self.random:
            control_latency = np.random.uniform(0.01,0.02)
            self._robot.SetControlLatency(control_latency)
        '''--------------------------------------------'''

        if self._is_render:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._env_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self._robot.GetBasePosition()

            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

            alpha = 1.
            if self._show_reference_id > 0:
                alpha = self._pybullet_client.readUserDebugParameter(self._show_reference_id)
            ref_col = [1,1,1,alpha]
            if hasattr(self._task, '_ref_model'):
                self._pybullet_client.changeVisualShape(self._task._ref_model, -1, rgbaColor=ref_col)
                for l in range(self._pybullet_client.getNumJoints(self._task._ref_model)):
                    self._pybullet_client.changeVisualShape(self._task._ref_model, l, rgbaColor=ref_col)

            delay = self._pybullet_client.readUserDebugParameter(self._delay_id)
            if (delay > 0):
                time.sleep(delay)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)

        ### Apply Action ###
        torques = self._robot.Step(action)

        for s in self.all_sensors():
            s.on_step(self)

        if self._task and hasattr(self._task,'update'):
            self._task.update(self)


        reward = self._reward(action,torques)

        done = self._termination()
        self._env_step_counter += 1
        if done:
            self._robot.Terminate()

        '''----- The following are added by ETGRL -----'''
        info = {}
        info["rot_quat"] = self._robot.GetBaseOrientation()
        info["rot_mat"] = self._pybullet_client.getMatrixFromQuaternion(info["rot_quat"])
        info["base"] = self._robot.GetBasePosition()
        info["footposition"] = self._robot.GetFootPositionsInBaseFrame()
        info["pose"] = self._robot.GetBaseRollPitchYaw()
        info["real_contact"] = self._robot.GetFootContacts()
        info["joint_angle"] = self._robot.GetMotorAngles()
        info["drpy"] = self._robot.GetBaseRollPitchYawRate()
        info["energy"] = self._robot.GetEnergyConsumptionPerControlStep()
        info["latency"] = self._robot.GetControlLatency()
        '''-----------------------------------------------'''

        return self._get_observation(), reward, done, info

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise ValueError('Unsupported render mode:{}'.format(mode))
        base_pos = self._robot.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._camera_dist,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def get_ground(self):
        """Get simulation ground model."""
        return self._world_dict['ground']

    def set_ground(self, ground_id):
        """Set simulation ground model."""
        self._world_dict['ground'] = ground_id

    @property
    def rendering_enabled(self):
        return self._is_render

    @property
    def last_base_position(self):
        return self._last_base_position

    @property
    def world_dict(self):
        return self._world_dict.copy()

    @world_dict.setter
    def world_dict(self, new_dict):
        self._world_dict = new_dict.copy()

    def _termination(self):
        if not self._robot.is_safe:
            return True

        if self._task and hasattr(self._task, 'done'):
            return self._task.done(self)

        for s in self.all_sensors():
            s.on_terminate(self)

        return False

    def _reward(self, action, torques):
        if self._task:
            return self._task(self, action, torques)
        return 0

    def _get_observation(self):
        sensors_dict = {}
        for s in self.all_sensors():
            sensors_dict[s.get_name()] = s.get_observation()

        observations = collections.OrderedDict(sorted(list(sensors_dict.items())))

        return observations

    def set_time_step(self, num_action_repeat, sim_step=0.001):
        if num_action_repeat < 1:
            raise ValueError('number of action repeats should be at least 1.')
        self._sim_time_step = sim_step
        self._num_action_repeat = num_action_repeat
        self._env_time_step = sim_step * num_action_repeat
        self._num_bullet_solver_iterations = (_NUM_SIMULATION_ITERATION_STEPS /self._num_action_repeat)
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(np.round(self._num_bullet_solver_iterations)))
        self._pybullet_client.setTimeStep(self._sim_time_step)
        self._robot.SetTimeSteps(self._num_action_repeat, self._sim_time_step)

    def get_time_since_reset(self):
        return self._robot.GetTimeSinceReset()

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def robot(self):
        return self._robot

    @property
    def env_step_counter(self):
        return self._env_step_counter

    @property
    def hard_reset(self):
        return self._hard_reset

    @property
    def last_action(self):
        return self._last_action

    @property
    def env_time_step(self):
        return self._env_time_step

    @property
    def task(self):
        return self._task

    @property
    def robot_class(self):
        return self._robot_class

    @property
    def gym_config(self):
        return self._gym_config

    def get_observation(self):
        return flatten_observations(self._get_observation())




















