from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import typing

_ARRAY = typing.Iterable[float]  # pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint: disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any]  # pylint: disable=invalid-name

from JYLite_env_meta.envs.sensors import sensor

'''====================================== NOT USE ============================================'''
class MotorAngleSensor(sensor.BoxSpaceSensor):
    """A sensor that reads motor angles from the robot."""

    def __init__(self,
                 num_motors: int,
                 noisy_reading: bool = True,
                 observe_sine_cosine: bool = False,
                 lower_bound: _FLOAT_OR_ARRAY = -np.pi,
                 upper_bound: _FLOAT_OR_ARRAY = np.pi,
                 name: typing.Text = "MotorAngle",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs MotorAngleSensor.
        Args:
          num_motors: the number of motors in the robot
          noisy_reading: whether values are true observations
          observe_sine_cosine: whether to convert readings to sine/cosine values for
            continuity
          lower_bound: the lower bound of the motor angle
          upper_bound: the upper bound of the motor angle
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_motors = num_motors
        self._noisy_reading = noisy_reading
        self._observe_sine_cosine = observe_sine_cosine

        if observe_sine_cosine:
            super(MotorAngleSensor, self).__init__(
                name=name,
                shape=(self._num_motors * 2,),
                lower_bound=-np.ones(self._num_motors * 2),
                upper_bound=np.ones(self._num_motors * 2),
                dtype=dtype)
        else:
            super(MotorAngleSensor, self).__init__(
                name=name,
                shape=(self._num_motors,),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dtype=dtype)

    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_angles = self._robot.GetMotorAngles()
        else:
            motor_angles = self._robot.GetTrueMotorAngles()
        if self._observe_sine_cosine:
            return np.hstack((np.cos(motor_angles), np.sin(motor_angles)))
        else:
            return motor_angles

    def reset(self):
        pass

class MotorAngleAccSensor(sensor.BoxSpaceSensor):
    """A sensor that reads motor angles from the robot."""

    def __init__(self,
                 num_motors: int,
                 noisy_reading: bool = True,
                 observe_sine_cosine: bool = False,
                 normal: int = 0,
                 dt: float = 0.020,
                 lower_bound: _FLOAT_OR_ARRAY = -np.pi,
                 upper_bound: _FLOAT_OR_ARRAY = np.pi,
                 name: typing.Text = "MotorAngleAcc",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs MotorAngleSensor.

        Args:
          num_motors: the number of motors in the robot
          noisy_reading: whether values are true observations
          observe_sine_cosine: whether to convert readings to sine/cosine values for
            continuity
          lower_bound: the lower bound of the motor angle
          upper_bound: the upper bound of the motor angle
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_motors = num_motors
        self._noisy_reading = noisy_reading
        self._observe_sine_cosine = observe_sine_cosine
        # self.last_angle = np.zeros(self._num_motors)
        self.last_angle = np.array([0,-0.9,1.8] * 4)
        self.normal = normal
        self.first_time = True
        self._mean = np.array([0, -0.9, 1.8] * 4 + [0] * 12)
        self._std = np.array([0.1] * 12 + [1] * 12)
        self.dt = dt
        if observe_sine_cosine:
            super(MotorAngleAccSensor, self).__init__(
                name=name,
                shape=(self._num_motors * 2 * 2,),
                lower_bound=-np.ones(self._num_motors * 2),
                upper_bound=np.ones(self._num_motors * 2),
                dtype=dtype)
        else:
            super(MotorAngleAccSensor, self).__init__(
                name=name,
                shape=(self._num_motors * 2,),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dtype=dtype)

    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_angles = np.asarray(self._robot.GetMotorAngles())
        else:
            motor_angles = np.asarray(self._robot.GetTrueMotorAngles())
        '''Method 1: Motor Velocity (Original method)'''
        if self.first_time:
            motor_acc = np.zeros(self._num_motors)
            self.first_time = False
        else:
            motor_acc = (motor_angles - self.last_angle) / self.dt

        '''Method 2: Motor Velocity (From API)'''
        # motor_acc = self._robot.GetTrueMotorVelocities()

        '''Method 3: Last action'''
        motor_acc = self.last_angle

        self.last_angle = motor_angles
        if self._observe_sine_cosine:
            return np.hstack((np.cos(motor_angles), np.sin(motor_angles)))
        else:
            if self.normal:
                return (np.concatenate((motor_angles, motor_acc)) - self._mean) / self._std
            else:
                return np.concatenate((motor_angles, motor_acc))

    def reset(self):
        self.last_angle = np.zeros(self._num_motors)
        self.first_time = True

class BaseDisplacementSensor(sensor.BoxSpaceSensor):
    """A sensor that reads displacement of robot base."""

    def __init__(self,
                 lower_bound: _FLOAT_OR_ARRAY = -0.1,
                 upper_bound: _FLOAT_OR_ARRAY = 0.1,
                 convert_to_local_frame: bool = False,
                 normal: int = 0,
                 dt: float = 0.020,
                 name: typing.Text = "BaseDisplacement",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs BaseDisplacementSensor.

        Args:
          lower_bound: the lower bound of the base displacement
          upper_bound: the upper bound of the base displacement
          convert_to_local_frame: whether to project dx, dy to local frame based on
            robot's current yaw angle. (Note that it's a projection onto 2D plane,
            and the roll, pitch of the robot is not considered.)
          name: the name of the sensor
          dtype: data type of sensor value
        """

        self._channels = ["x", "y", "z"]
        self._num_channels = len(self._channels)

        super(BaseDisplacementSensor, self).__init__(
            name=name,
            shape=(self._num_channels,),
            lower_bound=np.array([lower_bound] * 3),
            upper_bound=np.array([upper_bound] * 3),
            dtype=dtype)

        datatype = [("{}_{}".format(name, channel), self._dtype) for channel in self._channels]
        self._datatype = datatype
        self._convert_to_local_frame = convert_to_local_frame
        self.dt = dt
        self._last_yaw = 0
        self._last_base_position = np.zeros(3)
        self._current_yaw = 0
        self._current_base_position = np.zeros(3)
        self._mean = np.array([0] * 3)
        self._std = np.array([0.1] * 3)
        self.normal = normal

    def get_channels(self) -> typing.Iterable[typing.Text]:
        """Returns channels (displacement in x, y, z direction)."""
        return self._channels

    def get_num_channels(self) -> int:
        """Returns number of channels."""
        return self._num_channels

    def get_observation_datatype(self) -> _DATATYPE_LIST:
        """See base class."""
        return self._datatype

    def _get_observation(self) -> _ARRAY:
        """See base class."""
        dx, dy, dz = (self._current_base_position - self._last_base_position) / self.dt
        if self._convert_to_local_frame:
            dx_local = np.cos(self._last_yaw) * dx + np.sin(self._last_yaw) * dy
            dy_local = -np.sin(self._last_yaw) * dx + np.cos(self._last_yaw) * dy
            if self.normal:
                return (np.array([dx_local, dy_local, dz]) - self._mean) / self._std
            else:
                return np.array([dx_local, dy_local, dz])
        else:
            if self.normal:
                return (np.array([dx, dy, dz]) - self._mean) / self._std
            else:
                return np.array([dx, dy, dz])

    def on_reset(self, env):
        """See base class."""
        self._current_base_position = np.array(self._robot.GetBasePosition())
        self._last_base_position = np.array(self._robot.GetBasePosition())
        self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]
        self._last_yaw = self._robot.GetBaseRollPitchYaw()[2]

    def on_step(self, env):
        """See base class."""
        self._last_base_position = self._current_base_position
        self._current_base_position = np.array(self._robot.GetBasePosition())
        self._last_yaw = self._current_yaw
        self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]

    def reset(self):
        pass

class IMUSensor(sensor.BoxSpaceSensor):
    """An IMU sensor that reads orientations and angular velocities."""

    def __init__(self,
                 channels: typing.Iterable[typing.Text] = None,
                 noisy_reading: bool = True,
                 normal: int = 0,
                 lower_bound: _FLOAT_OR_ARRAY = None,
                 upper_bound: _FLOAT_OR_ARRAY = None,
                 name: typing.Text = "IMU",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs IMUSensor.

        It generates separate IMU value channels, e.g. IMU_R, IMU_P, IMU_dR, ...

        Args:
          channels: value channels wants to subscribe. A upper letter represents
            orientation and a lower letter represents angular velocity. (e.g. ['R',
            'P', 'Y', 'dR', 'dP', 'dY'] or ['R', 'P', 'dR', 'dP'])
          noisy_reading: whether values are true observations
          lower_bound: the lower bound IMU values
            (default: [-2pi, -2pi, -2000pi, -2000pi])
          upper_bound: the lower bound IMU values
            (default: [2pi, 2pi, 2000pi, 2000pi])
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._channels = channels if channels else ["R", "P", "dR", "dP"]
        print('The channels is:',self._channels)

        self._num_channels = len(self._channels)
        self._noisy_reading = noisy_reading
        self._mean = np.array([0] * 6)
        self._std = np.array([0.1] * 3 + [0.5] * 3)
        self.normal = normal
        # Compute the default lower and upper bounds
        if lower_bound is None and upper_bound is None:
            lower_bound = []
            upper_bound = []
            for channel in self._channels:
                if channel in ["R", "P", "Y"]:
                    lower_bound.append(-2.0 * np.pi)
                    upper_bound.append(2.0 * np.pi)
                elif channel in ["Rcos", "Rsin", "Pcos", "Psin", "Ycos", "Ysin"]:
                    lower_bound.append(-1.)
                    upper_bound.append(1.)
                elif channel in ["dR", "dP", "dY"]:
                    lower_bound.append(-2000.0 * np.pi)
                    upper_bound.append(2000.0 * np.pi)

        super(IMUSensor, self).__init__(
            name=name,
            shape=(self._num_channels,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)

        # Compute the observation_datatype
        datatype = [("{}_{}".format(name, channel), self._dtype)
                    for channel in self._channels]

        self._datatype = datatype

    def get_channels(self) -> typing.Iterable[typing.Text]:
        return self._channels

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_observation_datatype(self) -> _DATATYPE_LIST:
        """Returns box-shape data type."""
        return self._datatype

    def _get_observation(self) -> _ARRAY:
        if self.first_time:
            self.first_rpy = self._robot.GetBaseRollPitchYaw()
            self.first_time = False

        if self._noisy_reading:
            rpy = self._robot.GetBaseRollPitchYaw() - self.first_rpy
            drpy = self._robot.GetBaseRollPitchYawRate()
        else:
            rpy = self._robot.GetTrueBaseRollPitchYaw() - self.first_rpy
            drpy = self._robot.GetTrueBaseRollPitchYawRate()

        assert len(rpy) >= 3, rpy
        assert len(drpy) >= 3, drpy

        observations = np.zeros(self._num_channels)
        for i, channel in enumerate(self._channels):
            if channel == "R":
                observations[i] = rpy[0]
            if channel == "Rcos":
                observations[i] = np.cos(rpy[0])
            if channel == "Rsin":
                observations[i] = np.sin(rpy[0])
            if channel == "P":
                observations[i] = rpy[1]
            if channel == "Pcos":
                observations[i] = np.cos(rpy[1])
            if channel == "Psin":
                observations[i] = np.sin(rpy[1])
            if channel == "Y":
                observations[i] = rpy[2]
            if channel == "Ycos":
                observations[i] = np.cos(rpy[2])
            if channel == "Ysin":
                observations[i] = np.sin(rpy[2])
            if channel == "dR":
                observations[i] = drpy[0]
            if channel == "dP":
                observations[i] = drpy[1]
            if channel == "dY":
                observations[i] = drpy[2]
        if self.normal:
            return (observations - self._mean) / self._std
        else:
            return observations

    def reset(self):
        self.first_time = True

'''=========================== NOT USE ==============================='''
class SimpleFootForceSensor(sensor.BoxSpaceSensor):
    """A sensor that reads the contact force of a robot."""

    def __init__(self,
                 lower_bound: _FLOAT_OR_ARRAY = -1000,
                 upper_bound: _FLOAT_OR_ARRAY = 1000,
                 name: typing.Text = "FootForceSensor",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs PoseSensor.

        Args:
          lower_bound: the lower bound of the pose of the robot.
          upper_bound: the upper bound of the pose of the robot.
          name: the name of the sensor.
          dtype: data type of sensor value.
        """
        super(SimpleFootForceSensor, self).__init__(
            name=name,
            shape=(8,),  # x, y, orientation
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)

    def _get_observation(self) -> _ARRAY:
        # print(self._robot.GetFootContactsForce())
        return self._robot.GetFootContactsForce(mode='simple')

    def reset(self):
        pass

class FootContactSensor(sensor.BoxSpaceSensor):
    """A sensor that reads the contact force of a robot."""

    def __init__(self,
                 lower_bound: _FLOAT_OR_ARRAY = -1000,
                 upper_bound: _FLOAT_OR_ARRAY = 1000,
                 name: typing.Text = "FootContactSensor",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs PoseSensor.

        Args:
          lower_bound: the lower bound of the pose of the robot.
          upper_bound: the upper bound of the pose of the robot.
          name: the name of the sensor.
          dtype: data type of sensor value.
        """
        super(FootContactSensor, self).__init__(
            name=name,
            shape=(4,),  # x, y, orientation
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)

    def _get_observation(self) -> _ARRAY:
        # print(self._robot.GetFootContactsForce())
        return np.array(self._robot.GetFootContactsForce(mode='simple')[:4]).reshape(-1)

    def reset(self):
        pass

'''============================= NOT USE ================================='''
class FootPoseSensor(sensor.BoxSpaceSensor):
    """A sensor that reads the contact force of a robot."""

    def __init__(self,
                 lower_bound: _FLOAT_OR_ARRAY = -1000,
                 upper_bound: _FLOAT_OR_ARRAY = 1000,
                 normal: int = 0,
                 name: typing.Text = "FootPoseSensor",
                 dtype: typing.Type[typing.Any] = np.float64) -> None:
        """Constructs PoseSensor.

        Args:
          lower_bound: the lower bound of the pose of the robot.
          upper_bound: the upper bound of the pose of the robot.
          name: the name of the sensor.
          dtype: data type of sensor value.
        """
        super(FootPoseSensor, self).__init__(
            name=name,
            shape=(12,),  # x, y, orientation
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)
        self.normal = normal
        self._mean = np.array([1.7454079e-01, -1.5465108e-01, -2.0661314e-01, 1.7080666e-01,
                               1.6490668e-01, -2.0865265e-01, -1.9902834e-01, -1.2880404e-01,
                               -2.3593837e-01, -2.0215839e-01, 1.3673349e-01, -2.3642859e-01])
        self._std = np.array([3.9058894e-02, 2.4757426e-02, 4.2747084e-02,
                              4.1128017e-02, 2.7591322e-02, 4.3003809e-02, 4.3018311e-02, 2.8423777e-02,
                              4.7990609e-02, 4.6113804e-02, 2.8037265e-02, 4.9409315e-02])

    def _get_observation(self) -> _ARRAY:
        if self.normal:
            return (np.array(self._robot.GetFootPositionsInBaseFrame()).reshape(-1) - self._mean) / self._std
        else:
            return np.array(self._robot.GetFootPositionsInBaseFrame()).reshape(-1)

    def reset(self):
        pass