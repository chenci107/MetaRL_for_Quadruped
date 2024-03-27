import sys
sys.path.append('../../../../')

from JYLite_env_meta.TG_and_IK.Kinematics.JYLiteKinematics import JYLiteModel
from JYLite_env_meta.TG_and_IK.GaitGenerator.Bezier import BezierGait
from JYLite_env_meta import *
import time

def main():

    max_timesteps = 4e6

    env = JYLiteGymEnv(task="ground",render=True,)
    env.reset()

    JYLite = JYLiteModel()
    T_bf0 = JYLite.WorldToFoot
    bzg = BezierGait(dt=0.01)

    t = 0
    while t < (int(max_timesteps)):

        StepLength = 0.03
        LateralFraction=0.0
        YawRate=0.0
        StepVelocity=0.0010000000474974513
        ClearanceHeight=0.04500000178813934
        PenetrationDepth=0.003000000026077032
        orn = [0,0,0]
        pos = [0,0,0]

        T_bf = bzg.GenerateTrajectory(L=StepLength,
                                      LateralFraction=LateralFraction,
                                      YawRate=YawRate,
                                      vel=StepVelocity,
                                      T_bf_=T_bf0,
                                      clearance_height=ClearanceHeight,
                                      penetration_depth=PenetrationDepth, )

        joint_angles = JYLite.IK(orn, pos, T_bf)
        joint_angles = joint_angles.reshape(-1)
        state, reward, done, _ = env.step(joint_angles)
        time.sleep(1/120.)

        t += 1

    env.close()

if __name__ == "__main__":
    main()