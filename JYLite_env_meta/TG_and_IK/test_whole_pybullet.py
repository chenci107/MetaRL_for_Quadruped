import sys
sys.path.append('../../../')

from Kinematics.JYLiteKinematics import JYLiteModel
from JYLite_env_meta.TG_and_IK.GaitGenerator.Bezier import BezierGait
from util.gui import GUI
from JYLite_env_meta.TG_and_IK.JYLite_pybullet import JYLite as JYEnv

def main():

    max_timesteps = 4e6

    env = JYEnv()

    g_u_i = GUI(env.robot)

    JYLite = JYLiteModel()
    T_bf0 = JYLite.WorldToFoot
    bzg = BezierGait(dt=env._time_step)

    t = 0
    while t < (int(max_timesteps)):
        pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod = g_u_i.UserInput()
        # StepLength = 0.03

        bzg.Tswing = SwingPeriod

        T_bf = bzg.GenerateTrajectory(L=StepLength,
                                      LateralFraction=LateralFraction,
                                      YawRate=YawRate,
                                      vel=StepVelocity,
                                      T_bf_=T_bf0,
                                      clearance_height=ClearanceHeight,
                                      penetration_depth=PenetrationDepth, )
        # print('The StepLength is:',StepLength)
        # print('The LateralFraction is:',LateralFraction)
        # print('The YawRate is:',YawRate)
        # print('The StepVelocity is:',StepVelocity)
        # print('The ClearanceHeight is:',ClearanceHeight)
        # print('The penetration_depth is:',PenetrationDepth)
        # print('The orn is:',orn)
        # print('The pos is:',pos)



        # print('The T_bf is:',T_bf)

        joint_angles = JYLite.IK(orn, pos, T_bf)
        joint_angles = joint_angles.reshape(-1)

        # env.pass_joint_angles(joint_angles.reshape(-1))

        # fl_leg = joint_angles[0:3]
        # fr_leg = joint_angles[3:6]
        # hl_leg = joint_angles[6:9]
        # hr_leg = joint_angles[9:12]
        # fault_joint_angles = np.concatenate([fl_leg,fr_leg,hl_leg,hr_leg],axis=0)
        # print('The fault_joint_angles is:',fault_joint_angles)


        # fl_leg = joint_angles[0:3]
        # fr_leg = np.array([0.0, -0.82163, 1.8])
        # hl_leg = joint_angles[6:9]
        # hr_leg = joint_angles[9:12]
        # fault_joint_angles = np.concatenate([fl_leg,fr_leg,hl_leg,hr_leg],axis=0)

        # joint_angles[4] = -0.82163
        # fault_joint_angles = joint_angles

        state, reward, done, _ = env.step(joint_angles)
        x,y,z = env.get_pos()
        
        print('The z position is:',z)

        t += 1

    env.close()

if __name__ == "__main__":
    main()
