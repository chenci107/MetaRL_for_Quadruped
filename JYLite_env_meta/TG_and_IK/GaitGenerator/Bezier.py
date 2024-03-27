#!/usr/bin/env python

import numpy as np
from JYLite_env_meta.TG_and_IK.Kinematics.LieAlgebra import TransToRp
import copy

STANCE = 0
SWING = 1

# Bezier Curves from: https://dspace.mit.edu/handle/1721.1/98270
# Rotation Logic from: http://www.inase.org/library/2014/santorini/bypaper/ROBCIRC/ROBCIRC-54.pdf


class BezierGait():
    def __init__(self, dSref=[0.0, 0.0, 0.5, 0.5], dt=0.01, Tswing=0.2):
        # Phase Lag Per Leg: FL, FR, BL, BR
        # Reference Leg is FL, always 0
        self.dSref = dSref              # self.dSref is the phase lag of leg i with respect to the reference leg
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]
        self.NumControlPoints = 11      # Number of control points is n + 1 = 11 + 1 = 12
        self.dt = dt                    # Timestep, 0.01

        self.time = 0.0                 # Total Elapsed Time
        self.TD_time = 0.0              # Touchdown Time
        self.time_since_last_TD = 0.0   # Time Since Last Touchdown

        self.StanceSwing = SWING        # Trajectory Mode, indicate swing or stance of the reference leg
        self.SwRef = 0.0                # Swing Phase value [0, 1] of Reference Foot, the swing phase of the reference leg, see GetPhase()
        # self.Stref = 0.0
        self.TD = False                 # Whether Reference Foot has Touched Down

        self.Tswing = Tswing            # Swing time, the value is 0.2

        self.ref_idx = 0                # Reference Leg

        self.Phases = self.dSref        # Store all leg phases

        self.plot_x_FL = []
        self.plot_z_FL = []
        self.plot_x_FR = []
        self.plot_z_FR = []
        self.plot_x_BL = []
        self.plot_z_BL = []
        self.plot_x_BR = []
        self.plot_z_BR = []
        self.total_plot_steps = 0

    def reset(self):
        """Resets the parameters of the Bezier Gait Generator
        """
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]
        self.time = 0.0
        self.TD_time = 0.0
        self.time_since_last_TD = 0.0
        self.StanceSwing = SWING
        self.SwRef = 0.0
        self.TD = False

    def GetPhase(self, index, Tstance, Tswing):   # IJRR, e.q.(15) and (16)
        """Retrieves the phase of an individual leg.

        NOTE modification
        from original paper:

        if ti < -Tswing:
           ti += Tstride

        This is to avoid a phase discontinuity if the user selects
        a Step Length and Velocity combination that causes Tstance > Tswing.

        :param index: the leg's index, used to identify the required
                      phase lag
        :param Tstance: the current user-specified stance period
        :param Tswing: the swing period (constant, class member)
        :return: Leg Phase, and StanceSwing (bool) to indicate whether
                 leg is in stance or swing mode
        """
        StanceSwing = STANCE
        Sw_phase = 0.0
        Tstride = Tstance + Tswing
        ti = self.Get_ti(index, Tstride)

        # NOTE: PAPER WAS MISSING THIS LOGIC!!
        if ti < -Tswing:
            ti += Tstride

        ### Stance ###
        if ti >= 0.0 and ti <= Tstance:
            StanceSwing = STANCE
            if Tstance == 0.0:
                Stnphase = 0.0
            else:
                Stnphase = ti / float(Tstance)
            if index == self.ref_idx:
                self.StanceSwing = StanceSwing
            return Stnphase, StanceSwing

        ### Swing ###
        elif ti >= -Tswing and ti < 0.0:
            StanceSwing = SWING
            Sw_phase = (ti + Tswing) / Tswing
        elif ti > Tstance and ti <= Tstride:
            StanceSwing = SWING
            Sw_phase = (ti - Tstance) / Tswing
        if Sw_phase >= 1.0:            # Touchdown at End of Swing
            Sw_phase = 1.0
        if index == self.ref_idx:
            self.StanceSwing = StanceSwing
            self.SwRef = Sw_phase
            if self.SwRef >= 0.999:    # REF Touchdown at End of Swing
                self.TD = True
        return Sw_phase, StanceSwing

    def Get_ti(self, index, Tstride):  # e.q.(14), get ti
        """Retrieves the time index for the individual leg

        :param index: the leg's index, used to identify the required
                      phase lag
        :param Tstride: the total leg movement period (Tstance + Tswing)
        :return: the leg's time index
        """
        # NOTE: for some reason python's having numerical issues w this
        # setting to 0 for ref leg by force
        if index == self.ref_idx:
            self.dSref[index] = 0.0
        ti = self.time_since_last_TD - self.dSref[index] * Tstride
        return ti

    def Increment(self, dt, Tstride): #  IJRR, e.q.(12), get self.time_since_last_TD
        """Increments the Bezier gait generator's internal clock (self.time)
        :param dt: the time step
                      phase lag
        :param Tstride: the total leg movement period (Tstance + Tswing)
        :return: the leg's time index
        """
        self.CheckTouchDown()                                # get self.TD_time
        self.time_since_last_TD = self.time - self.TD_time   # get self.time_since_last_TD
        if self.time_since_last_TD > Tstride:
            self.time_since_last_TD = Tstride
        elif self.time_since_last_TD < 0.0:
            self.time_since_last_TD = 0.0
        # Increment Time at the end in case TD just happened
        # So that we get time_since_last_TD = 0.0
        self.time += dt

        # If Tstride = Tswing, Tstance = 0
        # RESET ALL
        if Tstride < self.Tswing + dt:
            self.time = 0.0
            self.time_since_last_TD = 0.0
            self.TD_time = 0.0
            self.SwRef = 0.0

    def CheckTouchDown(self):    # IJRR, e.q.(13), get self.TD_time
        """Checks whether a reference leg touchdown
           has occured, and whether this warrants
           resetting the touchdown time
        """
        if self.SwRef >= 0.9 and self.TD:
            self.TD_time = self.time
            self.TD = False
            self.SwRef = 0.0

    def BernSteinPoly(self, t, k, point):
        """Calculate the point on the Berinstein Polynomial
           based on phase (0->1), point number (0-11),
           and the value of the control point itself

           :param t: phase
           :param k: point number
           :param point: point value
           :return: Value through Bezier Curve
        """
        return point * self.Binomial(k) * np.power(t, k) * np.power(1 - t, self.NumControlPoints - k)

    def Binomial(self, k):
        """Solves the binomial theorem given a Bezier point number
           relative to the total number of Bezier points.

           :param k: Bezier point number
           :returns: Binomial solution
        """
        return np.math.factorial(self.NumControlPoints) / (np.math.factorial(k) * np.math.factorial(self.NumControlPoints - k))

    def BezierSwing(self, phase, L, LateralFraction, clearance_height=0.04):  # IJRR, e.q.(20)
        """Calculates the step coordinates for the Bezier (swing) period

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param clearance_height: foot clearance height during swing phase

           :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """

        # Polar Leg Coords
        X_POLAR = np.cos(LateralFraction)
        Y_POLAR = np.sin(LateralFraction)

        # Bezier Curve Points (12 pts)
        # NOTE: L is HALF of STEP LENGTH
        # Forward Component
        STEP = np.array([
            -L,  # Ctrl Point 0, half of stride len
            -L * 1.4,  # Ctrl Point 1 diff btwn 1 and 0 = x Lift vel
            -L * 1.5,  # Ctrl Pts 2, 3, 4 are overlapped for
            -L * 1.5,  # Direction change after
            -L * 1.5,  # Follow Through
            0.0,  # Change acceleration during Protraction
            0.0,  # So we include three
            0.0,  # Overlapped Ctrl Pts: 5, 6, 7
            L * 1.5,  # Changing direction for swing-leg retraction
            L * 1.5,  # requires double overlapped Ctrl Pts: 8, 9
            L * 1.4,  # Swing Leg Retraction Velocity = Ctrl 11 - 10
            L
        ])
        # Account for lateral movements by multiplying with polar coord.
        # LateralFraction switches leg movements from X over to Y+ or Y-
        # As it tends away from zero
        X = STEP * X_POLAR

        # Account for lateral movements by multiplying with polar coord.
        # LateralFraction switches leg movements from X over to Y+ or Y-
        # As it tends away from zero
        Y = STEP * Y_POLAR

        # Vertical Component
        Z = np.array([
            0.0,  # Double Overlapped Ctrl Pts for zero Lift
            0.0,  # Veloicty wrt hip (Pts 0 and 1)
            clearance_height * 0.9,  # Triple overlapped control for change in
            clearance_height * 0.9,  # Force direction during transition from
            clearance_height * 0.9,  # follow-through to protraction (2, 3, 4)
            clearance_height * 0.9,  # Double Overlapped Ctrl Pts for Traj
            clearance_height * 0.9,  # Dirctn Change during Protraction (5, 6)
            clearance_height * 1.1,  # Maximum Clearance at mid Traj, Pt 7
            clearance_height * 1.1,  # Smooth Transition from Protraction
            clearance_height * 1.1,  # To Retraction, Two Ctrl Pts (8, 9)
            0.0,  # Double Overlap Ctrl Pts for 0 Touchdown
            0.0,  # Veloicty wrt hip (Pts 10 and 11)
        ])

        stepX = 0.
        stepY = 0.
        stepZ = 0.
        # Bernstein Polynomial sum over control points
        for i in range(len(X)):
            stepX += self.BernSteinPoly(phase, i, X[i])
            stepY += self.BernSteinPoly(phase, i, Y[i])
            stepZ += self.BernSteinPoly(phase, i, Z[i])

        return stepX, stepY, stepZ

    def SineStance(self, phase, L, LateralFraction, penetration_depth=0.00):  # IJRR, e.q.(23) and (24)
        """Calculates the step coordinates for the Sinusoidal stance period

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param penetration_depth: foot penetration depth during stance phase

           :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """
        X_POLAR = np.cos(LateralFraction)  # LateralFraction = 0 ==> X_POLAR = 1
        Y_POLAR = np.sin(LateralFraction)  # LateralFraction = 0 ==> Y_POLAR = 0
        # moves from +L to -L
        step = L * (1.0 - 2.0 * phase)
        stepX = step * X_POLAR  # step * 1 = step
        stepY = step * Y_POLAR  # step * 0 = 0
        if L != 0.0:
            stepZ = -penetration_depth * np.cos((np.pi * (stepX + stepY)) / (2.0 * L))
        else:
            stepZ = 0.0
        return stepX, stepY, stepZ

    def YawCircle(self, T_bf, index):
        """ Calculates the required rotation of the trajectory plane
            for yaw motion

           :param T_bf: default body-to-foot Vector
           :param index: the foot index in the container
           :returns: phi_arc, the plane rotation angle required for yaw motion
        """

        # Foot Magnitude depending on leg type
        DefaultBodyToFoot_Magnitude = np.sqrt(T_bf[0]**2 + T_bf[1]**2)

        # Rotation Angle depending on leg type
        DefaultBodyToFoot_Direction = np.arctan2(T_bf[1], T_bf[0])

        # Previous leg coordinates relative to default coordinates
        g_xyz = self.Prev_fxyz[index] - np.array([T_bf[0], T_bf[1], T_bf[2]])

        # Modulate Magnitude to keep tracing circle
        g_mag = np.sqrt((g_xyz[0])**2 + (g_xyz[1])**2)
        th_mod = np.arctan2(g_mag, DefaultBodyToFoot_Magnitude)

        # Angle Traced by Foot for Rotation
        # FR and BL
        if index == 1 or index == 2:
            phi_arc = np.pi / 2.0 + DefaultBodyToFoot_Direction + th_mod
        # FL and BR
        else:
            phi_arc = np.pi / 2.0 - DefaultBodyToFoot_Direction + th_mod

        return phi_arc

    def SwingStep(self, phase, L, LateralFraction, YawRate, clearance_height,T_bf, key, index):
        """Calculates the step coordinates for the Bezier (swing) period
           using a combination of forward and rotational step coordinates
           initially decomposed from user input of
           L, LateralFraction and YawRate

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param YawRate: the desired body yaw rate
           :param clearance_height: foot clearance height during swing phase
           :param T_bf: default body-to-foot Vector
           :param key: indicates which foot is being processed
           :param index: the foot index in the container

           :returns: Foot Coordinates relative to unmodified body
        """

        # Yaw foot angle for tangent-to-circle motion
        phi_arc = self.YawCircle(T_bf, index)

        # Get Foot Coordinates for Forward Motion
        X_delta_lin, Y_delta_lin, Z_delta_lin = self.BezierSwing(phase, L, LateralFraction, clearance_height)

        X_delta_rot, Y_delta_rot, Z_delta_rot = self.BezierSwing(phase, YawRate, phi_arc, clearance_height)

        coord = np.array([X_delta_lin + X_delta_rot, Y_delta_lin + Y_delta_rot, Z_delta_lin + Z_delta_rot])

        self.Prev_fxyz[index] = coord

        return coord

    def StanceStep(self, phase, L, LateralFraction, YawRate, penetration_depth,T_bf, key, index):
        """Calculates the step coordinates for the Sine (stance) period
           using a combination of forward and rotational step coordinates
           initially decomposed from user input of
           L, LateralFraction and YawRate

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param YawRate: the desired body yaw rate
           :param penetration_depth: foot penetration depth during stance phase
           :param T_bf: default body-to-foot Vector
           :param key: indicates which foot is being processed
           :param index: the foot index in the container

           :returns: Foot Coordinates relative to unmodified body
        """

        # Yaw foot angle for tangent-to-circle motion
        phi_arc = self.YawCircle(T_bf, index)

        # Get Foot Coordinates for Forward Motion
        X_delta_lin, Y_delta_lin, Z_delta_lin = self.SineStance(phase, L, LateralFraction, penetration_depth)

        X_delta_rot, Y_delta_rot, Z_delta_rot = self.SineStance(phase, YawRate, phi_arc, penetration_depth)

        coord = np.array([X_delta_lin + X_delta_rot, Y_delta_lin + Y_delta_rot,Z_delta_lin + Z_delta_rot])

        self.Prev_fxyz[index] = coord

        return coord

    def GetFootStep(self, L, LateralFraction, YawRate, clearance_height,penetration_depth, Tstance, T_bf, index, key):
        """Calculates the step coordinates in either the Bezier or
           Sine portion of the trajectory depending on the retrieved phase

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param YawRate: the desired body yaw rate
           :param clearance_height: foot clearance height during swing phase
           :param penetration_depth: foot penetration depth during stance phase
           :param Tstance: the current user-specified stance period
           :param T_bf: default body-to-foot Vector
           :param index: the foot index in the container
           :param key: indicates which foot is being processed

           :returns: Foot Coordinates relative to unmodified body
        """
        phase, StanceSwing = self.GetPhase(index, Tstance, self.Tswing)
        if StanceSwing == SWING:
            stored_phase = phase + 1.0
        else:
            stored_phase = phase
        # Just for keeping track
        self.Phases[index] = stored_phase
        if StanceSwing == STANCE:
            return self.StanceStep(phase, L, LateralFraction, YawRate,penetration_depth, T_bf, key, index)
        elif StanceSwing == SWING:
            return self.SwingStep(phase, L, LateralFraction, YawRate,clearance_height, T_bf, key, index)

    def GenerateTrajectory(self,
                           L,                      # StepLength = 0.03
                           LateralFraction,        # LateralFraction = 0.0
                           YawRate,                # YawRate = 0.0
                           vel,                    # StepVelocity = 0.00100
                           T_bf_,                  # T_bf0 = JYPo.WorldToFoot
                           clearance_height=0.06,  # ClearanceHeight = 0.045
                           penetration_depth=0.01, # PenetrationDepth = 0.003
                           contacts=[0, 0, 0, 0],  # contacts = [0,0,0,0]
                           dt=None):
        """Calculates the step coordinates for each foot

           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param YawRate: the desired body yaw rate
           :param vel: the desired step velocity
           :param clearance_height: foot clearance height during swing phase
           :param penetration_depth: foot penetration depth during stance phase
           :param contacts: array containing 1 for contact and 0 otherwise
           :param dt: the time step

           :returns: Foot Coordinates relative to unmodified body
        """
        # First, get Tstance from desired speed and stride length
        # NOTE: L is HALF of stride length
        if vel != 0.0:
            Tstance = 2.0 * abs(L) / abs(vel)  # IJRR, e.q.(11)
        else:
            Tstance = 0.0
            L = 0.0
            self.TD = False
            self.time = 0.0
            self.time_since_last_TD = 0.0

        # Then, get time since last Touchdown and increment time counter
        if dt is None:
            dt = self.dt    # 0.01

        YawRate *= dt

        # Catch infeasible timesteps
        if Tstance < dt:
            Tstance = 0.0
            L = 0.0
            self.TD = False
            self.time = 0.0
            self.time_since_last_TD = 0.0
            YawRate = 0.0
        # NOTE: MUCH MORE STABLE WITH THIS
        elif Tstance > 1.3 * self.Tswing:
            Tstance = 1.3 * self.Tswing    # modify the Tstance to a valid range

        # Check contacts
        if contacts[0] == 1 and Tstance > dt:
            self.TD = True

        self.Increment(dt=dt, Tstride=Tstance + self.Tswing)  # self.Increment(dt, Tstride), get self.TD_time and self.time_since_last_TD

        self.total_plot_steps += 1

        T_bf = copy.deepcopy(T_bf_)
        for i, (key, Tbf_in) in enumerate(T_bf_.items()):
            if key == "FL":
                self.ref_idx = i
                self.dSref[i] = 0.0
            if key == "FR":
                self.dSref[i] = 0.5   # 0.5
            if key == "BL":
                self.dSref[i] = 0.5  # 0.55
            if key == "BR":
                self.dSref[i] = 0.0  # 0.0
            _, p_bf = TransToRp(Tbf_in)
            if Tstance > 0.0:
                step_coord = self.GetFootStep(L=L, LateralFraction=LateralFraction, YawRate=YawRate,
                                              clearance_height=clearance_height,
                                              penetration_depth=penetration_depth, Tstance=Tstance, T_bf=p_bf,
                                              index=i, key=key)

            else:
                step_coord = np.array([0.0, 0.0, 0.0])
            T_bf[key][0, 3] = Tbf_in[0, 3] + step_coord[0]
            T_bf[key][1, 3] = Tbf_in[1, 3] + step_coord[1]
            T_bf[key][2, 3] = Tbf_in[2, 3] + step_coord[2]

        #     if key == "FL":
        #         self.plot_x_FL.append(step_coord[0])
        #         self.plot_z_FL.append(step_coord[2])
        #     elif key == "FR":
        #         self.plot_x_FR.append(step_coord[0])
        #         self.plot_z_FR.append(step_coord[2])
        #     elif key == "BL":
        #         self.plot_x_BL.append(step_coord[0])
        #         self.plot_z_BL.append(step_coord[2])
        #     elif key == "BR":
        #         self.plot_x_BR.append(step_coord[0])
        #         self.plot_z_BR.append(step_coord[2])
        # if self.total_plot_steps >= 46:
        #     ax1=plt.subplot(2, 2, 1)
        #     ax1.axis(xmin=-0.05,xmax=0.05,ymin=-0.01,ymax=0.09)
        #     ax1.plot(self.plot_x_FL,self.plot_z_FL)
        #
        #     ax2=plt.subplot(2, 2, 2)
        #     ax2.axis(xmin=-0.05, xmax=0.05, ymin=-0.01, ymax=0.09)
        #     ax2.plot(self.plot_x_FR,self.plot_z_FR)
        #
        #     ax3=plt.subplot(2, 2, 3)
        #     ax3.axis(xmin=-0.05,xmax=0.05,ymin=-0.01,ymax=0.09)
        #     ax3.plot(self.plot_x_BL,self.plot_z_BL)
        #
        #     ax4=plt.subplot(2, 2, 4)
        #     ax4.axis(xmin=-0.05, xmax=0.05, ymin=-0.01, ymax=0.09)
        #     ax4.plot(self.plot_x_BR,self.plot_z_BR)
        #     plt.show()

        return T_bf
