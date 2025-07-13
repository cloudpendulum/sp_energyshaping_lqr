import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
import time
import wget
import subprocess
from pathlib import Path

class PendulumPlant:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81, inertia=None, torque_limit=np.inf):
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        if inertia is None:
            self.I = mass*length*length
        else:
            self.I = inertia
        self.torque_limit = torque_limit

        self.dof = 1
        self.x = np.zeros(2*self.dof) #position, velocity
        self.t = 0.0 #time

        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x

    def forward_kinematics(self, pos):
        """
        forward kinematics, origin at fixed point
        """
        ee_pos_x = self.l * np.sin(pos)
        ee_pos_y = -self.l*np.cos(pos)
        return [ee_pos_x, ee_pos_y]

    def forward_dynamics(self, pos, vel, tau):
        """
        return acceleration from current position, velocity and torque.
        use self.m, self.g, self.l, self.b and self.I if needed
        """
        torque = np.clip(tau, -self.torque_limit, self.torque_limit)

        accn = (torque - self.m*self.g*self.l*np.sin(pos) - self.b*vel) / self.I

        #print(torque)
        
        return accn

    def inverse_dynamics(self, pos, vel, accn):
        """
        return torque acting on the revolute joint (tau) in terms of inputs
        use self.m, self.g, self.l, self.b and self.I if needed
        """
        tau_id = accn*self.I + self.m*self.g*self.l*np.sin(pos) + self.b*vel

        return tau_id

    def rhs(self, t, x, tau):

        """
        Computes the integrand of the equations of motion.
        """
        accn = self.forward_dynamics(x[0], x[1], tau)
        integ = np.array([x[1], accn])
        return integ

    def euler_integrator(self, t, y, dt, tau):
        """
        Implement Forward Euler Integration for a time-step dt and state y
        y = [pos, vel]
        """
        integ = self.rhs(t, y, tau)
        y_new = y + dt*integ
        return y_new

    def runge_integrator(self, t, y, dt, tau):
        """
        Bonus: Implement a fourth order Runge-Kutta Integration scheme
        """
        k1 = self.rhs(t, y, tau)
        k2 = self.rhs(t + 0.5*dt, y + 0.5*dt*k1, tau)
        k3 = self.rhs(t + 0.5*dt, y + 0.5*dt*k2, tau)
        k4 = self.rhs(t + dt, y + dt*k3, tau)
        integ = (k1 + 2*(k2 + k3) + k4) / 6.0

        y_new = y + dt*integ

        return y_new

    def step(self, tau, dt, integrator="euler"):
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        if integrator == "runge_kutta":
            self.x = self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x = self.euler_integrator(self.t, self.x, dt, tau)
        self.t += dt
        # Store the time series output
        self.t_values.append(self.t)
        self.x_values.append(self.x.copy())
        self.tau_values.append(tau)

    def simulate(self, t0, y0, tf, dt, controller=None, integrator="euler"):
        self.set_state(t0, y0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []

        while (self.t <= tf):
            if controller is not None:
                tau = controller.get_control_output(self.x)
            else:
                tau = 0
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values

    def simulate_and_animate(self, t0, y0, tf, dt, controller=None, integrator="euler", save_video=False):
        """
        simulate and animate the pendulum
        """
        self.set_state(t0, y0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []

        #fig = plt.figure(figsize=(6,6))
        #self.animation_ax = plt.axes()
        fig, (self.animation_ax, self.ps_ax) = plt.subplots(1, 2, figsize=(10, 5))
        self.animation_plots = []
        ee_plot, = self.animation_ax.plot([], [], "o", markersize=25.0, color="blue")
        bar_plot, = self.animation_ax.plot([], [], "-", lw=5, color="black")
        #text_plot = self.animation_ax.text(0.1, 0.1, [], xycoords="figure fraction")
        self.animation_plots.append(ee_plot)
        self.animation_plots.append(bar_plot)

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps*[par_dict]

        #ps_fig = plt.figure(figsize=(6,6))
        #self.ps_ax = plt.axes()
        #self.ps_plots = []
        ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
        #self.ps_plots.append(ps_plot)
        self.animation_plots.append(ps_plot)

        animation = FuncAnimation(fig, self._animation_step, frames=frames, init_func=self._animation_init, blit=True, repeat=False, interval=dt*1000)
        animation2 = None
        #if phase_plot:
        #    animation2 = FuncAnimation(fig, self._ps_update, init_func=self._ps_init, blit=True, repeat=False, interval=dt*1000)

        if save_video:
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            animation.save('pendulum_swingup.mp4', writer=writer)
            #if phase_plot:
            #    Writer2 = mplanimation.writers['ffmpeg']
            #    writer2 = Writer2(fps=60, bitrate=1800)
            #    animation2.save('pendulum_swingup_phase.mp4', writer=writer2)
        #plt.show()

        return self.t_values, self.x_values, self.tau_values, animation#, animation2

    def _animation_init(self):
        """
        init of the animation plot
        """
        self.animation_ax.set_xlim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_ylim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")
        for ap in self.animation_plots:
            ap.set_data([], [])

        self._ps_init()
        return self.animation_plots

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """
        dt = par_dict["dt"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        if controller is not None:
            tau = controller.get_control_output(self.x)
        else:
            tau = 0
        self.step(tau, dt, integrator=integrator)
        ee_pos = self.forward_kinematics(self.x[0])
        #self.animation_plots[0].set_data(ee_pos[0], ee_pos[1])
        self.animation_plots[0].set_data((ee_pos[0],), (ee_pos[1],))
        self.animation_plots[1].set_data([0, ee_pos[0]], [0, ee_pos[1]])

        self._ps_update(0)

        return self.animation_plots

    def _ps_init(self):
        """
        init of the phase space animation plot
        """
        self.ps_ax.set_xlim(-np.pi, 2*np.pi)
        self.ps_ax.set_ylim(-10, 10)
        self.ps_ax.set_xlabel("degree [rad]")
        self.ps_ax.set_ylabel("velocity [rad/s]")
        for ap in self.animation_plots:
            ap.set_data([], [])
        return self.animation_plots

    def _ps_update(self, i):
        """
        update of the phase space animation plot
        """
        self.animation_plots[-1].set_data(np.asarray(self.x_values).T[0], np.asarray(self.x_values).T[1])
        return self.animation_plots

    def CubicTimeScaling(self, Tf, t):
        """Computes s(t) for a cubic time scaling
        Source: Modern Robotics Toolbox (https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py#L1455C1-L1469C61)
        :param Tf: Total time of the motion in seconds from rest to rest
        :param t: The current time t satisfying 0 < t < Tf
        :return: The path parameter s(t) corresponding to a third-order
                 polynomial motion that begins and ends at zero velocity
    
        Example Input:
            Tf = 2
            t = 0.6
        Output:
            0.216
        """
        return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

    def JointTrajectory(self, thetastart, thetaend, Tf, N):
        """Computes a straight-line trajectory in joint space
        Source: Modern Robotics Toolbox (modified) 
        :param thetastart: The initial joint variables
        :param thetaend: The final joint variables
        :param Tf: Total time of the motion in seconds from rest to rest
        :param N: The number of points N > 1 (Start and stop) in the discrete
                  representation of the trajectory
        :return: A trajectory as an N x n matrix, where each row is an n-vector
                 of joint variables at an instant in time. The first row is
                 thetastart and the Nth row is thetaend . The elapsed time
                 between each row is Tf / (N - 1)
    
        Example Input:
            thetastart = np.array([1, 0, 0, 1, 1, 0.2, 0,1])
            thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1])
            Tf = 4
            N = 6
            method = 3
        Output:
            np.array([[     1,     0,      0,      1,     1,    0.2,      0, 1]
                      [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1]
                      [1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1]
                      [1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1]
                      [1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1]
                      [   1.2,   0.5,    0.6,    1.1,     2,      2,    0.9, 1]])
        """
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = np.zeros((len(thetastart), N))
        for i in range(N):
            s = self.CubicTimeScaling(Tf, timegap * i)
            traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
        traj = np.array(traj).T
        return traj

    def run_on_hardware(self, tf, dt, controller=None, user_token = None, x0 = None, preparation_time = 0.0):

        import time
        
        if user_token is None:
            from cloudpendulumlocal.cloud_pendulum_local import Client
            user_token = ""
        else:
            from cloudpendulumclient.client import Client

        self.c = Client()
        
        session_token, self.live_url = self.c.start_experiment(
            user_token = user_token,
            experiment_type = "SimplePendulum",
            experiment_time = tf,
            preparation_time = preparation_time,
            record = True
        )
        print("Your experiment hash key is:", session_token)
            
        if x0 is not None:
            print("You have specified a desired state to start this experiment, x0: ", x0)
            print("Note: We will try to reach only the Desired Initial Position approximately. Initial Desired velocity will be ignored!")
            thetastart = [self.c.get_position(session_token)]
            thetaend = x0[0]
            tf0 = 2
            N0 = int(tf0 / dt)
            print("Going to pos:", thetaend, " and time taken is subtracted from your tf.")
            traj = self.JointTrajectory(thetastart, thetaend, tf0, N0)
            self.c.set_impedance_controller_params(0.25, 0.0015, session_token)
            i = 0
            meas_time = 0.0
            meas_dt = 0.0
            while meas_time < tf0 and i < N0:
                start_loop = time.time()
                meas_time += meas_dt
                self.c.set_position(traj[i], session_token)
                i = i + 1
                while time.time() - start_loop < dt:
                    pass
                meas_dt = time.time() - start_loop
            print("Desired position reached after: ", meas_time, " seconds.")
            print("Experiment will start from the state: ", [self.c.get_position(session_token), self.c.get_velocity(session_token)])
            tf = tf - meas_time
            print("Your remaining experimentation time is: ", tf, " seconds.")
            
        n = int(tf / dt)

        meas_time_vec = np.zeros(n)
        meas_pos = np.zeros(n)
        meas_vel = np.zeros(n)
        meas_tau = np.zeros(n)
        des_tau = np.zeros(n)

        # defining runtime variables
        i = 0
        meas_dt = 0.0
        meas_time = 0.0

        print("Control Loop Started!")
            
        self.c.set_impedance_controller_params(0.0, 0.0, session_token)

        # Control loop
        while meas_time < tf and i < n:
            start_loop = time.time()
            meas_time += meas_dt
            
            ## Do your stuff here - START
            measured_position = self.c.get_position(session_token)
            measured_velocity = self.c.get_velocity(session_token)
            measured_torque = self.c.get_torque(session_token)
            
            self.x = np.array([measured_position, measured_velocity])
            
            # Control logic
            if controller is not None:
                tau = controller.get_control_output(self.x)
                self.c.set_torque(tau, session_token)
            else:
                tau = 0                
                       
            # Collect data for plotting
            meas_time_vec[i] = meas_time
            meas_pos[i] = measured_position
            meas_vel[i] = measured_velocity    
            meas_tau[i] = measured_torque
            des_tau[i] = tau 
                
            ## Do your stuff here - END
            
            i += 1
            exec_time = time.time() - start_loop
            if exec_time > dt:
                print("Control loop is too slow!")
                print("Control frequency:", 1/exec_time, "Hz")
                print("Desired frequency:", 1/dt, "Hz")
                print()
            while time.time() - start_loop < dt:
                pass
            meas_dt = time.time() - start_loop
        print("Control Loop Ended!")
        
        download_url = self.c.stop_experiment(session_token)
        print("Experiment Finished!")          
        
        # Inform user about their current token status
        if user_token is not None:
            print("Your current user token status is: ", self.c.get_user_info(user_token))
        
        filename = wget.download(download_url,".")
        self.vod_filepath = f'{Path(filename).stem}.mp4'
        self.convert_flv_to_mp4(f'{filename}', self.vod_filepath)
        
        self.t_values = meas_time_vec
        self.x_values = np.vstack((meas_pos, meas_vel)).T
        self.tau_values = meas_tau
        self.des_tau_values = des_tau
        
        return self.t_values, self.x_values, self.tau_values, self.des_tau_values, self.vod_filepath
    
    def convert_flv_to_mp4(self, input_path, output_path):
        """
        Convert an FLV file to MP4 using FFmpeg.
    
        :param input_path: Path to the input FLV file.
        :param output_path: Path to the output MP4 file.
        """
        command = [
            "ffmpeg",
            "-i", input_path,    # Input file
            "-c:v", "copy",      # Copy video stream
            "-c:a", "copy",      # Copy audio stream
            output_path          # Output file
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode == 0:
            print(f"Conversion successful: {output_path}")
        else:
            print(f"Error during conversion: {process.stderr.decode()}")


def plot_timeseries(T, X, U):
    plt.plot(T, np.asarray(X).T[0], label="theta")
    plt.plot(T, np.asarray(X).T[1], label="theta dot")
    plt.plot(T, U, label="u")
    plt.legend(loc="best")
    plt.show()

