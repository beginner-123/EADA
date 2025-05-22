from script.dynamic_clpv_ds import *
from scipy.spatial import KDTree

import numpy as np
import pyLasaDataset as lasa
from Algorithms.Learn_NEUM import LearnNeum
from Algorithms.Learn_GPR_ODS import LearnOds

from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.animator import Animator
from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.obstacles import CuboidXd
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.control_robot import BaseRobot
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider

import time
import os


Type = 'Leaf_2'
data = getattr(lasa.DataSet, Type)
demos = data.demos
aa=1

def set_dalpv_ds():
    clpv_ds_data = loadmat('.\clpv_ds_parameter\clpv_ds.mat')
    ds_gmm = clpv_ds_data['ds_gmm'][0, 0]
    A_g = clpv_ds_data['A_k']
    b_g = clpv_ds_data['b_k']
    x = np.random.rand(2, 5)  # Example data
    Priors = ds_gmm['Priors']

    initial_state = [-43.7931034482758, -3.1034482]  # -43.7931034482758,-3.10344827586205
    disturb_index = 50
    disturb_vector = np.array([5, -10])
    options = {
        'i_max': 2000,
        'dt': 0.01,
        'tol': 0.005,
        'limits': [-50, 20, -20, 50]
    }
    data, _ = initinal_lpv_ds(initial_state, options, ds_gmm, A_g, b_g)

    x, xd = simulate_dynamics(initial_state, disturb_index, disturb_vector, options, ds_gmm, A_g, b_g, data)
    disturb_index = 50
    x, xd = dynamic_attractor_lpv_ds_simulation(initial_state, disturb_index, disturb_vector, options, ds_gmm, A_g, b_g,
                                                data)


def get_lpv_parameter(initial_state,options):
    clpv_ds_data = loadmat('.\clpv_ds_parameter\clpv_ds.mat')
    ds_gmm = clpv_ds_data['ds_gmm'][0, 0]
    A_g = clpv_ds_data['A_k']
    b_g = clpv_ds_data['b_k']
    x = np.random.rand(2, 5)  # Example data
    Priors = ds_gmm['Priors']

    data, _ = initinal_lpv_ds(initial_state, options, ds_gmm, A_g, b_g)

    return data,ds_gmm,A_g,b_g


def construct_demonstration_set(demos, start=1, end=-1, gap=5, used_tras=[1, 2, 3, 4, 5, 6]):
    n_tra = len(used_tras)
    x_set = []
    dot_x_set = []
    t_set = []
    for i in range(n_tra):
        x_set.append(demos[used_tras[i]].pos[:, start:end:gap].T)
        dot_x_set.append(demos[used_tras[i]].vel[:, start:end:gap].T)
        t_set.append(demos[used_tras[i]].t[0, start:end:gap])

    x_set = np.array(x_set)
    dot_x_set = np.array(dot_x_set)
    t_set = np.array(t_set)
    return x_set, dot_x_set, t_set

def get_neum_parameter():
    # ---------- Learning (Loading) the neural energy function NEUM ------------
    d_H = 20  
    manually_design_set_neum = construct_demonstration_set(demos, start=1, end=-1, gap=10)
    neum_learner = LearnNeum(manually_design_set=manually_design_set_neum, d_H=d_H, L_2=1e-6)
    print('--- Start energy function training (loading)---')
    beta = 1.0  # tanh的beta
    save_path = '../NeumParameters/Neum_parameter_for_' + Type + '_beta' + str(beta) + '_dH' + str(d_H) + '.txt'
    # Training or Loading
    # neum_parameters = neum_learner.train(save_path=save_path, beta=beta, maxiter=1000)
    neum_parameters = np.loadtxt(save_path)
    neum_learner.neum_parameters = neum_parameters
    print('--- Training (Loading) completed ---')
    print('plotting energy function learning results ...')
    # neum_learner.show_learning_result(neum_parameters, num_levels=10)
    print('Plotting finished')

    return neum_learner

def get_gpr_parameter():
    # ------------------- Learning (Loading) original ADS --------------------
    observation_noise = None
    gamma_oads = 0.5
    manually_design_set_oads = construct_demonstration_set(demos, start=40, end=-1, gap=5)
    ods_learner = LearnOds(manually_design_set=manually_design_set_oads, observation_noise=observation_noise,
                           gamma=gamma_oads)
    print('--- Start original ads training (loading) ---')
    save_path = '../OadsParameters/Oads_parameter_for_' + Type + '.txt'
    # Training or Loading. Using "ods_learner.set_param" when loading parameters
    # oads_parameters=ods_learner.train(save_path=save_path)
    oads_parameters = np.loadtxt(save_path)
    ods_learner.set_param(oads_parameters)
    print('--- Training (Loading) completed ---')
    return ods_learner

def get_gpr_velocity(position=None, ods_learner=None):
    if ods_learner is None:
        raise ValueError("ods_learner object must be provided")
    if position is None:
        position = np.array([[0.5, 0.5]])
    else:
        position=np.array(position).T
        position = np.atleast_2d(position)  

   
    if position.shape[1] != 2:  
        raise ValueError("Position must have exactly two coordinates per point")
    # Prediction
    dot_x = ods_learner.predict(position.reshape(1, 2)).reshape(-1)
    return dot_x


def get_gpr_desired_trajectory(position=None, ods_learner=None):
    if ods_learner is None:
        raise ValueError("ods_learner object must be provided")
    if position is None:
        raise ValueError("please provide position")
    else:
        x = np.array([initial_state]).T 
        xd = []
        N = options['i_max']
        dt = options['dt']
        tol = options['tol']
        for i in range(1, N + 1):

            current_x = x[:, -1].reshape(-1, 1) 
            v_gpr = get_gpr_velocity(current_x, ods_learner).reshape(2,1)
            xd.append(v_gpr)
            x = np.hstack((x, (current_x + np.array(xd[-1]) * dt).reshape(-1, 1)))  

            if i > 3 and (np.all(np.abs(xd[-3:]) < tol) or i > N - 2):
                print(f"Simulation stopped since it reaches the maximum number of allowed iterations {i}")
                print("Exiting without convergence!!! Increase the parameter 'options.i_max' to handle this error.")
                break

        show_figure = False
        if show_figure:
            fig, ax = plt.subplots()
            ax.set_xlim(options['limits'][:2])
            ax.set_ylim(options['limits'][2:])
            ax.scatter(x[0, :], x[1, :], c='black', s=10, label='Simulation with Disturbance')
            ax.legend()
            plt.show()


    return x, xd


class DynamicObstacleAnimator(Animator):
    '''
    this is the class of dynamic_avoider model
    '''

    def __init__(self, it_max, dt_simulation, ds_gmm,A_g,b_g,data,neum_learner=None,ods_learner=None,initial_state=None):
        super().__init__(it_max, dt_simulation)
        # self.ods_learner = ods_learner
        self.ds_gmm = ds_gmm
        self.A_g = A_g
        self.b_g = b_g
        self.neum_learner = neum_learner
        self.ods_learner = ods_learner
        self.data = data
        self.data_kd_tree = KDTree(data.T)
        self.img1 = None
        self.initial_state = initial_state

    def setup(self):
        self.dim=2
        # start_point = np.array([-43.7931034482758,-3.1034482])#这是出发点
        start_point = self.initial_state
        self.robot = BaseRobot(pose=ObjectPose(position=start_point, orientation=0))

        self.environment = ObstacleContainer()
        '''dynamic obstacle'''

        self.environment.append(
            CuboidXd(
                center_position=np.array([-10, 20]),
                orientation=40 * np.pi / 180,#障碍物转向角度
                axes_length=np.array([6, 6]),
                margin_absolut=self.robot.control_radius,
                angular_velocity=10 * np.pi / 180,
            )
        )


        '''random obstacle'''
        np.random.seed(45)#spoon 100 WShape 100
        limits = [-40, 20, -10, 50]
        for i in range(1):
            xmin, xmax, ymin, ymax = limits
            center_x = np.random.uniform(limits[0], limits[1])
            center_y = np.random.uniform(limits[2], limits[3])
            center_position = np.array([center_x, center_y])
            axes_length = np.random.uniform(low=3, high=5, size=2)  
        
            if center_x < (xmin + xmax) / 2 and center_y < (ymin + ymax) / 2:
                linear_velocity = np.random.uniform(low=3, high=5, size=2)
            elif center_x > (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:
                linear_velocity = np.random.uniform(low=-5, high=-3, size=2)
            elif center_x < (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:       
                linear_velocity = np.array([np.random.uniform(3, 5), np.random.uniform(-5, -3)])
            else:
                linear_velocity = np.array([np.random.uniform(-5, -3), np.random.uniform(3, 5)])

            obstacle = EllipseWithAxes(
                center_position=center_position,
                axes_length=axes_length,
                margin_absolut=self.robot.control_radius,
                tail_effect=False,
                linear_velocity =linear_velocity,
            )

            self.environment.append(obstacle)



        '''
        record : ellipse c:[-27,22],axes:[6,12]
        record : cuboid c:[-10,25],axes:[6,6]
        '''

        self.initial_dynamics = LinearSystem(
            attractor_position=np.array([0, 1.0]), maximum_velocity=1.0
        )

        self.avoider = FastObstacleAvoider(
            obstacle_environment=self.environment,
            robot=self.robot,
        )

        self.dynamics_avoider=ModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.environment,
            # robot=self.robot,
        )

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = start_point
        # Visualize Afterwards
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8))

    def generate_random_obstacles(self, n_obstacles, limits=[-55, 5, -25, 15]):
        xmin, xmax, ymin, ymax = limits

        for _ in range(n_obstacles):
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])

            if edge == 'top':
                center_x = np.random.uniform(xmin, xmax)
                center_y = ymax
            elif edge == 'bottom':
                center_x = np.random.uniform(xmin, xmax)
                center_y = ymin
            elif edge == 'left':
                center_x = xmin
                center_y = np.random.uniform(ymin, ymax)
            else:  # right
                center_x = xmax
                center_y = np.random.uniform(ymin, ymax)

            center_position = np.array([center_x, center_y])

            axes_length = np.random.uniform(low=5, high=8, size=2)

            if center_x < (xmin + xmax) / 2 and center_y < (ymin + ymax) / 2:
                linear_velocity = np.random.uniform(low=0, high=5, size=2)
            elif center_x > (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:
                linear_velocity = np.random.uniform(low=-5, high=0, size=2)
            elif center_x < (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:
                linear_velocity = np.array([np.random.uniform(0, 5), np.random.uniform(-5, 0)])
            else:
                linear_velocity = np.array([np.random.uniform(-5, 0), np.random.uniform(0, 5)])

            obstacle = EllipseWithAxes(
                center_position=center_position,
                axes_length=axes_length,
                linear_velocity=linear_velocity,
                margin_absolut=self.robot.control_radius,
            )
            self.environment.append(obstacle)




    def get_gpr_velocity(self,position=None, ods_learner=None):
        if ods_learner is None:
            raise ValueError("ods_learner object must be provided")
        if position is None:
            position = np.array([[0.5, 0.5]])
        else:
            position=np.array(position).T
            position = np.atleast_2d(position)  

        if position.shape[1] != 2:  
            raise ValueError("Position must have exactly two coordinates per point")
        # Prediction
        dot_x = ods_learner.predict(position.reshape(1, 2)).reshape(-1)
        return dot_x

    def get_gpr_subattractor_velocity(self,position=None, ods_learner=None):
        if ods_learner is None:
            raise ValueError("ods_learner object must be provided")
        if position is None:
            position = np.array([[0.5, 0.5]])
        else:
            position=np.array(position).T
            position = np.atleast_2d(position)  
        z = position.T
        zd = []
        tol_z = 0.5
        C = 5 * np.eye(2)  

        distance, nearest_index = self.data_kd_tree.query(z[:, -1])
        nearest_point = data[:, nearest_index]

        if np.all(np.abs(z[:, -1] - data[:, nearest_index]) < tol_z):
            current_z = z[:, -1].reshape(-1, 1)  
            v_gpr = get_gpr_velocity(current_z, ods_learner).reshape(2, 1)
            new_zd = v_gpr
        else:
            current_z = z[:, -1].reshape(-1, 1) 
            c1 = C.dot(data[:, nearest_index] - z[:, -1]).reshape(-1, 1)
            c2 = get_gpr_velocity(current_z, ods_learner).reshape(2, 1)
            new_zd = c1 + c2
        return new_zd.ravel()

    def func_rho(self,position):
        '''
        Define function rho(x),
        see Eq.(57) in the paper
        '''
        gamma = np.max(np.sqrt(np.sum(self.neum_learner.dot_x_set ** 2, axis=1))) / 1e3
        dvdx = self.neum_learner.dvdx(self.neum_learner.neum_parameters, position)
        return np.sqrt(np.dot(dvdx, dvdx)) * gamma

    def get_sds_velocity(self,position=None, neum_learner=None,initial_velocity=None):
        d_x=np.shape(position)[0]
        if neum_learner is None:
            raise ValueError("neum_learner object must be provided")
        if position is None:
            raise ValueError("position must be provided")
        else:
            # position = np.atleast_2d(position)
            dvdx = self.neum_learner.dvdx(self.neum_learner.neum_parameters, position)
            dvdx_norm_2 = np.dot(dvdx, dvdx)
            # ods_dot_x = self.ods_learner.predict(position.reshape(1, 2)).reshape(-1)
            ods_dot_x = initial_velocity
            rho = self.func_rho(position)
            temp = np.dot(ods_dot_x, dvdx) + rho
            if temp > 0:
                u = -temp / dvdx_norm_2 * dvdx
                dot_x = ods_dot_x + u
            else:
                u = np.zeros(self.neum_learner.d_x)
                dot_x = ods_dot_x
            return dot_x


    def update_step(self, ii):
        if not (ii % 10):
            print(f"It {ii}")

        import time
        start_time = time.time()
        # Initial Dynmics and Avoidance
        testvalue = self.robot.pose.position

        self.initial_velocity = self.get_gpr_velocity(self.robot.pose.position,self.ods_learner)/3
        self.modulated_velocity = self.dynamics_avoider.avoid(velocity=self.initial_velocity,position=self.robot.pose.position)

        end_time = time.time()
        print("Time taken for initial velocity calculation: ", end_time - start_time)


        self.robot.pose.position = (
            self.robot.pose.position + self.modulated_velocity * self.dt_simulation
        )

        self.position_list[:, ii + 1] = (
            self.modulated_velocity * self.dt_simulation + self.position_list[:, ii]
        )





        self.plot_environment(ii=ii)
        if ii%40==0:
            self.generate_random_obstacles(n_obstacles=4, limits=[-40, 20, -10, 50])


    def plot_environment(self, ii):
        # self.ax.clear()

        self.ax.cla()

        self.environment.do_velocity_step(delta_time=self.dt_simulation)

        plot_obstacles(
            obstacle_container=self.environment,
            ax=self.ax,
            x_lim=[options['limits'][0], options['limits'][1]],
            y_lim=[options['limits'][2], options['limits'][3]],
            draw_reference=True,
            # obstacle_color="red",
        )

        self.ax.plot(
            self.robot.pose.position[0],
            self.robot.pose.position[1],
            "o",
            color="k",
            markersize=18,
        )

        self.ax.plot(
            # self.initial_dynamics.attractor_position[0],
            # self.initial_dynamics.attractor_position[1],
            0,
            0,
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        self.ax.plot(
            self.position_list[0, : ii + 1],
            self.position_list[1, : ii + 1],
            ":",
            color="#135e08",
            linewidth=5,
            # markersize=12,
        )
        self.ax.plot(
            self.position_list[0, ii + 1],
            self.position_list[1, ii + 1],
            "o",
            color="#135e08",
            markersize=12,
        )


        self.ax.scatter(data[0, :], data[1, :], c='red', s=10, label='Simulation with Disturbance')



    def plot_color_gradient_based_on_distance(self,track_data, xmin, xmax, ymin, ymax, canvas_size=(100, 100), color_map='viridis',
                                     alpha=0.7):

        x_canvas = np.linspace(xmin, xmax, canvas_size[0])
        y_canvas = np.linspace(ymin, ymax, canvas_size[1])
        xv, yv = np.meshgrid(x_canvas, y_canvas)

        track_points = np.vstack((track_data[0], track_data[1])).T
        kdtree = KDTree(track_points)

        grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
        distances, _ = kdtree.query(grid_points)
        distances = distances.reshape(xv.shape)

        norm_distances = (distances - distances.min()) / (distances.max() - distances.min())

        cmap = plt.get_cmap(color_map)
        colors = cmap(1 - norm_distances)  #

        colors[..., 3] = alpha 


        plt.imshow(colors, extent=(xmin, xmax, ymin, ymax), origin='lower')
        # plt.plot(track_data[0], track_data[1], 'k-', linewidth=2)  
        # plt.colorbar(label='Distance to track')
        # plt.title(f'Color Gradient with Transparency using {color_map}')
        plt.show()


    def has_converged(self, ii):
        """Return 0 if still going, and
        >0 : has converged at `ii`
        -1 : stuck somewher away from the attractor
        """
        if (
            LA.norm(self.robot.pose.position - self.initial_dynamics.attractor_position)
            < 1e-1
        ):
            # Check distance to attractor
            return ii

        elif LA.norm(self.modulated_velocity) < 1e-2:
            #  Check Velocity
            return 1

        return 0

if __name__== '__main__':
    initial_state=[-25.537820886783074,-2.113474832009638]#Leaf_2

    options = {
        'i_max': 2000,
        'dt': 0.01,
        'tol': 0.5,
        'limits': [-40, 20, -10, 50]
    }
    # data, ds_gmm, A_g, b_g=get_lpv_parameter(initial_state,options)
    # # data_kd_tree = KDTree(data.T)
    # plt.ion()

    neum_learner = get_neum_parameter()
    ods_learner =get_gpr_parameter()

    data,u=get_gpr_desired_trajectory(initial_state,ods_learner)

    my_animator = DynamicObstacleAnimator(
        it_max=600,
        dt_simulation=0.005,
        ds_gmm=ds_gmm,
        A_g=A_g,
        b_g=b_g,
        data=data,
        neum_learner=neum_learner,
        ods_learner=ods_learner,
        initial_state=initial_state,
    )
    my_animator.setup()
    my_animator.run(save_animation=False)


