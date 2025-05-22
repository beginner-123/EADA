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

# from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.control_robot import BaseRobot
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider

import time
import os


Type = 'Leaf_2'
data = getattr(lasa.DataSet, Type)   #getattr用于返回对象的属性值
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

    # 运行仿真
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
        x_set.append(demos[used_tras[i]].pos[:, start:end:gap].T)  #第i个demos的2d运动轨迹，demos[i].pos是一个2*1000的矩阵 [所有行中采样，start到end，间隔gap] append:添加在x_set数组中
        dot_x_set.append(demos[used_tras[i]].vel[:, start:end:gap].T)# demos.vel是轨迹速度，同样是一个2*1000的数据组
        t_set.append(demos[used_tras[i]].t[0, start:end:gap])#demos.t是矢量输入的对应时间

    x_set = np.array(x_set)
    dot_x_set = np.array(dot_x_set)
    t_set = np.array(t_set)
    return x_set, dot_x_set, t_set

def get_neum_parameter():
    # ---------- Learning (Loading) the neural energy function NEUM ------------
    d_H = 20  # 映射到10维
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
        position = np.atleast_2d(position)  # 确保位置是二维数组

    # 确保位置数组形状正确，适应 ods_learner.predict 方法的输入要求
    if position.shape[1] != 2:  # 假设每个位置应包含两个坐标
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
        x = np.array([initial_state]).T  # 转置以匹配 MATLAB 中的列向量
        xd = []
        N = options['i_max']
        dt = options['dt']
        tol = options['tol']
        for i in range(1, N + 1):

            current_x = x[:, -1].reshape(-1, 1)  # 确保是二维数组
            v_gpr = get_gpr_velocity(current_x, ods_learner).reshape(2,1)
            xd.append(v_gpr)
            x = np.hstack((x, (current_x + np.array(xd[-1]) * dt).reshape(-1, 1)))  # 更新 x 并保持二维

            # 检查收敛
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
        # 调用父类构造函数，不传入 ods_learner 参数
        super().__init__(it_max, dt_simulation)
        # 存储 ods_learner 为子类属性
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
        # self.environment.append(
        #     EllipseWithAxes(
        #         center_position=np.array([-35, 50]),
        #         orientation=0,
        #         axes_length=np.array([6, 12]),
        #         margin_absolut=self.robot.control_radius,
        #         # relative_reference_point=np.array([-10, 0]),
        #         # distance_scaling=10,
        #         linear_velocity=np.array([0, -4]),
        #         tail_effect=False,
        #     )
        # )
        #
        self.environment.append(
            CuboidXd(
                center_position=np.array([-10, 20]),
                orientation=40 * np.pi / 180,#障碍物转向角度
                axes_length=np.array([6, 6]),
                margin_absolut=self.robot.control_radius,
                angular_velocity=10 * np.pi / 180,
            )
        )


        '''static obstacle'''
        # self.environment.append(
        #     EllipseWithAxes(
        #         center_position=np.array([-33,-2]),
        #         # orientation=0,
        #         axes_length=np.array([6, 9]),
        #         margin_absolut=self.robot.control_radius,
        #         # relative_reference_point=np.array([-10, 0]),
        #         # distance_scaling=10,
        #         # linear_velocity=np.array([-4, 0]),
        #         tail_effect=False,
        #     )
        # )
        #
        # self.environment.append(
        #     EllipseWithAxes(
        #         center_position=np.array([-20,-12]),
        #         # orientation=0,
        #         axes_length=np.array([10, 6]),
        #         margin_absolut=self.robot.control_radius,
        #         # relative_reference_point=np.array([-10, 0]),
        #         # distance_scaling=10,
        #         # linear_velocity=np.array([-4, 0]),
        #         tail_effect=False,
        #     )
        # )
        #
        #
        # self.environment.append(
        #     CuboidXd(
        #         center_position=np.array([-18, 33]),
        #         orientation=90 * np.pi / 180,#障碍物转向角度
        #         axes_length=np.array([5, 10]),
        #         margin_absolut=self.robot.control_radius,
        #         # angular_velocity=10 * np.pi / 180,
        #     )
        # )

        '''random obstacle'''
        np.random.seed(45)#spoon 100 WShape 100
        limits = [-40, 20, -10, 50]
        for i in range(1):
            xmin, xmax, ymin, ymax = limits
            center_x = np.random.uniform(limits[0], limits[1])
            center_y = np.random.uniform(limits[2], limits[3])
            center_position = np.array([center_x, center_y])
            # 随机生成障碍物的轴长
            axes_length = np.random.uniform(low=3, high=5, size=2)  # 可以根据需要调整轴的最小和最大值
            # 根据位置分配速度方向
            if center_x < (xmin + xmax) / 2 and center_y < (ymin + ymax) / 2:
                # 左上象限，速度指向右下（正x，正y），确保速度的最小值为3
                linear_velocity = np.random.uniform(low=3, high=5, size=2)
            elif center_x > (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:
                # 右下象限，速度指向左上（负x，负y），确保速度的最大值为-3
                linear_velocity = np.random.uniform(low=-5, high=-3, size=2)
            elif center_x < (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:
                # 左下象限，速度指向右上（正x，负y），x方向最小值为3，y方向最大值为-3
                linear_velocity = np.array([np.random.uniform(3, 5), np.random.uniform(-5, -3)])
            else:
                # 右上象限，速度指向左下（负x，正y），x方向最大值为-3，y方向最小值为3
                linear_velocity = np.array([np.random.uniform(-5, -3), np.random.uniform(3, 5)])

            # 创建椭圆障碍物并添加到环境
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
            # 随机生成中心位置
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])

            if edge == 'top':
                # 上边界：x在[xmin, xmax]之间，y固定为ymax
                center_x = np.random.uniform(xmin, xmax)
                center_y = ymax
            elif edge == 'bottom':
                # 下边界：x在[xmin, xmax]之间，y固定为ymin
                center_x = np.random.uniform(xmin, xmax)
                center_y = ymin
            elif edge == 'left':
                # 左边界：y在[ymin, ymax]之间，x固定为xmin
                center_x = xmin
                center_y = np.random.uniform(ymin, ymax)
            else:  # right
                # 右边界：y在[ymin, ymax]之间，x固定为xmax
                center_x = xmax
                center_y = np.random.uniform(ymin, ymax)

            center_position = np.array([center_x, center_y])

            # 随机生成轴长
            axes_length = np.random.uniform(low=5, high=8, size=2)

            # 根据位置设置初始速度方向
            if center_x < (xmin + xmax) / 2 and center_y < (ymin + ymax) / 2:
                # 左上象限，速度指向右下
                linear_velocity = np.random.uniform(low=0, high=5, size=2)
            elif center_x > (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:
                # 右下象限，速度指向左上
                linear_velocity = np.random.uniform(low=-5, high=0, size=2)
            elif center_x < (xmin + xmax) / 2 and center_y > (ymin + ymax) / 2:
                # 左下象限，速度指向右上
                linear_velocity = np.array([np.random.uniform(0, 5), np.random.uniform(-5, 0)])
            else:
                # 右上象限，速度指向左下
                linear_velocity = np.array([np.random.uniform(-5, 0), np.random.uniform(0, 5)])

            # 创建障碍物
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
            position = np.atleast_2d(position)  # 确保位置是二维数组

        # 确保位置数组形状正确，适应 ods_learner.predict 方法的输入要求
        if position.shape[1] != 2:  # 假设每个位置应包含两个坐标
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
            position = np.atleast_2d(position)  # 确保位置是二维数组
        z = position.T
        zd = []
        tol_z = 0.5
        C = 5 * np.eye(2)  # 系统矩阵

        # 使用k-d树找出与z[:, -1]最近的点
        distance, nearest_index = self.data_kd_tree.query(z[:, -1])
        nearest_point = data[:, nearest_index]

        if np.all(np.abs(z[:, -1] - data[:, nearest_index]) < tol_z):
            current_z = z[:, -1].reshape(-1, 1)  # 确保是二维数组
            v_gpr = get_gpr_velocity(current_z, ods_learner).reshape(2, 1)
            new_zd = v_gpr
        else:
            current_z = z[:, -1].reshape(-1, 1)  # 确保是二维数组
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
        # self.initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)#初始的速度生成器，在这里可以改初始的动力学模型
        self.initial_velocity = self.get_gpr_velocity(self.robot.pose.position,self.ods_learner)/3
        # self.initial_velocity = self.get_sds_velocity(self.robot.pose.position,self.neum_learner,self.initial_velocity)
        # self.modulated_velocity = self.avoider.avoid(self.initial_velocity)
        #self.initial_velocity = lpv_ds(self.robot.pose.position, self.ds_gmm, self.A_g, self.b_g).flatten()/10
        #self.initial_velocity = dynamic_attractor_lpv_ds_velocity(self.robot.pose.position, self.ds_gmm, self.A_g, self.b_g,self.data,self.data_kd_tree).flatten()/10
        # self.initial_velocity=self.get_gpr_subattractor_velocity(self.robot.pose.position,self.ods_learner)/5
        # self.initial_velocity = self.get_sds_velocity(self.robot.pose.position, self.neum_learner,
        #                                               self.initial_velocity)
        self.modulated_velocity = self.dynamics_avoider.avoid(velocity=self.initial_velocity,position=self.robot.pose.position)

        # print(self.initial_velocity,"#",self.modulated_velocity)#这里是初始速度和修改后速度的地方
        end_time = time.time()
        print("Time taken for initial velocity calculation: ", end_time - start_time)

        # # Update the position of each obstacle
        # for obstacle in self.environment:
        #     obstacle.update_position(dt=self.dt_simulation)

        self.robot.pose.position = (
            self.robot.pose.position + self.modulated_velocity * self.dt_simulation
        )

        self.position_list[:, ii + 1] = (
            self.modulated_velocity * self.dt_simulation + self.position_list[:, ii]
        )





        self.plot_environment(ii=ii)
        if ii==400:
            # # self.neum_learner.show_learning_result(self.neum_learner.neum_parameters, num_levels=10)
            # x_vals = np.linspace(options['limits'][0], options['limits'][1], 100)
            # y_vals = np.linspace(options['limits'][2], options['limits'][3], 100)
            # X, Y = np.meshgrid(x_vals, y_vals)
            # U, V = np.zeros_like(X), np.zeros_like(Y)
            #
            #
            #
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         point = np.array([[X[i, j]], [Y[i, j]]]).reshape(2)
            #         # 检查点是否在任何障碍物内部
            #         inside_obstacle = False
            #         for obstacle in self.environment:
            #             if obstacle.is_inside(point):
            #                 inside_obstacle = True
            #                 break
            #
            #         if not inside_obstacle:
            #             velocity= self.get_gpr_subattractor_velocity(point, self.ods_learner) /10
            #             # velocity = self.get_gpr_velocity(point, self.ods_learner) / 10
            #             velocity = self.get_sds_velocity(point, self.neum_learner, velocity)
            #             velocity = self.dynamics_avoider.avoid(velocity=velocity, position=point).reshape(2)
            #
            #             U[i, j] = velocity[0]
            #             V[i, j] = velocity[1]
            #         else:
            #             U[i, j] = 0
            #             V[i, j] = 0
            # self.ax.streamplot(X, Y, U, V, density=1.0, linewidth=0.3, maxlength=1.0, minlength=0.1,arrowstyle='simple', arrowsize=0.5)
            # self.plot_compared_algorithms()
            # # self.img1 = self.neum_learner.show_learning_result(self.neum_learner.neum_parameters, num_levels=10)
            # plt.savefig("./figures/compared_algorithm_2_"+Type, dpi=300, bbox_inches='tight')
            #np.save(os.path.join('compared_algrithm/compared_data/'+Type, 'without_sub_attractor_avoidance_'+Type+'.npy'), self.position_list)
            aa=1

        if ii%40==0:
            self.generate_random_obstacles(n_obstacles=4, limits=[-40, 20, -10, 50])


    def plot_environment(self, ii):
        # self.ax.clear()

        self.ax.cla()

        # # 重新绘制line1
        # if self.img1:
        #     self.ax.imshow(self.img1.get_array(), extent=self.img1.get_extent(), origin='lower', cmap='viridis')

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
        # self.neum_learner.parameters=np.loadtxt('../NeumParameters/Neum_parameter_for_Zshape_beta1.0_dH20.txt')
        # self.neum_learner.show_learning_result(self.neum_learner.parameters, num_levels=10)

        self.ax.scatter(data[0, :], data[1, :], c='red', s=10, label='Simulation with Disturbance')
        # if ii==1:
        #     self.img1=self.neum_learner.show_learning_result(self.neum_learner.neum_parameters, num_levels=10)
        # self.plot_color_gradient_based_on_distance(data, -40, 20, -10, 50, canvas_size=(100, 100), color_map='viridis',
        #                              alpha=0.7)


    def plot_compared_algorithms(self):
        # 加载数据
        x_track_point_static = np.load('compared_algrithm/compared_data/'+Type+'/dmp_x_track_point_static.npy')
        x_track_point_dynamic = np.load('compared_algrithm/compared_data/'+Type+'/dmp_x_track_point_dynamic.npy')
        x_track_point_steering = np.load('compared_algrithm/compared_data/'+Type+'/dmp_x_track_point_steering.npy')
        x_track_static_volume = np.load('compared_algrithm/compared_data/'+Type+'/dmp_x_track_static_volume.npy')
        x_track_dynamic_volume = np.load('compared_algrithm/compared_data/'+Type+'/dmp_x_track_dynamic_volume.npy')
        GMR_viapoint = np.load('compared_algrithm/compared_data/'+Type+'/GMR_viapoint_'+Type+'.npy')
        clpv_ds_data= loadmat('compared_algrithm/compared_data/'+Type+'/z_data_'+Type+'.mat')

        # 绘制图形
        # self.ax.plot(x_track_point_static[:, 0], x_track_point_static[:, 1], ':r', label='point static',linewidth=1.5,color='black')
        self.ax.plot(x_track_point_dynamic[:, 0], x_track_point_dynamic[:, 1], linestyle=(0, (3, 2, 1, 2)), color='purple',linewidth=1.5,
                 label='point dynamic')
        self.ax.plot(x_track_point_steering[:, 0], x_track_point_steering[:, 1], linestyle=(0, (2, 2, 1, 2, 1, 2)),linewidth=1.5,
                 color='brown', label='point steering')
        self.ax.plot(x_track_static_volume[:, 0], x_track_static_volume[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2)),linewidth=1.5,
                 color='blue', label='volume static')
        self.ax.plot(x_track_dynamic_volume[:, 0], x_track_dynamic_volume[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2, 1, 2)),linewidth=1.5,
                 color='orange', label='volume dynamic')
        self.ax.plot(GMR_viapoint[:, 0], GMR_viapoint[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2, 1, 2, 1, 2)),linewidth=1.5,
                 color='pink', label='GMR via point')
        self.ax.plot(clpv_ds_data['z'][0, :], clpv_ds_data['z'][1, :], linestyle=(0, (3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2)),linewidth=1.5,
                 color='red', label='CLPV-DS')


    def plot_color_gradient_based_on_distance(self,track_data, xmin, xmax, ymin, ymax, canvas_size=(100, 100), color_map='viridis',
                                     alpha=0.7):
        """
        根据轨迹在画布上生成颜色渐变图，并增加透明度
        :param track_data: 2行的numpy数组，第一行是x坐标，第二行是y坐标
        :param xmin: x轴最小值
        :param xmax: x轴最大值
        :param ymin: y轴最小值
        :param ymax: y轴最大值
        :param canvas_size: 画布大小，表示需要生成颜色的网格尺寸，默认(100, 100)
        :param color_map: 用于着色的颜色映射，默认是'viridis'
        :param alpha: 颜色的透明度，默认0.7，范围为0到1
        :return: 颜色渐变的图像
        """
        # 定义画布的x和y坐标
        x_canvas = np.linspace(xmin, xmax, canvas_size[0])
        y_canvas = np.linspace(ymin, ymax, canvas_size[1])
        xv, yv = np.meshgrid(x_canvas, y_canvas)

        # 构建KDTree来加速最小距离计算
        track_points = np.vstack((track_data[0], track_data[1])).T
        kdtree = KDTree(track_points)

        # 计算每个网格点到轨迹的最小距离
        grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
        distances, _ = kdtree.query(grid_points)
        distances = distances.reshape(xv.shape)

        # 归一化距离值到0-1之间，便于颜色映射
        norm_distances = (distances - distances.min()) / (distances.max() - distances.min())

        # 使用颜色映射，生成RGBA颜色，并设置透明度
        cmap = plt.get_cmap(color_map)
        colors = cmap(1 - norm_distances)  # 反转颜色映射，使得距离近为颜色的开始值

        # 增加透明度 (alpha 通道)
        colors[..., 3] = alpha  # 设置整个图像的alpha透明度

        # 显示图像
        plt.imshow(colors, extent=(xmin, xmax, ymin, ymax), origin='lower')
        # plt.plot(track_data[0], track_data[1], 'k-', linewidth=2)  # 用黑色线绘制轨迹
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
    # set_dalpv_ds()
    # initial_state = [-49.12587412587415,0.6993006993007072] #Angle
    # initial_state = [-10.9,-2]  #bendedline
    # initial_state = [-49.12587412587415,0.6993006993007072]  #Spoon
    # initial_state = [-46.531791907514396,2.8901734104046284]  #WShape
    # initial_state=[-36.05138840146707,44.534068025341654] #Zshape
    initial_state=[-25.537820886783074,-2.113474832009638]#Leaf_2
    # initial_state = [36.114188553700046,23.777043395603755]#Snake

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
    #np.save('./2d_data/gpr_data_'+Type, data)
    # np.save('./2d_data/gpr_u_Angle', u)



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


