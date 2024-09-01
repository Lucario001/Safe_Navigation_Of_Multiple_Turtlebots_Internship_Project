#!/usr/bin/env python3

from casadi import *
import numpy as np
import Common_Params as CP

def transition(x_casadi, u_casadi, cur_casadi):
    xnew_casadi = MX.zeros(x_casadi.shape[0], 1)
    xnew_casadi[0, 0] = x_casadi[0, 0] + (u_casadi[0, 0] * np.cos(cur_casadi[2]) - CP.l * u_casadi[1, 0] * np.sin(cur_casadi[2])) * CP.T
    xnew_casadi[1, 0] = x_casadi[1, 0] + (u_casadi[0, 0] * np.sin(cur_casadi[2]) + CP.l * u_casadi[1, 0] * np.cos(cur_casadi[2])) * CP.T
        
    return xnew_casadi


class Model_Predictive_Controller:
    
    def __init__(self, prediction_horizon_steps, control_horizon_steps, terminal_weight, intermediate_weight, control_weight, x_goal):
    	
    	self.Np = prediction_horizon_steps
    	self.Nc = control_horizon_steps
    	self.terminal_weight = terminal_weight
    	self.intermediate_weight = intermediate_weight
    	self.control_weight = control_weight
    	self.x_goal = x_goal
    	self.flag1 = 0
    	self.flag2 = 0
    	self.flag3 = 0
    	
    def set_initial_dist(self, x_init):
    	self.initial_distance = np.sqrt((x_init[0] - self.x_goal[0]) ** 2 + (x_init[1] - self.x_goal[1]) ** 2)
    
    def solve(self, x_init, prev_control, store_means, store_max_distances): 
    	if np.sqrt((x_init[0] - self.x_goal[0]) ** 2 + (x_init[1] - self.x_goal[1]) ** 2) < 0.75:
    	    self.terminal_weight = 1e-1
    	    self.intermediate_weight = 1e-3
    	    self.control_weight = 1e-4
    	#if len(store_means) != 0:
    	#    combined = list(zip(store_means, store_max_distances))
    	#    combined_sorted = sorted(combined, key = lambda x : x[1])
    	#    store_means, store_max_distances = zip(*combined_sorted)
    	#    store_means, store_max_distances = list(store_means), list(store_max_distances)
    	max_obstacles = len(store_means)
    	                                                                     # x_init stores the (x, y, theta) values for the turtlebot
    	opti = Opti()
    	u_casadi = opti.variable(CP.u_size, self.Np)                         # To store the predicted control inputs
    	                                                                     # The above is a matrix where the control inputs at one time instance are represented in one column and there are Np columns
    	u0_casadi = opti.parameter(CP.u_size, 1)                             # To store the initial control input
    	max_control = [CP.tb_max_V, CP.tb_max_W]
    	
    	for i in range(CP.u_size):
    	    opti.set_value(u0_casadi[i, 0], max_control[i])
    	    
    	for i in range(self.Nc, self.Np):                                    # Setting the control inputs beyond the control horizon till prediction horizon as control input at (control horizon - 1) 
    	    u_casadi[:, i] = u_casadi[:, self.Nc - 1]                       
    	
    	x_casadi = opti.variable(CP.sv_size, self.Np)                        # To store the predicted future states
    	
    	x0_casadi = opti.parameter(CP.sv_size, 1)                            # Stores the initial state
    	
    	xgoal_casadi = opti.parameter(CP.sv_size, 1)                         # Stores the goal state, which will later be used in the optimization problem
    	
    	current_casadi = MX.zeros(len(x_init), self.Np)                      # For storing the (x, y, theta) values for the future states
    	
    	for i in range(len(x_init)):                                         # x_casadi stores the first predicted state value at the zeroth index  
    	    current_casadi[i, 0] = x_init[i]                                 # current_casadi stores the current value of the turtlebot at the zeroth index 
   
    	opti.set_value(x0_casadi[0, 0], x_init[0] + CP.l * np.cos(x_init[2]))
    	opti.set_value(x0_casadi[1, 0], x_init[1] + CP.l * np.sin(x_init[2]))
    	
    	opti.set_value(xgoal_casadi[0, 0], self.x_goal[0])
    	opti.set_value(xgoal_casadi[1, 0], self.x_goal[1])
    	
    	transformation_matrix = MX.ones(CP.u_size, CP.u_size)
    	transformation_matrix[0, 0] = np.cos(current_casadi[2, 0])
    	transformation_matrix[0, 1] = - CP.l * np.sin(current_casadi[2, 0])
    	transformation_matrix[1, 0] = np.sin(current_casadi[2, 0])
    	transformation_matrix[1, 1] = CP.l * np.cos(current_casadi[2, 0])
    	
    	x_casadi[:, 0] = transition(x0_casadi, u_casadi[:, 0], current_casadi[:, 0])
    	
    	for i in range(1, self.Np):
    	    current_casadi[0, i] = current_casadi[0, i - 1] + u_casadi[0, i - 1] * np.cos(current_casadi[2, i - 1]) * CP.T
    	    current_casadi[1, i] = current_casadi[1, i - 1] + u_casadi[0, i - 1] * np.sin(current_casadi[2, i - 1]) * CP.T
    	    current_casadi[2, i] = current_casadi[2, i - 1] + u_casadi[1, i - 1] * CP.T
    	    x_casadi[:, i] = transition(x_casadi[:, i - 1], u_casadi[:, i], current_casadi[:, i])
    	
    	cost = 0
    	for i in range(self.Np):
    	    if i < self.Np - 1:
    	    	cost += self.intermediate_weight * (x_casadi[:, i] - xgoal_casadi[:, 0]).T @ (x_casadi[:, i] - xgoal_casadi[:, 0])
    	    else:
    	    	cost += self.terminal_weight * (x_casadi[:, i] - xgoal_casadi[:, 0]).T @ (x_casadi[:, i] - xgoal_casadi[:, 0])
    	    cost += self.control_weight * (u_casadi[:, i].T @ u_casadi[:, i])
    	
    	opti.minimize(cost)
    	
    	for i in range(self.Np):
    	    opti.set_initial(u_casadi[:, i], prev_control)
    	
    	obstacle_matrix = MX.zeros(min(len(store_means), max_obstacles), 2)
    	cbf_vector = MX.zeros(min(len(store_means), max_obstacles), 1)
    	flag = 0
    	for i in range(min(len(store_means), max_obstacles)):
    	    flag = 1
    	    obstacle_matrix[i, 0] = 2 * (x0_casadi[0, 0] - store_means[i][0])
    	    obstacle_matrix[i, 1] = 2 * (x0_casadi[1, 0] - store_means[i][1])
    	    obs_dist = np.sqrt((x_init[0] + CP.l * np.cos(x_init[2]) - store_means[i][0]) ** 2 + (x_init[1] + CP.l * np.sin(x_init[2]) - store_means[i][1]) ** 2)
    	    #cbf_vector[i, 0] = - CP.alpha * ((x0_casadi[0, 0] - store_means[i][0]) ** 2 + (x0_casadi[1, 0] - store_means[i][1]) ** 2 - (store_max_distances[i]) ** 2)
    	    cbf_vector[i, 0] = - CP.alpha * ((x0_casadi[0, 0] - store_means[i][0]) ** 2 + (x0_casadi[1, 0] - store_means[i][1]) ** 2 - (CP.d_tolerance + CP.d_circle / 2 + store_max_distances[i]) ** 2)
    	
    	
    	# Setting limits on control inputs 
    	 
    	u_limits = MX.zeros(CP.u_size, 1)                                                          # The upper limit on the control input        
    	l_limits = MX.zeros(CP.u_size, 1)                                                          # The lower limit on the control input
    	u_limits[0, 0] = CP.tb_max_V
    	l_limits[0, 0] = - CP.tb_max_V
    	u_limits[1, 0] = CP.tb_max_W
    	l_limits[1, 0] = - CP.tb_max_W
    	
    	"""
    	numerical_obstacle_matrix = np.zeros((len(store_means), 2))
    	numerical_cbf_vector = np.zeros((len(store_means), 1))
    	flag = 0
    	for i in range(len(store_means)):
    	    flag = 1
    	    numerical_obstacle_matrix[i, 0] = 2 * (x_init[0] + CP.l * np.cos(x_init[2]) - store_means[i][0])
    	    numerical_obstacle_matrix[i, 1] = 2 * (x_init[1] + CP.l * np.sin(x_init[2]) - store_means[i][1])
    	    obs_dist = np.sqrt((x_init[0] + CP.l * np.cos(x_init[2]) - store_means[i][0]) ** 2 + (x_init[1] + CP.l * np.sin(x_init[2]) - store_means[i][1]) ** 2)
    	    #if CP.d_circle + store_max_distances[i] > obs_dist:
    	        #cbf_vector[i, 0] = - CP.alpha * ((x0_casadi[0, 0] - store_means[i][0]) ** 2 + (x0_casadi[1, 0] - store_means[i][1]) ** 2 - (obs_dist - CP.d_circle) ** 2)
    	    #else:
    	    numerical_cbf_vector[i, 0] = - CP.alpha * ((x_init[0] + CP.l * np.cos(x_init[2]) - store_means[i][0]) ** 2 + (x_init[1] + CP.l * np.sin(x_init[2]) - store_means[i][1]) ** 2 - (CP.d_circle + store_max_distances[i]) ** 2)
    	
    	print(numerical_obstacle_matrix)
    	print(numerical_cbf_vector)"""
    	
    	
    	
    	
    	for i in range(self.Np):
    	    if flag:
    	        opti.subject_to(obstacle_matrix @ transformation_matrix @ u_casadi[:, i] >= cbf_vector)
    	    opti.subject_to(u_casadi[:, i] <= u_limits[:, 0])
    	    opti.subject_to(u_casadi[:, i] >= l_limits[:, 0])
    	    if i == self.Np - 1:
    	        break
    	    transformation_matrix[0, 0] = np.cos(current_casadi[2, i + 1])
    	    transformation_matrix[0, 1] = - CP.l * np.sin(current_casadi[2, i + 1])
    	    transformation_matrix[1, 0] = np.sin(current_casadi[2, i + 1])
    	    transformation_matrix[1, 1] = CP.l * np.cos(current_casadi[2, i + 1])
    	    
    	    obstacle_matrix = MX.zeros(min(len(store_means), max_obstacles), 2)
    	    cbf_vector = MX.zeros(min(len(store_means), max_obstacles), 1)
    	    
    	    flag = 0
    	    for j in range(min(len(store_means), max_obstacles)):
    	        flag = 1
    	        obstacle_matrix[j, 0] = 2 * (x_casadi[0, i] - store_means[j][0])
    	        obstacle_matrix[j, 1] = 2 * (x_casadi[1, i] - store_means[j][1])
    	        #cbf_vector[j, 0] = - CP.alpha * ((x_casadi[0, i] - store_means[j][0]) ** 2 + (x_casadi[1, i] - store_means[j][1]) ** 2 - (store_max_distances[j]) ** 2)
    	        cbf_vector[j, 0] = - CP.alpha * ((x_casadi[0, i] - store_means[j][0]) ** 2 + (x_casadi[1, i] - store_means[j][1]) ** 2 - (CP.d_tolerance + CP.d_circle / 2 + store_max_distances[j]) ** 2)
    	
    	solver_opts = {'ipopt' : {'print_level' : 0, 'linear_solver' : 'mumps'}}
    	opti.solver('ipopt', solver_opts)
    	try:
    	    sol = opti.solve()
    	    return [sol.value(u_casadi)[0, 0], sol.value(u_casadi)[1, 0]]
    	except:
    	    return [0.0, 0.0]

    	
    	
    	 
    	 
    	
