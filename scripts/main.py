#!/usr/bin/env python3

import Common_Params as CP
import numpy as np
import model_predictive_controller as mpc
import turtlebot_controller as tc
import rospy
import multiprocessing
import time
import matplotlib.pyplot as plt

def circle(radius, center=(0, 0), num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return x, y
    
def hexagon(radius, center=(0, 0)):
    angles = np.linspace(0, 2 * np.pi, 7)  # 7 points: 6 vertices + 1 to close the hexagon
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return x, y

def worker(namespace, controller, queue):
    rospy.init_node(f'{namespace}_controller', anonymous=True)
    controller.start_subscribers()
    time.sleep(1)
    controller.set_initial()
    controller.start_turtlebot()
    result = controller.stop_turtlebot()
    queue.put(result)
    time.sleep(1)
    rospy.loginfo(f'Process ended for {namespace}')
    rospy.loginfo('Shutting down the node for turtlebot : {namespace}')
    rospy.signal_shutdown('Condition met, shutting down')

def main():

    queue = multiprocessing.Queue()
    #store_goal_locations = np.array([[1.5, 0.5], [-1.5, 0.5]])  
    #store_goal_locations = np.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5], [1.0, -0.5]])  
    #store_goal_locations = np.array([[4.0, 2.0], [4.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    store_goal_locations = np.array([[4.0, 6.0], [4.0, 2.0], [4.0, 4.0], [4.0, 0.0], [0.0, 2.0], [0.0, 6.0], [0.0, 0.0], [0.0, 4.0]])
    turtlebot_controllers = [None for _ in range(CP.No_of_robots)]
    mpc_controllers = [None for _ in range(CP.No_of_robots)]
    for i in range(CP.No_of_robots):
        mpc_controllers[i] = mpc.Model_Predictive_Controller(
            prediction_horizon_steps=6, 
            control_horizon_steps=4, 
            terminal_weight=1e-6, 
            intermediate_weight=1e-6, 
            control_weight=1e-6, 
            x_goal=store_goal_locations[i]
        )
        turtlebot_controllers[i] = tc.TurtleBotController(f'tb3_{i}', mpc_controllers[i], store_goal_locations[i])
    
    processes = []
    for i, controller in enumerate(turtlebot_controllers):
        process = multiprocessing.Process(target=worker, args=(f'tb3_{i}', controller, queue))
        processes.append(process)
        process.start()
    
    try:
        # Wait for all processes to complete
        for process in processes:
            process.join()
        print('All processes ended')
    except KeyboardInterrupt:
        rospy.loginfo('Node interrupted by user')
    finally:        
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        print('All processes have been terminated')
    results = [queue.get() for _ in range(len(turtlebot_controllers))]
    results.sort(key = lambda x : x[0])
    max_len = len(max(results, key = lambda x : len(x[1]))[1])
    for i in range(len(results)):
    	while len(results[i][1]) < max_len:
    	    results[i][1].append(results[i][1][-1])
    	    results[i][2].append(results[i][2][-1])
    
    x1, y1 = circle(0.15, (-1.1, -1.1))
    x2, y2 = circle(0.15, (-1.1, 0.0))
    x3, y3 = circle(0.15, (-1.1, 1.1))
    x4, y4 = circle(0.15, (0.0, -1.1))
    x5, y5 = circle(0.15, (0.0, 0.0))
    x6, y6 = circle(0.15, (0.0, 1.1))
    x7, y7 = circle(0.15, (1.1, -1.1))
    x8, y8 = circle(0.15, (1.1, 0.0))
    x9, y9 = circle(0.15, (1.1, 1.1))
    
    xh, yh = hexagon(3.0, (0.0, 0.0))
    
    #plt.fill(x1, y1, color = 'grey', edgecolor = 'black')
    #plt.fill(x2, y2, color = 'grey', edgecolor = 'black')
    #plt.fill(x3, y3, color = 'grey', edgecolor = 'black')
    #plt.fill(x4, y4, color = 'grey', edgecolor = 'black')
    #plt.fill(x5, y5, color = 'grey', edgecolor = 'black')
    #plt.fill(x6, y6, color = 'grey', edgecolor = 'black')
    #plt.fill(x7, y7, color = 'grey', edgecolor = 'black')
    #plt.fill(x8, y8, color = 'grey', edgecolor = 'black')
    #plt.fill(x9, y9, color = 'grey', edgecolor = 'black')
    #plt.plot(xh, yh, color = 'black')
    start_locations = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]])
    colors = ['r', 'b', 'g', 'y', 'orange', 'pink', 'brown', 'grey']
    for i in range(len(results)):
    	plt.plot(results[i][1], results[i][2], label = results[i][0], color = colors[i])
    	plt.xlabel('X-Coordinate')
    	plt.ylabel('Y-Coordinate')
    	plt.title('Trajectories')
    for i in range(len(results)):
    	plt.plot([start_locations[i][0]], [start_locations[i][1]], color = colors[i])
    	plt.text(start_locations[i][0] - 0.25, start_locations[i][1] - 0.25, f's_{i}')
    	plt.plot([store_goal_locations[i][0]], [store_goal_locations[i][1]], color = colors[i])
    	plt.text(store_goal_locations[i][0] - 0.25, store_goal_locations[i][1] + 0.25, f'g_{i}')
    plt.legend()
    plt.show()
    
        
if __name__ == '__main__':
    print('Program is starting')
    main()
    print('Program ended')

