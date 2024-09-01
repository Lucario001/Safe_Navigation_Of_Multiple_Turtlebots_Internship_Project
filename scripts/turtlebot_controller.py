#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf.transformations
import numpy as np
import Common_Params as CP
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import time

class TurtleBotController:
    
    def __init__(self, namespace, mpc_controller, goal):
    	self.namespace = namespace
    	self.mpc = mpc_controller
    	self.goal = goal
    	self.current_position = [0.0, 0.0, 0.0]
    	self.LIDAR_Data = []
    	self.LIDAR_Angles = []
    	self.turtlebot_status = False
    	
    def start_subscribers(self):
        rospy.loginfo(f'Start : {self.namespace}')
        rospy.Subscriber(f'/{self.namespace}/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber(f'/{self.namespace}/odom', Odometry, self.odom_callback)
    
    def set_initial(self):
        self.mpc.set_initial_dist(self.current_position)
    	
    def get_sensor_observation(self):
    	
    	self.observed_points = []                                                       # For storing observed points     
    	self.store_clusters = []                                                        # For storing clusters
    	self.store_means = []                                                           # For storing means of the clusters
    	self.store_max_distances = []                                                   # For storing maximum distances of the mean from any element in the cluster
    	
    	for i in range(len(self.LIDAR_Data)):                                           # The angles are with respect to the direction faced by the turtlebots
    	    if self.LIDAR_Data[i] <= 0.5:
    	    	self.observed_points.append([self.current_position[0] + self.LIDAR_Data[i] * np.cos(self.current_position[2] + self.LIDAR_Angles[i]), self.current_position[1] + self.LIDAR_Data[i] * np.sin(self.current_position[2] + self.LIDAR_Angles[i])])
    	
    	self.observed_points = np.array(self.observed_points)
    	
    	if len(self.observed_points):
    	    db = DBSCAN(eps = 0.25, min_samples = 3).fit(self.observed_points)
    	    labels = db.labels_                                                             # Gets the cluster labels
    	
    	    for label_id in np.unique(labels):
    	        if label_id == -1:
    	            continue
    	        self.store_clusters.append(self.observed_points[labels == label_id])
    	        self.store_means.append(np.mean(self.observed_points[labels == label_id], axis = 0))
    	    for j in range(len(self.store_means)):
    	        mean = self.store_means[j]
    	        min_point = min(self.store_clusters[j], key = lambda x : distance.euclidean(x, [self.current_position[0], self.current_position[1]]))
    	        max_dist = distance.euclidean(min_point, mean)
    	        if distance.euclidean(mean, [self.current_position[0], self.current_position[1]]) <= distance.euclidean(min_point, [self.current_position[0], self.current_position[1]]):
    	            max_dist = 0.0
    	        self.store_max_distances.append(max_dist)

    def start_turtlebot(self):
        self.vel_pub = rospy.Publisher(f'/{self.namespace}/cmd_vel', Twist, queue_size = 10)
        self.rate = rospy.Rate(1 / 0.01)
        
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        
        self.get_sensor_observation()
        
        rospy.loginfo(f'Turtlebot {self.namespace} is starting and its start position is : {self.current_position}')
        self.store_x = []
        self.store_y = []
        
        while np.sqrt((self.current_position[0] + CP.l * np.cos(self.current_position[2]) - self.goal[0]) ** 2 + (self.current_position[1] + CP.l * np.sin(self.current_position[2]) - self.goal[1]) ** 2) > CP.goal_threshold:
            rospy.loginfo(f'Solving for Robot : {self.namespace}')
            self.store_x.append(self.current_position[0])
            self.store_y.append(self.current_position[1])
            V, W = self.mpc.solve(x_init = self.current_position, prev_control = [self.vel_msg.linear.x, self.vel_msg.angular.z], store_means = self.store_means, store_max_distances = self.store_max_distances)
            rospy.loginfo(f'{self.namespace} : V : {V} W : {W}')
            self.vel_msg.linear.x = V
            self.vel_msg.angular.z = W
            self.vel_pub.publish(self.vel_msg)
            start_time = rospy.Time.now()
            while (rospy.Time.now() - start_time).to_sec() < CP.T:
                if np.sqrt((self.current_position[0] + CP.l * np.cos(self.current_position[2]) - self.goal[0]) ** 2 + (self.current_position[1] + CP.l * np.sin(self.current_position[2]) - self.goal[1]) ** 2) <= CP.goal_threshold:
                    self.vel_msg.linear.x = 0
                    self.vel_msg.angular.z = 0
                    self.vel_pub.publish(self.vel_msg)
                    break
            
            #self.rate.sleep()
            
            self.get_sensor_observation()

    def stop_turtlebot(self):
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        self.vel_pub.publish(self.vel_msg)
        return [self.namespace, self.store_x, self.store_y]

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Extract x and y
        x = position.x
        y = position.y
        
        # Convert quaternion to Euler angles to get theta (yaw)
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        theta = euler[2]
        self.current_position = [x, y, theta]
    	
    def lidar_callback(self, data):
    	self.LIDAR_Data = data.ranges
    	self.LIDAR_Angles = [data.angle_min + i * data.angle_increment for i in range(len(data.ranges))]
    	
