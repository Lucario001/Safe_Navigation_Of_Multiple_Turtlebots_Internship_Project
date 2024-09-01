#!/usr/bin/env python3

tb_wheel_radius = 0.033                                      # [m]
tb_track_width = 0.16                                        # [m]

tb_max_V = 0.22                                              # [m/s]
tb_max_W = 2.84                                              # [rad/s]

goal_threshold = 0.1                                         # [m] Minimum distance from goal after which it can stop

No_of_robots = 8           

tb_length = 0.138                                            # [m] Length of the turtlebot=BURGER  
tb_width = 0.178                                             # [m] Width of the turtlebot=BURGER  
tb_height = 0.192                                            # [m] Height of the turtlebot=BURGER

l = tb_length / 4                                            # [m] Look ahead distance

u_size = 2                                                   # Control input size
sv_size = 2                                                  # State Variable size

T = 0.1                                                      # Sample time

alpha = 0.1                                                  # For Class K function

lidar_range = 3.5                                            # [m] The maximum lidar range
min_lidar_range = 0.12                                       # [m] The minimum lidar range
d_circle = 0.21                                              # [m] Diameter of smallest circle that covers the entire turtlebot

d_tolerance = 0.05                                           # [m] Based on lidar diameter and turtlebot diameter 
