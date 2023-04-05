#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from scipy import signal

class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback, 
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback, 
                                          queue_size=1)
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle", 100)
        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.pose_initialization_callback, 
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.points = None
        self.down_sampled_points = None
        self.probs = None

    # def setDownSamplePoints(self, ranges):
    #     N = len(ranges)
    #     down_sample_scale = N // self.N
    #     down_sampled_ranges = [-1 * self.N]

    #     avg = 0
    #     for i in range(N):
    #         avg += 1.0 * ranges[i] / down_sample_scale
    #         if(i % down_sample_scale == down_sample_scale - 1):
    #             down_sampled_ranges[i] = avg
    #             avg = 0
    #     if(down_sampled_ranges[-1] == -1):
    #         down_sampled_ranges[-1] = avg

    #     self.down_sampled_points = down_sampled_ranges

    def lidar_callback(self, msg):
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        time_increment = msg.time_increment
        ranges = msg.ranges
        range_max = msg.range_max

        #make sure down sampled points are set
        self.down_sampled_points = signal.decimate(ranges, self.num_beams_per_particle)
        
        #update probability
        self.probs = self.sensor_model(self.points, self.down_sampled_ranges)

        #resample points
        self.down_sampled_points = np.random.choice(self.down_sampled_points, self.N, self.probs)
    
    def odom_callback(self, msg):
        header = msg.header
        child_frame_id = msg.child_frame_id

        pose = msg.pose.pose
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        
        q_x = pose.orientation.x
        q_y = pose.orientation.y
        q_z = pose.orientation.z
        q_w = pose.orientation.w

        roh, pitch, theta = euler_from_quaternion(q_x, q_y, q_z, q_w)
        # linear = msg.twist.linear
        # angular = msg.twist.agular
        odom = [x, y, theta]

        #make sure down sampled points are set
        # self.setDownSamplePoints()

        #update points
        self.down_sampled_points = np.array(self.motion_model(self.down_sampled_points, odom))
        
        #publish averaged position
        #NOTE: If this ever throughs an array is numpy error, there might've been concurrecy issues between probability and odom models
        #IMPORTANT TODO: FIX the average position thing to work with angles, right now I just straight averaged, which doesn't work for sphereical coordinates, 
        # there's a seciton in the notebook about this
        average = np.average(self.down_sampled_points, axis=0)
        self.odom_pub.pub(self.point_to_message(average[0], average[1], average[2]))

    def point_to_message(x, y, theta):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y

        o = quaternion_from_euler(0,0,theta)
        msg.pose.pose.orientation.x = o[0]
        msg.pose.pose.orientation.y = o[1]
        msg.pose.pose.orientation.z = o[2]
        msg.pose.pose.orientation.w = o[3]
        return msg

    def pose_initialization_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        q_x = msg.pose.orientation.x
        q_y = msg.pose.orientation.y
        q_z = msg.pose.orientation.z
        q_w = msg.pose.orientation.w
    
        roh, pitch, theta = euler_from_quaternion(q_x, q_y, q_z, q_w)
        self.points = [[x, y, theta] * self.N]

        #publishes initial pose to odometry data -> goes to odom model I think
        msg = PoseWithCovarianceStamped()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.x = q_x
        msg.pose.pose.orientation.y = q_y
        msg.pose.pose.orientation.z = q_z
        msg.pose.pose.orientation.w = q_w
        self.odom_pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
