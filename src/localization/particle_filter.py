#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
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
        self.location_viz_pub = rospy.Publisher("base_link_pf", PoseArray, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        self.N = 200

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
        self.points = None
        self.probs = None

    def transformation_matrix(self,th):
        return np.array([[np.cos(th), -np.sin(th), 0],
                            [np.sin(th), np.cos(th), 0],
                            [0, 0, 1]])
    
    def get_average():
        avg_x = 0
        avg_y = 0
        N = len(self.points)
        for i in range(len(self.points)):
            p = self.points[i]
            point = np.matmul(self.transformation_matrix(p[2]), np.array([[p[0], p[1]]]))
            avg_x += point[0][0] / N
            avg_y += point[1][0] / N
        return (avg_x, avg_y)

    def lidar_callback(self, msg):
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        time_increment = msg.time_increment
        ranges = msg.ranges
        range_max = msg.range_max

        #make sure down sampled points are set
        self.points = signal.decimate(ranges, self.num_beams_per_particle)
        
        #update probability
        self.probs = self.sensor_model.evaluate(self.points, ranges)

        #resample points
        #try random.choices if this throws errors
        self.points = np.random.choice(self.points, self.N, self.probs)

        (odom_x, odom_y, odom_theta) = self.get_average()
        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = odom_x
        odom_msg.pose.pose.position.y = odom_y

        odom_msg.pose.pose.orientation = quaternion_from_euler(0, 0, odom_theta)
        self.odom_pub(odom_msg)
    
    def odom_callback(self, msg):
        header = msg.header
        child_frame_id = msg.child_frame_id

        #ISSUE 1: odom only takes one point as publish but we want to work with multiple points
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
        self.points = np.array(self.motion_model(self.points, odom))
        
        #publish averaged position
        #NOTE: If this ever throughs an array is numpy error, there might've been concurrecy issues between probability and odom models
        #IMPORTANT TODO: FIX the average position thing to work with angles, right now I just straight averaged, which doesn't work for sphereical coordinates, 
        # there's a seciton in the notebook about this

        (avg_x, avg_y) = self.get_average()

        #publish average pose to odom viz for visualization
        pose = Pose()
        pose.position.x = avg_x
        pose.position.y = avg_y
        pose_msg = PoseArray()
        pose_msg.poses = [pose]
        self.location_viz_pub.pub(pose_msg)

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
