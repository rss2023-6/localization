#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion

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
                                          self.lidar_callback, # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.pose_initialization_callback, # TODO: Fill this in
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
        self.N = 200
        self.points = None

    def lidar_callback(self, msg):
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        time_increment = msg.time_increment
        ranges = msg.ranges
        range_max = msg.range_max
        N = len(ranges)
        down_sample_scale = N // self.N
        down_sampled_ranges = [-1 * self.N]

        avg = 0
        for i in range(N):
            avg += 1.0 * ranges[i] / down_sample_scale
            if(i % down_sample_scale == down_sample_scale - 1):
                down_sampled_ranges[i] = avg
                avg = 0
        if(down_sampled_ranges[-1] == -1):
            down_sampled_ranges[-1] = avg

        self.sensor_model(self.points, down_sampled_ranges)
    
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

        (roll, pitch, theta) = euler_from_quaternion(q_x, q_y, q_z, q_w)
        # linear = msg.twist.linear
        # angular = msg.twist.agular
        odom = [x, y, theta]

        self.motion_model(self.points, odom)
   
    def pose_initialization_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        q_x = msg.pose.orientation.x
        q_y = msg.pose.orientation.y
        q_z = msg.pose.orientation.z
        q_w = msg.pose.orientation.w
    
        (roll, pitch, theta) = euler_from_quaternion(q_x, q_y, q_z, q_w)
        self.points = [[x, y, theta] * self.N]

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
