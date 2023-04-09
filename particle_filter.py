#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
import numpy as np
import random


from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler


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
        self.num_particles = rospy.get_param("~num_particles", 200)
        self.particles = np.zeros((self.num_particles, 3))
        
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback, 
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback, 
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.pose_init_callback, 
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

        #initial particles
        
        #Whenever you get odometry data use the motion model to update the particle positions

        #Whenever you get a lidar scan use the sensor model to update the particle weights

        #Resample the particles

        #Publish the new particle positions' mean as a transformation frame




    def lidar_callback(self, msg):
        updated_prob = self.sensor_model.evaluate(self.particles, np.array(msg.ranges))
        rospy.loginfo(updated_prob[0:9])
        rowindexarray = np.arange(self.num_particles)
        sampled_row = np.random.choice(rowindexarray, p=updated_prob)
        sampled_point = self.particles[sampled_row]
        
        x = sampled_point[0]
        y = sampled_point[1]
        theta = sampled_point[2]

        for i in range(self.num_particles):
            self.particles[i,0] = x + random.gauss(0, 0.1)
            self.particles[i,1] = y + random.gauss(0, 0.1)
            self.particles[i,2] = theta + random.gauss(0, 0.1)
        self.avg_and_publish()

    def odom_callback(self, msg):
        odometry = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])
        updated_particles = self.motion_model.evaluate(self.particles, odometry)
        self.particles = updated_particles
        self.avg_and_publish()

    def pose_init_callback(self, msg):
        #initialize particles
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w
    
        roh, pitch, theta = euler_from_quaternion([q_x, q_y, q_z, q_w])

        for i in range(self.num_particles):
            self.particles[i,0] = x + random.gauss(0, 0.1)
            self.particles[i,1] = y + random.gauss(0, 0.1)
            self.particles[i,2] = theta + random.gauss(0, 0.1)
            
    def avg_and_publish(self):
        x = 0
        y = 0
        theta_unit_x = 0
        theta_unit_y = 0
        for i in range(self.num_particles):
            x += self.particles[i,0]
            y += self.particles[i,1]
            theta_unit_x += np.cos(self.particles[i,2])
            theta_unit_y += np.sin(self.particles[i,2])
        x /= self.num_particles
        y /= self.num_particles
        theta_unit_x /= self.num_particles
        theta_unit_y /= self.num_particles
        theta = np.arctan2(theta_unit_y, theta_unit_x)

        qx, qy, qz, qw = quaternion_from_euler(0, 0, theta)

        odom = Odometry()
        odom.header.frame_id = "/map"
        #odom.child_frame_id = "particle_filter_frame"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
