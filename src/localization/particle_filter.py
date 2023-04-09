#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
import random

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from  geometry_msgs.msg import TransformStamped
import tf2_ros as tf2
from scipy import signal

class ParticleFilter:

    def __init__(self):
        self.particles = np.array([None])
        self.old_location = np.array([None])

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

        self.particle_filter_frame = rospy.get_param("~particle_filter_frame", "/base_link_pf")
        

        self.num_particles = rospy.get_param("~num_particles", 200)
        
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
        self.pub_particles = rospy.Publisher("/particles", PoseArray, queue_size = 1)
        
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
        self.probs = None
        self.avg_location = np.zeros((1, 3))
        self.old_avg = np.zeros([1, 3])

    def transformation_matrix(self,th):
        return np.array([[np.cos(th), -np.sin(th), 0],
                            [np.sin(th), np.cos(th), 0],
                            [0, 0, 1]])
    
    def get_average(self, particles):
        avg_x = 0
        avg_y = 0
        N = len(particles)
        for i in range(len(particles)):
            p = particles[i]
            point = np.matmul(self.transformation_matrix(p[2]), np.array([[p[0], p[1]]]))
            avg_x += point[0][0] / N
            avg_y += point[1][0] / N
        return (avg_x, avg_y)


    def get_argmax(self):
        argmax = np.argmax(self.probs)
        return self.particles[argmax]

    def lidar_callback(self, msg):
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        time_increment = msg.time_increment
        ranges = msg.ranges
        range_max = msg.range_max

        #make sure down sampled particles are set
        ranges = signal.decimate(np.array(ranges), len(ranges)/self.num_beams_per_particle)
        
        #update probability
        if self.particles.all() != None:
            self.probs = self.sensor_model.evaluate(self.particles, ranges)
            norm_probs = self.probs/sum(self.probs)

            #resample particles
            #try random.choices if this throws errors
            self.particles = self.particles[np.random.choice(self.particles.shape[0], size=self.num_particles, p=norm_probs)]
            
            #self.avg_location = self.get_average(self.particles)
            
            self.avg_location = self.get_argmax()

            odom_msg = Odometry()
            odom_msg.pose.pose.position.x = self.avg_location[0]
            odom_msg.pose.pose.position.y = self.avg_location[1]

            #Might have to do the q_x, q_y thing directly 
            o = quaternion_from_euler(0, 0, self.avg_location[2])
            odom_msg.pose.pose.orientation.x = o[0]
            odom_msg.pose.pose.orientation.y = o[1]
            odom_msg.pose.pose.orientation.z = o[2]
            odom_msg.pose.pose.orientation.w = o[3]

            self.odom_pub.publish(odom_msg)
            
            broadcaster = tf2.TransformBroadcaster()

            t = TransformStamped()
            t.header.stamp = rospy.Time().now()
            t.header.frame_id = "world"
            t.child_frame_id = self.particle_filter_frame

            t.transform.translation.x = self.avg_location[0]
            t.transform.translation.y = self.avg_location[1]

            t.transform.rotation.x = o[0]
            t.transform.rotation.y = o[1]
            t.transform.rotation.z = o[2]
            t.transform.rotation.w = o[3]
            
            broadcaster.sendTransform(t)
    
    def odom_callback(self, msg):

        if self.particles.all() != None:
            odom = None
            pose = msg.pose.pose
            x = pose.position.x
            y = pose.position.y
                
            q_x = pose.orientation.x
            q_y = pose.orientation.y
            q_z = pose.orientation.z
            q_w = pose.orientation.w

            theta = euler_from_quaternion([q_x, q_y, q_z, q_w])[2]
            odom = self.get_odometry(np.array([x, y, theta]))
            
            self.old_avg = self.avg_location

            #make sure down sampled particles are set
            # self.setDownSampleparticles()

            #update particles
            self.particles = np.array(self.motion_model.evaluate(self.particles, odom))
            
            #publish averaged position
            #NOTE: If this ever throughs an array is numpy error, there might've been concurrecy issues between probability and odom models
            #IMPORTANT TODO: FIX the average position thing to work with angles, right now I just straight averaged, which doesn't work for sphereical coordinates, 
            # there's a seciton in the notebook about this

            #self.avg_location = self.get_average()

            self.visualise_particles()

    def get_odometry(self, k):
        '''
        copied and adapted from Nico's solution to individual lab part 5 a)
        takes in absolute odometry and returns the relative change compared to last known location
        '''
        if (self.old_location.all() == None):
            return [0, 0, 0]
        
        #Define k as the point in the space where the car is at time k
        #Define k_1 as the inverse of the point in the space where the car is at time k - 1 
        #We are trying to find deltaX which can be thought of as the pose of k in respect to the k_1 frame
        norm_k = [k[0], k[1], 1]
        k_1 = np.linalg.inv(self.old_location) 
 
        k_1ToWTransform = np.array([[np.cos(k_1[2]), -1.0 * np.sin(k_1[2]), k_1[0]],
                                    [np.sin(k_1[2]), np.cos(k_1[2]), k_1[1]],
                                    [0, 0, 1]])

        #Normalised Position of k in respect to k_1 can be found by multipliying the transformation matrix (k_1 to World) by the normalised position of k in the world frame   
        DeltaX = np.matmul(k_1ToWTransform, norm_k)

        #This finds x,y components of delta X but we are still missing theta
        #Theta of world to k equals theta of world to k-1 + theta of k-1 to k. 
        #Therefore, theta of k-1 to k is theta of world to k - theta of world to k-1
        DeltaX[2] = k[2] - self.old_location[2] 

        self.old_location = k

        return DeltaX
        

       
    def visualise_particles(self):
         '''
         Publishes a visualation of particles to view them moving in rviz. 
         Only useful for testing/validation. Should not be running in final solution

         input: 
            particles: N x 3 matrix of the form:
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

         '''
         poses = np.empty(self.num_particles, dtype=Pose) 
         for i in range(self.num_particles):
            #convert particles x,y and theta into poses for display
            p = Pose()
            p.position.x = self.particles[i][0]
            p.position.y = self.particles[i][1]
            quat_angle = quaternion_from_euler(0, 0, self.particles[i][1])
            p.orientation.x = quat_angle[0]
            p.orientation.y = quat_angle[1]
            p.orientation.z = quat_angle[2]
            p.orientation.w = quat_angle[3]
            poses[i] = p

         pa = PoseArray()
         pa.poses = poses 
         pa.header.frame_id = "map"

         self.pub_particles.publish(pa)

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
        self.old_location = np.array([None])

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w
    
        roh, pitch, theta = euler_from_quaternion([q_x, q_y, q_z, q_w])
        self.particles = np.repeat(np.array([[x, y, theta]]), self.num_particles, axis=0)
       

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
