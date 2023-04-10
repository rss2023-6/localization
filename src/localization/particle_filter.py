#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
import random
import numpy as np
import random
# from scipy import signal

from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from  geometry_msgs.msg import TransformStamped
import tf2_ros as tf2
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from threading import Thread
from threading import RLock
from scipy import signal

from threading import RLock

class ParticleFilter:

    def __init__(self):
        self.particles = np.array([None])
        self.previous_odom_message = None
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
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame", "/base_link_pf")
        
        self.initialized = False

        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        self.num_particles = rospy.get_param("~num_particles", 200)
        
        self.map_topic = rospy.get_param("~map_topic", "/map")
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
                                          self.pose_init_callback, 
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.pub_particles = rospy.Publisher("/pf/viz/particles", PoseArray, queue_size = 1)
        # Initialize the models
        self.lock = RLock()
        self.previous_odom_message = None

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
        self.probs = None
        self.avg_location = np.zeros((1, 3))
        self.lock = RLock() 


    def get_argmax(self):
        argmax = np.argmax(self.probs)
        return self.particles[argmax]
    
    def update_particles(self, particles):
        with self.lock:
            self.particles = particles 
        
    def lidar_callback(self, msg):
        if not self.initialized:
            return
        
        #make sure down sampled particles are set
        ranges = signal.decimate(np.array(msg.ranges), len(msg.ranges)/self.num_beams_per_particle)
        
        #update probability
        if self.particles.all() != None:
            self.probs = self.sensor_model.evaluate(self.particles, ranges)
            norm_probs = self.probs/sum(self.probs)

            #resample particles
            #try random.choices if this throws errors
            self.update_particles(self.particles[np.random.choice(self.particles.shape[0], size=self.num_particles, p=norm_probs)])
            
            #self.avg_location = self.get_average(self.particles)
            self.avg_location = self.get_argmax()
            # self.abg_location = self.avg()

            odom_msg = Odometry()
            odom_msg.header.frame_id = self.map_topic
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

    def avg(self):
        x = 0
        y = 0
        theta_unit_x = 0
        theta_unit_y = 0

        x = np.average(self.particles[:,0])
        y = np.average(self.particles[:,1])
        theta_unit_x += np.average(np.cos(self.particles[:,2]))
        theta_unit_y += np.average(np.sin(self.particles[:,2]))
        theta = np.arctan2(theta_unit_y, theta_unit_x)

        # rospy.loginfo("x: {}, y: {}, theta: {}".format(x, y, theta))
        # qx, qy, qz, qw = quaternion_from_euler(0, 0, theta)
        # rospy.loginfo("qx: {}, qy: {}, qz: {}, qw: {}".format(qx, qy, qz, qw))

        return [x, y, theta]
        # odom = Odometry()
        # odom.header.stamp = rospy.Time().now()
        # odom.header.frame_id = self.map_topic
        # odom.child_frame_id = self.particle_filter_frame
        # odom.pose.pose.position.x = x
        # odom.pose.pose.position.y = y
        # odom.pose.pose.position.z = 0
        # odom.pose.pose.orientation.x = qx
        # odom.pose.pose.orientation.y = qy
        # odom.pose.pose.orientation.z = qz
        # odom.pose.pose.orientation.w = qw
        # self.odom_pub.publish(odom)

        # broadcaster = tf2.TransformBroadcaster()

        # t = TransformStamped()
        # t.header.stamp = rospy.Time().now()
        # t.header.frame_id = "world"
        # t.child_frame_id = self.particle_filter_frame

        # t.transform.translation.x = x
        # t.transform.translation.y = y

        # t.transform.rotation.x = qx
        # t.transform.rotation.y = qy
        # t.transform.rotation.z = qz
        # t.transform.rotation.w = qw
        
        # broadcaster.sendTransform(t)
    
    def odom_callback(self, msg):
        if not self.initialized:
            return
        if(self.previous_odom_message != None):
                dt = (msg.header.stamp - self.previous_odom_message.header.stamp).to_sec()
                odometry = np.array([msg.twist.twist.linear.x * dt, msg.twist.twist.linear.y * dt, msg.twist.twist.angular.z * dt])
                self.update_particles(np.array(self.motion_model.evaluate(self.particles, odometry)))
                self.visualise_particles()
        self.previous_odom_message = msg
        #make sure down sampled particles are set
        # self.setDownSampleparticles()

        #update particles
       
        
        #publish averaged position
        #NOTE: If this ever throughs an array is numpy error, there might've been concurrecy issues between probability and odom models
        #IMPORTANT TODO: FIX the average position thing to work with angles, right now I just straight averaged, which doesn't work for sphereical coordinates, 
        # there's a seciton in the notebook about this 

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
            quat_angle = quaternion_from_euler(0, 0, self.particles[i][2])
            p.orientation.x = quat_angle[0]
            p.orientation.y = quat_angle[1]
            p.orientation.z = quat_angle[2]
            p.orientation.w = quat_angle[3]
            poses[i] = p

         pa = PoseArray()
         pa.poses = poses 
         pa.header.frame_id = self.map_topic

         self.pub_particles.publish(pa)

    def pose_init_callback(self, msg):
        #initialize particles
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w
    
        roll, pitch, theta = euler_from_quaternion([q_x, q_y, q_z, q_w])

        particles = np.repeat(np.array([[x, y, theta]]))

        for i in range(self.num_particles):
            particles[i,0] = x + random.gauss(0, .1)
            particles[i,1] = y + random.gauss(0, .1)
            particles[i,2] = theta + random.gauss(0, .1)

        self.update_particles(particles)
        self.initialized = True
       
if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()