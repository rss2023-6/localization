#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
import numpy as np
import random
from scipy import signal

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
from threading import Lock

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
        
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        self.num_particles = rospy.get_param("~num_particles", 10)
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle", 1)
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
        self.pub_particles = rospy.Publisher("/particles", PoseArray, queue_size = 1)
        # Initialize the models

        self.lock = Lock()

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
        rospy.loginfo("lidar callback")
        updated_prob = self.sensor_model.evaluate(self.particles, signal.decimate(np.array(msg.ranges), self.num_beams_per_particle))
        updated_prob = updated_prob/np.sum(updated_prob)
        
        rowindexarray = np.arange(self.num_particles)
        sampled_rows = np.random.choice(rowindexarray, size=self.num_particles, p=updated_prob)
        sampled_points = self.particles[sampled_rows]
        
        self.lock.acquire()
        self.particles = sampled_points
        self.avg_and_publish()
        self.lock.release()

    def odom_callback(self, msg):
        rospy.loginfo("odom callback")
        odometry = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])
        updated_particles = self.motion_model.evaluate(self.particles, odometry)
        
        self.lock.acquire()
        self.particles = updated_particles
        self.avg_and_publish()
        self.visualise_particles()
        self.lock.release()

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
            self.particles[i,0] = x + random.gauss(0, .1)
            self.particles[i,1] = y + random.gauss(0, .1)
            self.particles[i,2] = theta + random.gauss(0, .1)
        rospy.loginfo(self.particles)

    def avg_and_publish(self):
        x = 0
        y = 0
        theta_unit_x = 0
        theta_unit_y = 0

        x = np.average(self.particles[:,0])
        y = np.average(self.particles[:,1])
        theta_unit_x += np.average(np.cos(self.particles[:,2]))
        theta_unit_y += np.average(np.sin(self.particles[:,2]))
        theta = np.arctan2(theta_unit_y, theta_unit_x)

        rospy.loginfo("x: {}, y: {}, theta: {}".format(x, y, theta))
        qx, qy, qz, qw = quaternion_from_euler(0, 0, theta)
        rospy.loginfo("qx: {}, qy: {}, qz: {}, qw: {}".format(qx, qy, qz, qw))

        odom = Odometry()
        odom.header.stamp = rospy.Time().now()
        odom.header.frame_id = "/map"
        odom.child_frame_id = "/base_link_pf"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)

        broadcaster = tf2.TransformBroadcaster()

        t = TransformStamped()
        t.header.stamp = rospy.Time().now()
        t.header.frame_id = "world"
        t.child_frame_id = self.particle_filter_frame

        t.transform.translation.x = x
        t.transform.translation.y = y

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        broadcaster.sendTransform(t)

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
            p.position.x = self.particles[i,0]
            p.position.y = self.particles[i,1]
            quat_angle = quaternion_from_euler(0, 0, self.particles[i,2])
            p.orientation.x = quat_angle[0]
            p.orientation.y = quat_angle[1]
            p.orientation.z = quat_angle[2]
            p.orientation.w = quat_angle[3]
            poses[i] = p

         pa = PoseArray()
         pa.poses = poses 
         pa.header.stamp = rospy.Time().now()
         pa.header.frame_id = "map"
         self.pub_particles.publish(pa)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()