from __future__ import division
import math
import numpy as np
from localization.scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs
import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import signal

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        self.z_max_px = 200

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_resolution = 1
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        table = np.empty((self.table_width, self.table_width))
        discretized_d = np.linspace(0, self.z_max_px, num=self.table_width)
        for i in range(self.table_width):
            for j in range(self.table_width):
                table[i][j] = self.p_hit(discretized_d[j], discretized_d[i]) # d = row, z_k = column
            table[i] = table[i]/sum(table[i]) #normalise z_k range over fixed d
            for j in range(self.table_width):
                table[i][j] = self.p_z(discretized_d[j], discretized_d[i], table[i][j])
            table[i] = table[i]/sum(table[i]) 
        table = np.swapaxes(table, 0, 1)
        self.sensor_model_table = table #directly modify self.sensor_model_table

        #self.visualize_model()


    def visualize_model(self):
        x = np.linspace(0, self.table_width, self.table_width)
        y = np.linspace(0, self.table_width, self.table_width)
        X,Y = np.meshgrid(x,y)
        
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X,Y,self.sensor_model_table.T)
        ax.invert_xaxis()

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
        

    
    def p_hit(self, z_k, d):
        mult = 1/(np.sqrt(2.0 * np.pi * (self.sigma_hit**2)))
        exp = np.exp((-1.0 * (z_k - d)**2)/(2.0 * (self.sigma_hit**2)))
        return mult*exp if 0 <= z_k <= self.z_max_px else 0

    def p_short(self, z_k, d):
        return (2.0/d) * (1 - (z_k/d)) if 0 <= z_k <= d and d != 0 else 0

    def p_max(self, z_k):
        return 1 if self.z_max_px == z_k else 0

    def p_rand(self, z_k):
        return 1/self.z_max_px if 0 <= z_k <= self.z_max_px else 0

    def p_z(self, z_k, d, p_hit_z):
        return self.alpha_hit * p_hit_z + self.alpha_short * self.p_short(z_k, d) + self.alpha_max * self.p_max(z_k) + self.alpha_rand * self.p_rand(z_k)

    def px_2_m(self, px):
        return px*float(self.map_resolution)*self.lidar_scale_to_map_scale

    def m_2_px(self, m):
        return m/(float(self.map_resolution)*self.lidar_scale_to_map_scale)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles)

        #observation = signal.decimate(obvservation, 1)
        
        # First convert ray trace and lidar units to px 
        scans = self.m_2_px(scans)
        obs = self.m_2_px(observation)

        # Clip Values Between 0 and z_max and rounds them
        scans = np.clip(scans,0,self.z_max_px)
        obs = np.clip(obs,0,self.z_max_px)
        clean_scans = np.rint(scans).astype(np.uint16)
        clean_obs = np.rint(obs).astype(np.uint16)

        # Compute probabilities
        probs = np.ones(clean_scans.shape[0],np.float64)
        for i in range(clean_scans.shape[0]): #rows
            for j in range(clean_scans.shape[1]): #columns
                curdec = self.sensor_model_table[clean_obs[j]][clean_scans[i][j]]
                probs[i] *= curdec
                

        final_result = np.power(probs, 1.0/2.2)
        
        return final_result

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map_resolution = map_msg.info.resolution
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
