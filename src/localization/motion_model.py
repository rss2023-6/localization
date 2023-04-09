import rospy
import numpy as np

class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.deterministic = rospy.get_param("~deterministic")
        
        ####################################

    def noise(self):
         if not self.deterministic: #false so add noise
            return np.random.randn(3)/50.0
         else:
            return np.array([0,0,0])
         
    def get_new_pose(self, k, odom):
        '''
        Adapted from Nico's solution to individual part 1)
        '''
        #Let k be the particle's last known location
        #let k_1 be the new location of the particle with noise
        
        noisy_k = np.array([k[0], k[1], k[2]]) + self.noise()
        norm_odom = np.array([odom[0], odom[1], 1])
        WorldToKTransform = np.array([[np.cos(noisy_k[2]), -1.0 * np.sin(noisy_k[2]), noisy_k[0]],
                                      [np.sin(noisy_k[2]), np.cos(noisy_k[2]), noisy_k[1]],
                                      [0, 0, 1]])
        
        norm_k_1 = np.matmul(WorldToKTransform, norm_odom)
        norm_k_1[2] = noisy_k[2] + odom[2]
        return norm_k_1

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        
        ####################################

        # raise NotImplementedError
        shape = (len(particles), len(particles[0]))
        new_particles = np.zeros(shape)

        for i in range(shape[0]):
            p = particles[i] #list
            new_p = self.get_new_pose(p, odometry)
            new_particles[i] = new_p
        return new_particles.tolist()

        ####################################
