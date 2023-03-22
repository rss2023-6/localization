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

    def transformation_matrix(self,th):
            return np.array([[np.cos(th), -np.sin(th), 0],
                             [np.sin(th), np.cos(th),  0]
                             [0,            0,         1]])

    def noise(self):
         if not self.deterministic: #false so add noise
            return np.random.randn(3,1)/100.0
         else:
            return np.array([0,0,0])
              

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
        shape = particles.shape()
        new_particles = np.zeros(shape)

        for i in range(shape[0]):
            p = particles[i]
            new_p = p + np.matmul(self.transformation_matrix(p[2]),odometry) + self.noise()
            new_particles[i] = new_p
        return new_particles

        ####################################
