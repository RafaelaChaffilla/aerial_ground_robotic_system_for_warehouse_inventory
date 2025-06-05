#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class ImuNoiseAdder:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('imu_noise_adder')
        rospy.loginfo("imu started!!!!!!")
        # Parameters
        self.angular_velocity_stddev = rospy.get_param('~angular_velocity_stddev', 0.3)
        self.linear_acceleration_stddev = rospy.get_param('~linear_acceleration_stddev', 0.1)
        self.orientation_stddev = rospy.get_param('~orientation_stddev', 0.1)
        self.bias = rospy.get_param('~gaussian_noise', 0.0)
        self.max_ang = rospy.get_param('~max_meas_ang', 4.36)
        self.max_lin = rospy.get_param('~max_meas_lin', 10)
        self.min_ang = rospy.get_param('~min_meas_ang', -4.36)
        self.min_lin = rospy.get_param('~min_meas_lin', 10)
        
        imu_no_noise_topic = rospy.get_param('~no_noise_imu_topic', 'idmind_imu/imu/no_noise')
        
        # Subscribers
        self.imu_sub = rospy.Subscriber(imu_no_noise_topic, Imu, self.imu_callback)
        
        # Publishers
        self.noisy_imu_pub = rospy.Publisher('idmind_imu/imu', Imu, queue_size=10)

    def imu_callback(self, msg):
        noisy_msg = Imu()
        noisy_msg.header = msg.header
        # print(msg.header)
        
        # Add noise to angular velocity
        noisy_msg.angular_velocity.x = min(max(msg.angular_velocity.x + np.random.normal(self.bias, self.angular_velocity_stddev), self.min_ang), self.max_ang)
        noisy_msg.angular_velocity.y = min(max(msg.angular_velocity.y + np.random.normal(self.bias, self.angular_velocity_stddev), self.min_ang), self.max_ang)
        noisy_msg.angular_velocity.z = min(max(msg.angular_velocity.z + np.random.normal(self.bias, self.angular_velocity_stddev), self.min_ang), self.max_ang)
        
        # Add noise to linear acceleration
        noisy_msg.linear_acceleration.x = min(max(msg.linear_acceleration.x + np.random.normal(self.bias, self.linear_acceleration_stddev), self.min_lin), self.max_lin)
        noisy_msg.linear_acceleration.y = min(max(msg.linear_acceleration.y + np.random.normal(self.bias, self.linear_acceleration_stddev), self.min_lin), self.max_lin)
        noisy_msg.linear_acceleration.z = min(max(msg.linear_acceleration.z + np.random.normal(self.bias, self.linear_acceleration_stddev), self.min_lin), self.max_lin)
        
        # Add noise to orientation (quaternion)
        (roll, pitch, yaw) = euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        (noisy_roll, noisy_pitch, noisy_yaw) = (roll + np.random.normal(self.bias, self.orientation_stddev), pitch + np.random.normal(self.bias, self.orientation_stddev), yaw + np.random.normal(self.bias, self.orientation_stddev))
        orientation_array = quaternion_from_euler(noisy_roll, noisy_pitch, noisy_yaw)
        noisy_msg.orientation = Quaternion(orientation_array[0], orientation_array[1], orientation_array[2], orientation_array[3])
        
        orientation_cov = (self.orientation_stddev + 4e-3)**2 if self.orientation_stddev != 0 else 1.5e-5
        linear_acceleration_cov = (self.linear_acceleration_stddev+0.01)**2 if self.linear_acceleration_stddev != 0 else 1e-4
        angular_velocity_cov = (self.angular_velocity_stddev+0.05)**2 if self.angular_velocity_stddev != 0 else 0.03
        noisy_msg.orientation_covariance = np.array([orientation_cov, 0., 0., 0. , orientation_cov, 0., 0., 0., orientation_cov])
        noisy_msg.linear_acceleration_covariance = np.array([linear_acceleration_cov, 0., 0., 0. , linear_acceleration_cov, 0., 0., 0., linear_acceleration_cov])
        noisy_msg.angular_velocity_covariance = np.array([angular_velocity_cov, 0., 0., 0. , angular_velocity_cov, 0., 0., 0., angular_velocity_cov])
        
        # Publish the noisy IMU data
        self.noisy_imu_pub.publish(noisy_msg)

if __name__ == '__main__':
    try:
        imu_noise_adder = ImuNoiseAdder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
