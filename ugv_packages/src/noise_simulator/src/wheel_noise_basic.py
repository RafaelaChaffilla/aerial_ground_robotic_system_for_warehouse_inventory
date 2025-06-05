#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler, euler_from_quaternion

TICK_FREQ = 100.

class EncoderNoiseAdder:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('encoder_noise_adder')
        
        rospy.Subscriber("/rovid/idmind_motors/odom/no_noise", Odometry, self.encoder_callback)

        # wheels Parameters
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.0875)
        self.dist_wheels = rospy.get_param('~wheel_dist', 0.0875)
        self.wheel_max_accel = rospy.get_param('~wheel_max_accel', 1.)
        
        # sensor params
        self.stddev = rospy.get_param('~speed_stddev', 0.1)
        self.bias = rospy.get_param('~gaussian_noise', 0.0)
        self.radius_bias_l = rospy.get_param('~radius_bias_left', 0.0)
        self.radius_bias_r = rospy.get_param('~radius_bias_right', 0.0)
        self.track_width_bias = rospy.get_param('~track_width_bias', 0.0)
        self.slipage_lin = rospy.get_param('~slipage_lin', 1.0)
        self.slipage_ang = rospy.get_param('~slipage_ang', 1.0)
        
        # position params
        self.init_x = rospy.get_param('~init_x', 0.0)
        self.init_y = rospy.get_param('~init_y', 0.0)
        self.init_th = rospy.get_param('~init_th', 0.0)
        
        self.store_prev_values(self.init_x, self.init_y, self.init_th, -1, 0, 0)
        self.new_message = []
        
        self.encoder_avail = False  
        self.tick_avail = False
        
        # Publishers
        self.noisy_imu_pub = rospy.Publisher('idmind_motors/odom', Odometry, queue_size=10)
        self.publisher = rospy.Publisher('/welptopic', String, queue_size=10)
            
        self.wheel_radius_bias()
        
    def store_prev_values(self, x, y, th, current_time, vx, vth):
        self.old_x = x
        self.old_y = y
        self.old_th = th
        self.old_time = current_time
        self.old_vx = vx
        self.old_vth = vth
            
    def wheel_radius_bias(self):
        self.wheel_radius_left = self.wheel_radius + self.radius_bias_l
        self.wheel_radius_right = self.wheel_radius + self.radius_bias_r
    
    def wheel_slipage(self, velocity, omega):
        slip_ratio_lin = max(self.slipage_lin, min(1, 1 - 0.02*self.slipage_lin*abs(velocity)))
        slip_ratio_ang = max(self.slipage_ang, min(1, 1 - 0.01*self.slipage_ang*abs(omega)))
        return slip_ratio_lin, slip_ratio_ang
    
    def noise(self, dt):
        self.vx += np.random.normal(self.bias, self.stddev)
        self.vth += np.random.normal(self.bias, self.stddev*((self.dist_wheels + self.track_width_bias)))
    
    def encoder_callback(self, msg):
        # print("NEW MESSAGE ODOM NO NOISEEEEEE??????")  
        self.current_time = msg.header.stamp.to_sec()
        self.vx = msg.twist.twist.linear.x
        self.vth = msg.twist.twist.angular.z

        dt = self.current_time - self.old_time
        
        # print(self.current_time)  
        # print(self.old_time)
        if (dt <= 1e-9 or self.old_time == -1):
            self.old_time = self.current_time
            # self.store_prev_values(self.old_x, self.old_y, self.old_th, self.current_time, self.tick_left, self.tick_right)
            return
                
        self.noise(dt)
        slipage_lin, slipage_ang = self.wheel_slipage(self.vx, self.vth)
        
        self.vx *= slipage_lin
        self.vth *= slipage_ang
        
        dth = (self.vth+self.old_vth)/2 * dt 
        
        dx = ((self.vx+self.old_vx)/2 * np.cos(self.old_th + dth)) * dt
        dy = ((self.vx+self.old_vx)/2 * np.sin(self.old_th + dth)) * dt
        
        self.new_pos = Odometry()
        self.new_pos.header.stamp = rospy.Time.from_sec(self.current_time)
        self.new_pos.header.frame_id = "odom"
        
        self.new_pos.pose.pose.position.x = self.old_x + dx
        self.new_pos.pose.pose.position.y = self.old_y + dy
        self.new_pos.pose.pose.position.z = 0
        
        a = quaternion_from_euler(0,0, self.old_th + dth)
        
        self.new_pos.pose.pose.orientation.x = a[0]
        self.new_pos.pose.pose.orientation.y = a[1]
        self.new_pos.pose.pose.orientation.z = a[2]
        self.new_pos.pose.pose.orientation.w = a[3]
        
        self.new_pos.twist.twist.linear.x = self.vx
        self.new_pos.twist.twist.linear.y = 0
        self.new_pos.twist.twist.angular.z = self.vth
        
        #covariances updated
        std_dev_send = (self.stddev+0.02)*self.slipage_lin if self.stddev != 0 else 0.02
        self.new_pos.pose.covariance = np.array([1, 0., 0., 0., 0., 0.,\
                                                 0., 1, 0., 0., 0., 0.,\
                                                 0., 0., 1, 0., 0., 0.,\
                                                 0., 0., 0., 1, 0., 0.,\
                                                 0., 0., 0., 0., 1, 0.,\
                                                 0., 0., 0., 0., 0., 1 ])*(std_dev_send)**2
        self.new_pos.twist.covariance = np.array([1, 0., 0., 0., 0., 0.,\
                                                  0., 1, 0., 0., 0., 0.,\
                                                  0., 0., 1, 0., 0., 0.,\
                                                  0., 0., 0., 1/(self.dist_wheels + self.track_width_bias), 0., 0.,\
                                                  0., 0., 0., 0., 1/(self.dist_wheels + self.track_width_bias), 0.,\
                                                  0., 0., 0., 0., 0., 1/(self.dist_wheels + self.track_width_bias)])*(std_dev_send)**2

        self.noisy_imu_pub.publish(self.new_pos)
        self.store_prev_values(self.old_x + dx, self.old_y + dy, self.old_th + dth, self.current_time, self.vx, self.vth)  
        


if __name__ == '__main__':
    try:
        encoder_noise_adder = EncoderNoiseAdder()
        rospy.spin()
        # rate = rospy.Rate(TICK_FREQ)
        # while not encoder_noise_adder.tick_avail:
        #     continue
        # while not rospy.is_shutdown():
        #     if(len(encoder_noise_adder.new_message)==0):
        #         # wait for next message of joint states
        #         rate.sleep()
        #         continue
        #     encoder_noise_adder.angle_to_odom()
        #     rate.sleep()
    except rospy.ROSInterruptException:
        pass
