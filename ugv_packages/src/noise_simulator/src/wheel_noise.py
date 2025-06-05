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
        
        rospy.Subscriber("joint_states", JointState, self.encoder_callback)

        # wheels Parameters
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.0875)
        self.dist_wheels = rospy.get_param('~wheel_dist', 0.0875)
        self.wheel_max_accel = rospy.get_param('~wheel_max_accel', 1.)
        
        # sensor params
        self.stddev = rospy.get_param('~position_stddev', 0.1)
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
        
        self.store_prev_values(self.init_x, self.init_y, self.init_th, -1, -1, -1)
        self.new_message = []
        
        self.encoder_avail = False  
        self.tick_avail = False
        
        # Publishers
        self.noisy_imu_pub = rospy.Publisher('idmind_motors/odom', Odometry, queue_size=10)
        self.publisher = rospy.Publisher('/welptopic', String, queue_size=10)
            
        self.wheel_radius_bias()
        
    def store_prev_values(self, x, y, th, current_time, tick_left, tick_right):
        self.old_tick_left = tick_left
        self.old_tick_right = tick_right
        self.old_x = x
        self.old_y = y
        self.old_th = th
        self.old_time = current_time
            
    def wheel_radius_bias(self):
        self.wheel_radius_left = self.wheel_radius + self.radius_bias_l
        self.wheel_radius_right = self.wheel_radius + self.radius_bias_r
    
    def wheel_slipage(self, velocity, omega):
        slip_ratio_lin = max(self.slipage_lin, min(1, 1 - 0.02*self.slipage_lin*abs(velocity)))
        slip_ratio_ang = max(self.slipage_ang, min(1, 1 - 0.01*self.slipage_ang*abs(omega)))
        return slip_ratio_lin, slip_ratio_ang
    
    def noise(self):
        self.tick_right += np.random.normal(self.bias, self.stddev)
        self.tick_left += np.random.normal(self.bias, self.stddev)
    
    def angle_to_odom(self):
        self.tick_left = self.new_message[0].position[self.new_message[0].name.index('base_to_front_left')]
        self.tick_right = self.new_message[0].position[self.new_message[0].name.index('base_to_front_right')]
        self.current_time = self.new_message[0].header.stamp.to_sec()
        self.new_message.pop(0)
        self.noise()
        delta_wheel_ang_left = self.tick_left - self.old_tick_left
        delta_wheel_ang_right = self.tick_right - self.old_tick_right
        
        # if (self.current_time - self.old_time < 1/TICK_FREQ*1.2):
        #     dt=1/TICK_FREQ
        # else:
        #     print(self.current_time - self.old_time)
        #     print(self.current_time)
        #     # dt = self.current_time - self.old_time
        #     dt = np.int((self.current_time - self.old_time)*TICK_FREQ)/TICK_FREQ
        dt = np.max([1, np.int((self.current_time - self.old_time)*TICK_FREQ)])/TICK_FREQ
        # if(dt > 0.01):
        #     print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        #     print(self.current_time)
        #     print(dt)
            # dt=1/TICK_FREQ
        # dt = np.ceil(self.current_time - self.old_time)/TICK_FREQ
        # dt = self.current_time - self.old_time
        # dt = 1/TICK_FREQ
        # print("dt")
        # print(dt)
        # if dt <= 1e-9:
        #     # self.store_prev_values(self.old_x, self.old_y, self.old_th, self.current_time, self.tick_left, self.tick_right)
        #     return
        
        delta_wheel_ang_vel_left = delta_wheel_ang_left/dt
        delta_wheel_ang_vel_right = delta_wheel_ang_right/dt
        # delta_wheel_ang_vel_left = np.max([np.min([delta_wheel_ang_left/dt, self.wheel_max_accel/dt]), -self.wheel_max_accel/dt])
        # delta_wheel_ang_vel_right = np.max([np.min([delta_wheel_ang_right/dt, self.wheel_max_accel/dt]), -self.wheel_max_accel/dt])
        
        
        #wheel radius bias
        delta_vx_left = np.max([np.min([self.wheel_radius_left * delta_wheel_ang_vel_left, self.wheel_max_accel/dt]), -self.wheel_max_accel/dt])
        delta_vx_right = np.max([np.min([self.wheel_radius_right * delta_wheel_ang_vel_right, self.wheel_max_accel/dt]), -self.wheel_max_accel/dt])
        # delta_vx_left = self.wheel_radius_left * delta_wheel_ang_vel_left
        # delta_vx_right = self.wheel_radius_right * delta_wheel_ang_vel_right
        
        #track width bias
        vx = (delta_vx_left + delta_vx_right)/2
        vth = (delta_vx_right - delta_vx_left)/(self.dist_wheels + self.track_width_bias)
        
        slipage_lin, slipage_ang = self.wheel_slipage(vx, vth)
        
        vx *= slipage_lin
        vth *= slipage_ang
        
        dth = vth * dt 
        
        dx = (vx * np.cos(self.old_th + dth)) * dt
        dy = (vx * np.sin(self.old_th + dth)) * dt
        
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
        
        self.new_pos.twist.twist.linear.x = vx
        self.new_pos.twist.twist.linear.y = 0
        self.new_pos.twist.twist.angular.z = vth
        
        #covariances updated
        self.new_pos.pose.covariance = np.array([1, 0., 0., 0., 0., 0.,\
                                                 0., 1, 0., 0., 0., 0.,\
                                                 0., 0., 1, 0., 0., 0.,\
                                                 0., 0., 0., 1, 0., 0.,\
                                                 0., 0., 0., 0., 1, 0.,\
                                                 0., 0., 0., 0., 0., 1 ])*(self.wheel_radius*slipage_lin*self.stddev)**2
        self.new_pos.twist.covariance = np.array([1, 0., 0., 0., 0., 0.,\
                                                  0., 1, 0., 0., 0., 0.,\
                                                  0., 0., 1, 0., 0., 0.,\
                                                  0., 0., 0., 1, 0., 0.,\
                                                  0., 0., 0., 0., 1, 0.,\
                                                  0., 0., 0., 0., 0., 1])*(self.wheel_radius*TICK_FREQ*slipage_lin*self.stddev)**2

        self.noisy_imu_pub.publish(self.new_pos)
        self.store_prev_values(self.old_x + dx, self.old_y + dy, self.old_th + dth, self.current_time, self.tick_left, self.tick_right)
        
    def encoder_callback(self, msg):     
        # with open(self.pre_path + 'wheel_left.csv', 'a') as file:
        #     file.write("%f, %f\n" %(msg.header.stamp.to_sec(), msg.position[msg.name.index('base_to_front_left')]))
        # with open(self.pre_path + 'wheel_right.csv', 'a') as file:
        #     file.write("%f, %f\n" %(msg.header.stamp.to_sec(), msg.position[msg.name.index('base_to_front_right')]))

        if not self.encoder_avail:
            self.old_tick_left = msg.position[msg.name.index('base_to_front_left')]
            self.old_tick_right = msg.position[msg.name.index('base_to_front_right')]
            self.old_time = msg.header.stamp.to_sec()
            self.encoder_avail = True  
        else:
            if ((len(self.new_message) > 0 and msg.header.stamp.to_sec() > self.new_message[-1].header.stamp.to_sec()) or len(self.new_message) == 0):
                self.new_message.append(msg)
            else:
                print("new message is old, ugh")
            # self.publisher.publish('full pose publish') 
            # self.tick_left = msg.position[msg.name.index('base_to_front_left')]
            # self.tick_right = msg.position[msg.name.index('base_to_front_right')]
            # self.current_time = msg.header.stamp.to_sec()
            self.tick_avail = True  


if __name__ == '__main__':
    try:
        encoder_noise_adder = EncoderNoiseAdder()
        rate = rospy.Rate(TICK_FREQ)
        while not encoder_noise_adder.tick_avail:
            continue
        while not rospy.is_shutdown():
            if(len(encoder_noise_adder.new_message)==0):
                # wait for next message of joint states
                rate.sleep()
                continue
            encoder_noise_adder.angle_to_odom()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
