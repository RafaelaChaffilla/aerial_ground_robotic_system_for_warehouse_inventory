#!/usr/bin/env python

import rospy
import numpy as np
import tf
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial.distance import mahalanobis

DO_IMU_CORRECTION = 1
DO_ODOM_CORRECTION = 1
DO_POSITION_CORRECTION = 1

class EKF_ugv():
    def __init__(self):
        # Initialize Node
        rospy.init_node('ekf_ugv')
        rospy.loginfo("ekf rafa node initialized.")
        
        # Get Ros Parameters
        imu_topic = rospy.get_param("~imu_topic", 'idmind_imu/imu')
        odom_topic = rospy.get_param("~odom_topic", 'idmind_motors/odom')
        position_topic = rospy.get_param("~position_topic", 'hectorrafa_odom')
        
        publish_topic = rospy.get_param("~publish_topic", '/rovid/ekf_ugv')
        
        self.Q = np.array(rospy.get_param("~process_noise_Q", [[0.05, 0, 0, 0, 0, 0, 0],  
                                                               [0, 0.05, 0, 0, 0, 0, 0], 
                                                               [0, 0, 0.05, 0, 0, 0, 0],
                                                               [0, 0, 0, 0.08, 0, 0, 0],
                                                               [0, 0, 0, 0, 0.08, 0, 0],
                                                               [0, 0, 0, 0, 0, 0.9, 0],
                                                               [0, 0, 0, 0, 0, 0, 0.9]]))
        
        self.P0 = np.array(rospy.get_param("~init_noise_P", [[1e-9, 0, 0, 0, 0, 0, 0], 
                                                            [0, 1e-9, 0, 0, 0, 0, 0], 
                                                            [0, 0, 1e-9, 0, 0, 0, 0],
                                                            [0, 0, 0, 1e-9, 0, 0, 0],
                                                            [0, 0, 0, 0, 1e-9, 0, 0],
                                                            [0, 0, 0, 0, 0, 1e-9, 0],
                                                            [0, 0, 0, 0, 0, 0, 1e-9]]))
        self.init_state = rospy.get_param("~init_state", [1., 1., 0., 0., 0., 0., 0.])
        
        
        self.rate  = rospy.get_param("~frequency", 50)
        self.sensor_timeout = rospy.get_param("~sensor_timeout", 2.)
        self.publish_to_tf  = rospy.get_param("~publish_tf", False)
        self.base_frame     = rospy.get_param("~base_frame", "rovid/base_link")
        self.odom_frame     = rospy.get_param("~odom_frame", "rovid/odom")
        
        
        # SUBSCRIBERS
        rospy.Subscriber(imu_topic, Imu, self.receive_imu)
        rospy.Subscriber(odom_topic, Odometry, self.receive_odom)
        rospy.Subscriber(position_topic, Odometry, self.receive_position)
        
        # PUBLISHERS
        self.pose_publisher = rospy.Publisher(publish_topic, Odometry, queue_size=10)
        if (self.publish_to_tf):
            self.tf_pub = tf.TransformBroadcaster()
        
        # Initialize Variables
        self.x_k_1 = np.array(self.init_state).reshape((7,1))
        self.P_k_1 = self.P0
        self.Ts    = 1./self.rate
        
        self.accel_limit = 4.0
        
        self.imu_queue  = []
        self.odom_queue = []
        self.pos_queue  = []
        
        self.time_k_1     = rospy.get_rostime()
        
        self.start = 0
        
        self.new_odom = False
        self.new_imu  = False
        self.new_pose = False

    #########################
    #   Callback Functions  #
    #########################
    def receive_imu(self, msg):
        z_imu = np.array([[msg.angular_velocity.z], [msg.linear_acceleration.x]])
        
        imu_noise = np.array([[0.03, 0.], [0., 0.0025]])
        
        imu_time = msg.header.stamp

        imu_pack = [z_imu, imu_noise, imu_time]
        
        if len(self.imu_queue) > 3:
            self.imu_queue = self.imu_queue[-3:]
            
        self.imu_queue.append(imu_pack)
        self.start += 1
        self.new_imu = True
        
    def receive_odom(self, msg):
        z_odom = np.array([[-msg.twist.twist.linear.x], [0.], [msg.twist.twist.angular.z]])
        
        odom_noise = np.array([[0.05**2, 0, 0], [0, 0.05**2, 0], [0, 0, 0.6**2]])
        
        odom_time = msg.header.stamp
        
        odom_pack = [z_odom, odom_noise, odom_time]
        
        if (len(self.odom_queue) > 0 and 
            (abs(self.odom_queue[-1][0][0] - z_odom[0])/abs((self.odom_queue[-1][2] - odom_time).to_sec()) > 2 or
            abs(self.odom_queue[-1][0][2] - z_odom[2])/abs((self.odom_queue[-1][2] - odom_time).to_sec()) > 500*np.pi/180)):
            return
        
        if len(self.odom_queue) > 3:
            self.odom_queue = self.odom_queue[-3:]
            
        self.odom_queue.append(odom_pack)
        
        if self.start == 0:
            self.init_time = odom_time
            
        self.start += 1
        self.new_odom = True
        
    def receive_position(self, msg):        
        (_, _, yaw) = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        z_pos = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [yaw]])
        pos_noise = np.array([[msg.pose.covariance[0]+0.1, 0, 0], [0, msg.pose.covariance[7]+0.1, 0], [0, 0, msg.pose.covariance[35]+0.01]])
        pos_time  = msg.header.stamp
        
        pos_pack = [z_pos, pos_noise, pos_time]
        if len(self.pos_queue) > 3:
            self.pos_queue = self.pos_queue[-3:]
            
        self.pos_queue.append(pos_pack)
        self.new_pose = True
        
    ########################
    #    Kalman Filter     #
    ########################
    def initialize(self):
        if len(self.imu_queue) > 0:
            self.x_k_1[2] = self.imu_queue[-1][0][0]
            self.P_k_1[2][2] = self.imu_queue[-1][1][0][0]
            x_k = self.x_k_1
            P_k = np.array(self.P_k_1)
        elif len(self.odom_queue) > 0:
            self.x_k_1[3] = self.odom_queue[-1][0][0]
            self.P_k_1[3][3] = self.odom_queue[-1][1][0][0]
            self.x_k_1[4] = self.odom_queue[-1][0][1]
            self.P_k_1[4][4] = self.odom_queue[-1][1][1][1]
            x_k = self.x_k_1
            P_k = np.array(self.P_k_1)
        self.publish_odom_message(rospy.get_rostime(), x_k, P_k)
        self.x_k_1 = x_k
        self.P_k_1 = P_k
        self.time_k_1 = rospy.get_rostime()

    
    def Kalman(self):
        time = rospy.get_rostime()
        x_k, P_k = self.prediction(time)
        self.check_timeout()
        if(self.new_imu):
            x_k, P_k = self.correction(x_k, P_k, time, sensor="imu")
        if(self.new_odom):
            x_k, P_k = self.correction(x_k, P_k, time, sensor="odom")
        if(self.new_pose):
            x_k, P_k = self.correction(x_k, P_k, time, sensor="pose")
        self.publish_odom_message(time, x_k, P_k)
        self.x_k_1 = x_k
        self.P_k_1 = P_k
        self.time_k_1 = time
    
    def prediction(self, time):
        delta_t = (time - self.time_k_1).to_sec()
        A = np.array([[1, 0, 0, np.cos(self.x_k_1[2, 0])*delta_t, -np.sin(self.x_k_1[2, 0])*delta_t, 0, 0.5*np.cos(self.x_k_1[2, 0])*delta_t*delta_t],
                      [0, 1, 0, np.sin(self.x_k_1[2, 0])*delta_t,  np.cos(self.x_k_1[2, 0])*delta_t, 0, 0.5*np.sin(self.x_k_1[2, 0])*delta_t*delta_t],
                      [0, 0, 1,                                0,                                 0, delta_t,                                      0],
                      [0, 0, 0,                                1,                                 0, 0,                                      delta_t],
                      [0, 0, 0,                                0,                                 1, 0,                                            0],
                      [0, 0, 0,                                0,                                 0, 1,                                            0],
                      [0, 0, 0,                                0,                                 0, 0,                                            1]])
        
        
        x_k = np.matmul(A, self.x_k_1)
        
        J = A
        J[0,2] = -np.sin(self.x_k_1[2, 0])*(self.x_k_1[3, 0]*delta_t+0.5*self.x_k_1[6, 0]*delta_t*delta_t) - np.cos(self.x_k_1[2, 0])*self.x_k_1[4, 0]*delta_t
        J[1,2] = np.cos(self.x_k_1[2, 0])*(self.x_k_1[3, 0]*delta_t+0.5*self.x_k_1[6, 0]*delta_t*delta_t) - np.sin(self.x_k_1[2, 0])*self.x_k_1[4, 0]*delta_t
        
        x_k[2] = (x_k[2, 0] + np.pi) % ( 2*np.pi) - np.pi
        
        P_k = (np.matmul(J, np.matmul(self.P_k_1, np.transpose(J))) + self.Q*delta_t)
        
        return x_k, P_k
    
    def correction(self, x_k, P_k, time, sensor = None):
                   
        H, z_k, R_k = self.get_H(x_k[2, 0], sensor)
        
        if H is None:
            return x_k, P_k
        
        if sensor == "pose":
            delta_t = (time - self.pos_queue[-1][2]).to_sec()
            A_back = np.array([[1, 0, 0, -np.cos(x_k[2, 0])*delta_t,  np.sin(x_k[2, 0])*delta_t, 0, -0.5*np.cos(x_k[2, 0])*delta_t*delta_t],
                               [0, 1, 0, -np.sin(x_k[2, 0])*delta_t, -np.cos(x_k[2, 0])*delta_t, 0, -0.5*np.sin(x_k[2, 0])*delta_t*delta_t],
                               [0, 0, 1,                                0,                                 0, -delta_t,                                      0],
                               [0, 0, 0,                                1,                                 0, 0,                                      -delta_t],
                               [0, 0, 0,                                0,                                 1, 0,                                            0],
                               [0, 0, 0,                                0,                                 0, 1,                                            0],
                               [0, 0, 0,                                0,                                 0, 0,                                            1]])
            H = np.matmul(H, A_back)
        
        S = np.matmul(H, np.matmul(P_k, np.transpose(H))) + R_k
        K = np.matmul(P_k, np.matmul(np.transpose(H), np.linalg.inv(S)))
        x_k_c = x_k + np.matmul(K, z_k-np.dot(H, x_k))
        
        aux_2 = np.eye(7) - np.matmul(K, H)
        P_k_c = np.matmul(aux_2, np.matmul(P_k, np.transpose(aux_2))) + np.matmul(K, np.matmul(R_k, np.transpose(K)))
        
        return x_k_c, P_k_c
   
    ###########################
    #   Auxiliary Functions   #
    ###########################
    def publish_odom_message(self, time, x_k, P_k):
        new_pose = Odometry()
        new_pose.header.stamp = time
        new_pose.header.frame_id = "rovid/odom"
        new_pose.pose.pose.position.x = x_k[0, 0]
        new_pose.pose.pose.position.y = x_k[1, 0]
        (new_pose.pose.pose.orientation.x, new_pose.pose.pose.orientation.y, new_pose.pose.pose.orientation.z, new_pose.pose.pose.orientation.w) = quaternion_from_euler(0, 0, x_k[2, 0])
        new_pose.pose.covariance = np.zeros((36,))
        new_pose.pose.covariance[0] = P_k[0,0] # x
        new_pose.pose.covariance[7] = P_k[1,1] # y
        new_pose.pose.covariance[35] = P_k[2,2] # theta
        
        new_pose.twist.twist.linear.x = x_k[3, 0]
        new_pose.twist.twist.angular.z = x_k[5, 0]
        new_pose.twist.covariance = np.zeros((36,))
        new_pose.twist.covariance[0] = P_k[3,3] # x dot
        new_pose.twist.covariance[35] = P_k[5,5] # theta dot
        
        self.pose_publisher.publish(new_pose)
        
        if (self.publish_to_tf):
            pos, ori = new_pose.pose.pose.position, new_pose.pose.pose.orientation
            self.tf_pub.sendTransform((pos.x, pos.y, pos.z),
                                    (ori.x, ori.y, ori.z, ori.w),
                                    new_pose.header.stamp,
                                    self.base_frame,
                                    self.odom_frame)
        
    def get_H(self, theta, sensor):
        if sensor == "imu":
            #IMU ONLY
            fixed_imu_pack = self.imu_queue[-1]
            self.new_imu = False
            H = np.array([[0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
            z_k = fixed_imu_pack[0]
            R_k = fixed_imu_pack[1]
            
            return H, z_k, R_k
        elif sensor == "odom":
            #ODOM ONLY
            fixed_odom_pack = self.odom_queue[-1]
            self.new_odom = False
            H = np.array([[0,0,0,1,0,0,0],
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0]])
            z_k = fixed_odom_pack[0]
            
            R_k = fixed_odom_pack[1]
            
            return H, z_k, R_k
        elif sensor == "pose":
            #POSITION ONLY
            fixed_pos_pack = self.pos_queue[-1]
            self.new_pose = False

            H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0]])
            z_k = fixed_pos_pack[0]
            if(np.abs(z_k[2] - theta) >= np.pi/2):
                angle_innovation = (z_k[2] - theta + np.pi) % (2 * np.pi) - np.pi
                z_k[2] = theta + angle_innovation
            
            R_k = fixed_pos_pack[1]
            
            return H, z_k, R_k
        else:
            return None, None, None
    
    def check_timeout(self):
        # check any dynamic sensor time out (if it was received at any time)
        if(len(self.imu_queue) > 0 and (rospy.get_rostime() - self.imu_queue[-1][2]).to_sec() > self.sensor_timeout):
            # imu timed out
            print("imu timed out!! LEn of imu queue: %f\n" %len(self.imu_queue))

            z_imu = np.array([[0], [0]])
            imu_noise = np.array([[0.3, 0.], [0., 0.25]])     
            imu_time = rospy.get_rostime()
            imu_pack = [z_imu, imu_noise, imu_time]
            if len(self.imu_queue) > 3:
                self.imu_queue = self.imu_queue[-3:]
            self.imu_queue.append(imu_pack)
            self.new_imu = True

        if(len(self.odom_queue) > 0 and (rospy.get_rostime() - self.odom_queue[-1][2]).to_sec() > self.sensor_timeout):
            # odom timed out
            print("odom timed out!!")

            z_odom = np.array([[0], [0], [0]])
            odom_noise = np.array([[0.5**2, 0, 0], [0, 0.5**2, 0], [0, 0, 0.7**2]])   
            odom_time = rospy.get_rostime()
            odom_pack = [z_odom, odom_noise, odom_time]
            if len(self.odom_queue) > 3:
                self.odom_queue = self.odom_queue[-3:]
            self.odom_queue.append(odom_pack)
            self.new_odom = True
            
   
if __name__ == '__main__':
    try:
        ekf  = EKF_ugv()
        rate = rospy.Rate(ekf.rate)
        while ekf.start < 1:
            continue
        ekf.initialize()
        while not rospy.is_shutdown():
            ekf.Kalman()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass