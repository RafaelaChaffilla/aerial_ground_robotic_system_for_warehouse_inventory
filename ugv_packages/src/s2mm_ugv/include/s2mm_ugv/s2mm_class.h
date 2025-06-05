#ifndef S2MM_CLASS
#define S2MM_CLASS

#include "ros/ros.h"
#include "ros/console.h"
#include "std_msgs/String.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/OccupancyGrid.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include <tf2/LinearMath/Quaternion.h>
#include "tf2/impl/convert.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <iostream>
#include <cmath>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <mutex>
#include <iomanip>

using namespace std;
using namespace Eigen;

class S2MM_class{
    public:
        S2MM_class(ros::NodeHandle *nh){
            ROS_INFO("S2MM Node initialized\n");
            
            // Parameters
            string scan_topic, odom_topic, map_topic, loc_topic;
            double init_x, init_y, init_th;

            ros::param::param<string>("~scan_topic", scan_topic, "full_laser_scan");
            ros::param::param<string>("~odom_topic", odom_topic, "/rovid/ekf_ugv");
            ros::param::param<string>("~map_topic", map_topic, "/rovid/map_full");
            ros::param::param<string>("~localization_topic", loc_topic, "/rovid/s2mm_odom");

            ros::param::param<double>("~init_x", init_x, 0.0);
            ros::param::param<double>("~init_y", init_y, 0.0);
            ros::param::param<double>("~init_th", init_th, 0.0);

            ros::param::param<bool>("~publish_tf", publish_tf, true);
            ros::param::param<string>("~output_folder", output_folder, "f_run");
            // cout << "output_folder: " << output_folder << endl;

            // Current pose set
            init_pose << init_x, init_y, init_th;
            position_estimate << init_x, init_y, init_th;
            current_odom_pose.pose.pose.position.x = init_x;
            current_odom_pose.pose.pose.position.y = init_y;
            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, init_th);
            q.normalize();
            current_odom_pose.pose.pose.orientation = tf2::toMsg(q);

            // Subscribers
            scan_sub = nh->subscribe<sensor_msgs::LaserScan>(scan_topic, 1, &S2MM_class::laser_handler, this);
            odom_sub = nh->subscribe<nav_msgs::Odometry>(odom_topic, 1, &S2MM_class::odom_pose_update, this);
            map_sub = nh->subscribe<nav_msgs::OccupancyGrid>(map_topic, 10, &S2MM_class::receive_map, this);
            
            // Publishers
            pose_publisher = nh->advertise<nav_msgs::Odometry>(loc_topic, 10);

        }

        void receive_map(const nav_msgs::OccupancyGrid::ConstPtr& msg){
            map_resolution = (msg->info).resolution;
            origin_coords << (msg->info).origin.position.x, (msg->info).origin.position.y;
            cv::Mat map_cv = cv::Mat((msg->info).height, (msg->info).width, CV_8U, (void*)msg->data.data());
            cv::Mat binary_map;
            cv::threshold(map_cv, binary_map, 20, 250, 1);
            cv::Mat distance_transform_map;
            label_to_index.clear();
            DistanceMapIndeces(binary_map, distance_transform_map, map_transform_indeces, label_to_index);
            
            map_received = true;
            map_sub.shutdown();
        }

        void odom_pose_update(const nav_msgs::Odometry::ConstPtr& msg){
            if (time_odom_list.size() >= 20){ //if array is already full
                time_odom_list.pop_front();
                odom_list.pop_front();
            }
            odom_list.push_back(*msg);
            time_odom_list.push_back(msg->header.stamp.toSec());

        }

        void scan_matching(VectorXd ox, VectorXd oy, double &error, Matrix3d &inverse_Hessian){
            tf2::Quaternion quat;
            tf2::fromMsg(current_odom_pose.pose.pose.orientation, quat);
            double roll, pitch, yaw;
            tf2::Matrix3x3 (quat).getRPY(roll, pitch, yaw);
            Vector3d initial_estimate;
            initial_estimate << current_odom_pose.pose.pose.position.x, current_odom_pose.pose.pose.position.y, yaw;

            Vector2d sigma_3_x_y;
            sigma_3_x_y << sqrt(current_odom_pose.pose.covariance[0])*3, sqrt(current_odom_pose.pose.covariance[7])*3;

            VectorXd ox_copy = ox;
            VectorXd oy_copy = oy;
            
            MinimizeError(ox_copy, oy_copy, map_transform_indeces, label_to_index, initial_estimate, map_resolution, origin_coords, sigma_3_x_y, position_estimate, error, inverse_Hessian);
        }
    
        void laser_handler(const sensor_msgs::LaserScan::ConstPtr& msg){
            min_range = msg->range_min;
            max_range = msg->range_max;
            
            std::vector<double> ox_vector, oy_vector;
            
            for (int i=0; i < (msg->ranges).size();  ++i){
                if ((msg->ranges)[i] > LIDAR_ROI){
                    continue;
                }
                ox_vector.push_back((msg->ranges)[i]*cos(msg->angle_min+i*(msg->angle_increment)));
                oy_vector.push_back((msg->ranges)[i]*sin(msg->angle_min+i*(msg->angle_increment)));
            }

            ox.resize(ox_vector.size());
            oy.resize(oy_vector.size());

            ox = Eigen::Map<Eigen::VectorXd>(ox_vector.data(), ox_vector.size());
            oy = Eigen::Map<Eigen::VectorXd>(oy_vector.data(), oy_vector.size());

            laser_time = (msg->header).stamp;
        
        }

        void estimate_position(){
            double laser_time_secs = laser_time.toSec(), aux = 0;
            ros::Time publish_laser_time = laser_time;
            double min_difference = abs(time_odom_list[0] - laser_time_secs);
            int min_odom_ind = 0;

            for (int i=1; i<time_odom_list.size(); ++i){
                aux = abs(time_odom_list[i] - laser_time_secs);
                if (aux < min_difference){
                    min_difference = aux;
                    min_odom_ind = i;
                }
            }

            current_odom_pose = odom_list[min_odom_ind];
            
            if (!map_received or ox.size() == 0){
                return;
            }
            
            scan_matching(ox, oy, error, inv_hess);
            
            tf2::Quaternion new_quaternion;
            new_quaternion.setRPY(0,0,position_estimate[2]);
            new_quaternion = new_quaternion.normalize();
            if (isnan(position_estimate[0])){
                return;
            }
            nav_msgs::Odometry new_pose;
            new_pose.header.stamp = publish_laser_time;
            new_pose.header.frame_id = "map";
            new_pose.pose.pose.position.x = position_estimate[0];
            new_pose.pose.pose.position.y = position_estimate[1];
            new_pose.pose.pose.orientation = tf2::toMsg(new_quaternion);
            new_pose.pose.covariance[0]  = inv_hess(0,0);
            new_pose.pose.covariance[7]  = inv_hess(1,1);
            new_pose.pose.covariance[35] = inv_hess(2,2);
            pose_publisher.publish(new_pose);
        }

    private:
        // ROS Related
        ros::Publisher pose_publisher;
        ros::Publisher map_publisher;
        ros::Subscriber map_sub;
        ros::Subscriber scan_sub;
        ros::Subscriber odom_sub;
        // Position info
        Vector3d init_position, position_estimate;
        // Odometry info
        nav_msgs::Odometry current_odom_pose;
        deque<nav_msgs::Odometry> odom_list;
        deque<double> time_odom_list;
        // Map info
        double map_resolution;
        Vector3d init_pose;
        Vector2d origin_coords;
        MatrixXi previous_map;
        cv::Mat map_transform_indeces;
        vector<cv::Vec2i> label_to_index;
        // Laser info
        double min_range, max_range;
        ros::Time laser_time;
        VectorXd ox, oy;
        // General
        Matrix3d inv_hess{};
        double error = 0;
        bool publish_tf;
        bool map_received = false;

};

#endif