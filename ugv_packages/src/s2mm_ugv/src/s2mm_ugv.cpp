#include "../s2mm_ugv/include/s2mm_ugv/aux_funtions.h"
#include "../s2mm_ugv/include/s2mm_ugv/s2mm_class.h"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv){
    ros::init(argc, argv, "s2mm_ugv");
    ros::NodeHandle nh;
    S2MM_UGV s2mm_node(&nh);

    float frequency;
    ros::param::param<float>("~frequency", frequency, 10);
    ros::Rate rate(frequency);

    string scan_topic;
    ros::param::param<string>("~odom_topic", scan_topic, "/rovid/ekf_ugv");
    try {
        boost::shared_ptr<nav_msgs::Odometry const> laser_scan_msg;
        laser_scan_msg = ros::topic::waitForMessage<nav_msgs::Odometry>(scan_topic, nh);

        if (laser_scan_msg != nullptr) {
            ROS_INFO("Received laser scan message.");
        } else {
            throw ros::Exception("No message received within the specified duration.");
        }
    } catch (ros::Exception& e) {
        ROS_ERROR("Error occured: %s ", e.what());
    }
    
    while (ros::ok()){
        s2mm_node.estimate_position();
        rate.sleep();
        ros::spinOnce();
    }
    
    return 0;
}