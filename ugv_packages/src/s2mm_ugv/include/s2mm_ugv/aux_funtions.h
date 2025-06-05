#ifndef AUX_FUNCTIONS
#define AUX_FUNCTIONS

#include "ros/ros.h"

#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <random>
#include <mutex>
#include <algorithm>

#include <chrono>

using namespace std;
using namespace Eigen;

float LIDAR_ROI = 15;
double ET1_DEFAULT = 15;
double e_t_1 = ET1_DEFAULT;
double lambda = 1;

Vector2d ind_to_pos(Vector2d index, Vector2d origin_position, double map_resolution){
    Vector2d position;
    position << origin_position[0] + index[1]*map_resolution, origin_position[1] + index[0]*map_resolution;
    return position;
}

Vector2d pos_to_ind(Vector2d pose, Vector2d origin_position, double map_resolution){
    Vector2d indexes;
    indexes << (int)((pose[1]-origin_position[1])/map_resolution), (int)((pose[0]-origin_position[0])/map_resolution);
    return indexes;
}

MatrixXi pos_to_ind_vectorized(MatrixXd poses, Vector2d origin_position, double map_resolution){
    // pose must be a 2xN matrix like [[x0, x1, x2, x3, ...],
    //                                 [y0, y1, y2, y3, ...]]
    // returns 2xN with [[col0, col1, col2, col3, ...],
    //                   [row0, row1, row2, row3, ...]]
    MatrixXi indexes;
    MatrixXd origin_id(2,2);
    origin_id << origin_position[0], 0,
                 0, origin_position[1];

    indexes = ((poses -  origin_id*MatrixXd::Ones(2, poses.cols()))/map_resolution).cast <int> ();
    
    return indexes;
}

void DistanceMapIndeces(cv::Mat map, cv::Mat &distanceTransformMap, cv::Mat &mapTransformIndeces, vector<cv::Vec2i> &label_to_index){
    cv::distanceTransform(map, distanceTransformMap, mapTransformIndeces, cv::DIST_L2, 3, cv::DIST_LABEL_PIXEL);
    label_to_index.push_back(-1); // because the indeces start at 1 so we need to store someting in index 0
    for (int row = 0; row < map.rows; ++row)
        for (int col = 0; col < map.cols; ++col)
            if(map.at<uchar>(row,col)==0) // is obstacle
                label_to_index.push_back(cv::Vec2i(row,col));

}

void Si_vectorized(MatrixXd si, Vector3d epsilon, double sine_epsilon, double cossine_epsilon, MatrixXd &Si_value, MatrixXd &delta_si_value){
    //    si received in meters and in robot's body frame. epsilon in world frame
       //  Parameters:
           //  si: Matrix 2xN where N is the number of laser points: [[six0, six1, six2, six3, six4,...],
           //                                                         [siy0, siy1, siy2, siy3, siy4,...]]
           //  epsilon: Robot position ([x,y,yaw] in world frame)
       //  Returns:
           //  Si_value: Laser position [Six, Siy] in world frame
           //  delta_si_value: third column of the Si derivative in epsilon. for each laser scan, the matrix is [[1, 0, a], [0,1,b]], this returns [[a0, a1, a2, a3, a4,...],
           //                                                                                                                                       [b0, b1, b2, b3, b4,...]]
    Matrix2d rot_mat;
    rot_mat << cossine_epsilon, -sine_epsilon,
               sine_epsilon, cossine_epsilon;

    Matrix2d epsilon_id;
    epsilon_id << epsilon[0], 0,
                  0, epsilon[1];
    MatrixXd epsilon_matrix = MatrixXd::Ones(si.rows(), si.cols());
    epsilon_matrix = epsilon_id*epsilon_matrix;
    Si_value = rot_mat*si + epsilon_matrix;
    Matrix2d delta_rot_mat;
    delta_rot_mat << -sine_epsilon, -cossine_epsilon,
                     cossine_epsilon, -sine_epsilon;

    delta_si_value = delta_rot_mat*si;
}

void Si(double si_x, double si_y, Vector3d epsilon, double sine_epsilon, double cossine_epsilon, Vector2d &Si_value, MatrixXd &delta_si_value){
//         si_x and si_y received in meters and in robot's body frame. epsilon in world frame
//         Parameters:
//             si_x: laser x position in robot body frame
//             si_y: laser y position in robot body frame
//             epsilon: Robot position ([x,y,yaw] in world frame)
//         Returns:
//             Si_value: Laser position [Six, Siy] in world frame
//             delta_si_value: Si derivative in epsilon
    
    Si_value << cossine_epsilon*si_x - sine_epsilon*si_y + epsilon[0], 
                sine_epsilon*si_x + cossine_epsilon*si_y + epsilon[1];

    delta_si_value << 1, 0, -sine_epsilon*si_x - cossine_epsilon*si_y, 
                      0, 1, cossine_epsilon*si_x - sine_epsilon*si_y;
    // delta_si_value << 1, 0, 0, 
    //                   0, 1, 0;

};

void M(cv::Mat &mapTransformIndeces, vector<cv::Vec2i> &label_to_index, Vector2d point_ind, Vector2d delta_si, double map_res, Vector2d origin_position, double &M_value, RowVector2d &delta_M_array){
//         Parameters:
//             inds: [x_array, y_array] that indicates closest obstacle to each point in map frame
//             Point_ind: [col,row] position in world frame to be analysed
//             mas_res: resolution of map
//             origin_position: [x,y] in world frame indicating position of indeces [0,0] in map matrix
//         Returns:
//             M_value: Distance of Point to closest obstacle in map
//             M_delta: [dM/dx, dM/dy], in world coordinates
    // cout << "point index:" << point_ind << endl;
    int closest_obstacle_label = mapTransformIndeces.at<int>((int)point_ind[1], (int)point_ind[0]);
    delta_M_array << point_ind[0]-label_to_index[closest_obstacle_label][1], point_ind[1]-label_to_index[closest_obstacle_label][0];
    M_value = sqrt(pow(delta_M_array[0], 2) + pow(delta_M_array[1], 2)); 
    delta_M_array = delta_M_array/M_value; 
    M_value = M_value*map_res;
}

void outlier_detection(VectorXd &M_value, double percentage_outliers){
    // RELATIVE MOTION THRESHOLD
    int n = M_value.rows();
    double e_t;
    if (e_t_1 < lambda*e_t_1){
        e_t = e_t_1;
    }else{
        e_t = lambda*e_t_1;
    }

    e_t_1 = e_t;

    // changes distance of outlier points to 0 so it doesn't affect the error
    VectorXd below_dmax_boolean(n); // 1 is where M should stay the same, 0 is where it is an outlier
    below_dmax_boolean = (M_value.array() <= e_t + 0.20).cast<int>().cast<double>();
    
    VectorXd pipi = below_dmax_boolean.array()*M_value.array();
    
    double difference = (M_value - pipi).norm();
    double number_non_outliers = (below_dmax_boolean.transpose()*VectorXd::Ones(n))[0];
    VectorXi outlier_indeces = (M_value.array() <= e_t + 0.20).cast<int>();
    double number_non_outliers_int = (outlier_indeces.transpose()*VectorXi::Ones(n))[0];
    
    M_value = below_dmax_boolean.array()*M_value.array();

    percentage_outliers = (n-number_non_outliers_int)/n;

}

void applyMToMatrix(cv::Mat &mapTransformIndeces, vector<cv::Vec2i> &label_to_index, MatrixXd &points, double map_res, Vector2d origin_position, VectorXd &M_values, MatrixXd &delta_M_matrix) {
    int num_points = points.cols();
    M_values.resize(num_points);
    delta_M_matrix.resize(2, num_points);
    auto poses_ind = pos_to_ind_vectorized(points, origin_position, map_res);
    Vector2i point_index;
    MatrixXi closest_obstacles(2, num_points);
    int closest_obstacle_label;
    for(int i=0; i < points.cols(); ++i){
        point_index = poses_ind.col(i);
        if(point_index[1] > 0 && point_index[1] < mapTransformIndeces.rows && 
           point_index[0] > 0 && point_index[0] < mapTransformIndeces.cols){
            closest_obstacle_label = mapTransformIndeces.at<int>(point_index[1], point_index[0]);
            closest_obstacles.col(i) << label_to_index[closest_obstacle_label][1], label_to_index[closest_obstacle_label][0];
           }
        }
    delta_M_matrix = (poses_ind - closest_obstacles).cast <double> ()*map_res;
    M_values = (delta_M_matrix.array().square().colwise().sum()).sqrt();
    delta_M_matrix = delta_M_matrix.array().rowwise() / M_values.transpose().array();
    delta_M_matrix = (M_values.array() == 0).transpose().replicate(delta_M_matrix.rows(), 1).select(0, delta_M_matrix);
}

void DeltaEpsilon(VectorXd ox, VectorXd oy, cv::Mat &mapTransformIndeces, vector<cv::Vec2i> &label_to_index, Vector3d epsilon, double map_res, Vector2d origin_position, 
                  Vector3d &delta_epsilon, double &error_x, double &error_y, double &error_th, Matrix3d &inverted_Hessian){
    // Initializing Variables
    Vector3d G{}, aux_error_2{};
    Matrix3d H{};

    VectorXd M_value;
    MatrixXd delta_M_value;
    MatrixXd aux(3, ox.size());
    MatrixXd aux_error(aux.rows(), aux.cols()); 
    
    MatrixXd delta_Si_values(2,ox.size());
    MatrixXd Si_values(2,ox.size());

    MatrixXd si(2, ox.size());
    
    double ox_number = 0;
    double sine_epsilon = sin(epsilon[2]), cossine_epsilon = cos(epsilon[2]);

    // Calculate Transformation of laser scans
    si.row(0) = ox;
    si.row(1) = oy;
    Si_vectorized(si, epsilon, sine_epsilon, cossine_epsilon, Si_values, delta_Si_values);
    // Calculate M function
    applyMToMatrix(mapTransformIndeces, label_to_index, Si_values, map_res, origin_position, M_value, delta_M_value);
    // Apply Laser outlier removal
    double perc_outliers;
    outlier_detection(M_value, perc_outliers);
    // Calculate delta epsilon
    aux << delta_M_value.row(0), 
           delta_M_value.row(1), 
           (delta_M_value.cwiseProduct(delta_Si_values)).colwise().sum();
    H = aux*aux.transpose();
    if (H.determinant() < 10e-6) {
        H = H + Matrix3d::Identity()*10e-3;
    }
    G = aux* M_value;
    
    delta_epsilon = H.inverse() * G;
    inverted_Hessian << H.inverse();
    // calculate errors
    aux_error = (aux.array().rowwise() * M_value.transpose().array()).array().square().eval();
    aux_error_2 = aux_error.rowwise().sum();
    error_x = aux_error_2[0];
    error_y = aux_error_2[1];
    error_th = aux_error_2[2];

}

void MinimizeError(VectorXd ox, VectorXd oy, cv::Mat &mapTransformIndeces, vector<cv::Vec2i> &label_to_index, Vector3d epsilon, double map_res, Vector2d origin_position, Vector2d sigma_3_x_y,
                   Vector3d &estimation, double &error, Matrix3d &inverted_Hessian, int max_iter = 50, double tolerance = 0.05){

    Vector3d estimation_copy;
    estimation_copy << epsilon[0], epsilon[1], epsilon[2];

    VectorXd errors(9), errors_x(9), errors_y(9), errors_th(9);
    vector<Matrix3d> hessians(9);
    errors << 0,0,0,0,0,0,0,0,0;
    errors_x << 0,0,0,0,0,0,0,0,0;
    errors_y << 0,0,0,0,0,0,0,0,0;
    errors_th << 0,0,0,0,0,0,0,0,0;
    vector<Vector3d> x_changed(9);
    
    vector<Vector3d> x_neighbours = {
        epsilon,
        epsilon + Vector3d(-2, 0, 0),
        epsilon + Vector3d(1, 1, 0), 
        epsilon + Vector3d(1, -1, 0), 
        epsilon + Vector3d(-1, 1, 0), 
        epsilon + Vector3d(-1, -1, 0),
        epsilon + Vector3d(0, 2, 0), 
        epsilon + Vector3d(0, -2, 0),
        epsilon + Vector3d(2, 0, 0)
    };

    Vector3d delta_epsilon{}, prev_delta_epsilon{}; 
    float prev_error = 0;
    
    for (size_t i = 0; i < x_neighbours.size(); ++i) {
        Vector3d x = x_neighbours[i];

        e_t_1 = 1.5*sqrt(sigma_3_x_y[0]*sigma_3_x_y[0] + sigma_3_x_y[1]*sigma_3_x_y[1]) ;
        lambda = 1;

        prev_delta_epsilon = Vector3d::Zero();

        for (size_t j = 0; j < max_iter; ++j) {
            DeltaEpsilon(ox, oy, mapTransformIndeces, label_to_index, x, map_res, origin_position, delta_epsilon, errors_x[i], errors_y[i], errors_th[i], hessians[i]);
            
            if (isnan(delta_epsilon[0]) || isnan(delta_epsilon[1]) || isnan(delta_epsilon[2])){
                break;
            }

            x -= delta_epsilon;
            if((delta_epsilon + prev_delta_epsilon).norm() < 0.05){
                // its oscillating around one point
                prev_delta_epsilon = Vector3d::Zero();
                if (prev_error <= errors_x[i]+errors_y[i]){
                    x += delta_epsilon;
                }
                break;
            }

            if (pow(delta_epsilon[0], 2) + pow(delta_epsilon[1], 2)  < pow(tolerance+0.005 , 2) && delta_epsilon[2] < tolerance*20){
                // is within the tolerance 
                break;
            }

            if (j > 1){
                lambda = (pow(delta_epsilon[0], 2) + pow(delta_epsilon[1], 2))/(pow(prev_delta_epsilon[0], 2) + pow(prev_delta_epsilon[1], 2));
            }
            
            prev_delta_epsilon = delta_epsilon;
            prev_error = errors_x[i]+errors_y[i];

        }
        
        errors[i] = errors_x[i]+errors_y[i];
        if(abs(x[0]-epsilon[0]) >= sigma_3_x_y(0)|| abs(x[1]-epsilon[1]) >= sigma_3_x_y(1)){ 
            errors[i] = std::numeric_limits<int>::max();
        }

        x_changed[i] = x;
        
    }

    int min_error_index;
    error = errors.minCoeff(&min_error_index);

    std::vector<int> close_indices;
    double min_ekf_diff = (x_changed[min_error_index]-estimation_copy).norm();
    for (int i = 0; i < errors.size(); ++i) {
        if (std::abs(errors[i] - error) <= 0.5 && i != min_error_index) {
            // another estimation has error close to smallest error
            double non_min_ekf_diff = (x_changed[i]-estimation_copy).norm();
            if (min_ekf_diff > non_min_ekf_diff){
                min_error_index = i;
                min_ekf_diff = non_min_ekf_diff;
                error = errors[i];
            }
        }
    }

    estimation = x_changed[min_error_index];

    estimation[2] = fmod((estimation[2] + 3.1416), (2*3.1416)) - 3.1416;
    
    Matrix3d p_aux;
    p_aux << errors_x[min_error_index]/ox.size() + 0.04, 0, 0, 
            0, errors_y[min_error_index]/ox.size() + 0.04, 0, 
            0, 0, errors_th[min_error_index]/ox.size() + 0.04;  // covariance is squared!!
    
    inverted_Hessian = 4*p_aux + Matrix3d::Identity()*0.15*0.15;

    if (error == std::numeric_limits<int>::max() || abs(estimation[0]-epsilon[0]) >= 1){
        estimation[0] = std::numeric_limits<double>::quiet_NaN();
    }
    
}

#endif