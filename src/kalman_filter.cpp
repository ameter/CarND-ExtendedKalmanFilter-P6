#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Predict the state
  x_ = F_ * x_; //(5-8)
  MatrixXd Ft = F_.transpose(); //(5-9)
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Update the state by using Kalman Filter equations (5-7).
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  
  // New state
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //Update the state by using Extended Kalman Filter equations (5-14).
  float x = x_(0);
  float y1 = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  float rho = sqrt(x * x + y1 * y1);
  float theta = atan2(y1, x);
  float ro_dot = (x * vx + y1 * vy) / rho;
  VectorXd z_pred = VectorXd(3);
  z_pred << rho, theta, ro_dot;
  
  // (5-7)
  VectorXd y = z - z_pred; // before, z_predict was H * x
  
  // Normalize y
  if (y(1) > M_PI) {
    y(1) -= 2 * M_PI;
  } else if (y(1) < -M_PI) {
    y(1) += 2 * M_PI;
  }
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  
  // New state
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
