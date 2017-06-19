# Catch the run away car
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, my goal is to write a software pipeline to intercept an escaped car driving in a circular path. The run away car is being sensed by a stationary sensor that can measure both Lidar and Radar data. To capture the run away car this pipeline will use Unscented Kalman Filters, Sensor Fusion and Kinematic Models to predict the position of the escaped car and guide an interceptor car to a position within 0.1 unit distance from it.

## The Project

The steps to accomplish the goal in this project are the following:

* Properly initialize the estimated state of the object being tracked using both Lidar and Radar measurements.
* Predict the next state at t + delta t
* Update your prediction with new measurement coming from either Lidar and Radar
* Use a kinematic model to predict the escaped car position and guide the interceptor vehicle

## Sensor Fusion

Sensor fusion is the technique of successfully combining data from different sensors in a way that the resulting state estimate is more precise than by using each individual sensor.

For autonomous vehicles applications sensor fusion is even more important, as it allows not only a better estimate of the predicted state of objects, but it also overcomes the weaknesses in each individual sensor by having a suite of different sensors that together can very precisely estimate the state of the objects being measured.

The chart below compares several characteristics of the three main sensor in a self-driving car.

![alt text][image1]
Source: Udacity's lesson on sensor fusion.

## Kalman Filters

[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) is an algorithm that uses Bayesian inference to estimate a continuous state of an object that is being observed through noisy measurements from various sensors (Radar, Lidar, etc).

The Kalman filter uses multiple sequential measurements to form an estimate of the system's state (in this case each state represents the position, velocity, etc from the object we are measuring). By leveraging from Bayesian inference it uses the multiple measurements to "filter out" the noise in the sensor data and get a better estimate of the object state than by just plain observation.

The main limitation of the original Kalman filter is that it is based on linear dynamical systems and for that the performance of a standard Kalman filter in real world problems is far from ideal.

Two important modifications of the basic Kalman filters are [Extended Kalman filter (EKF)](https://en.wikipedia.org/wiki/Extended_Kalman_filter) and [Unscented Kalman filter (UKF)](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter) as they both allow for non-linearities. While the Extended Kalman filter addresses the non-linearities by using a linear approximation of the state transition function, the Unscented Kalman filter proposes a more robust technique that generates "random sigma points" based on the current state covariance matrix, these points are then shifted into a new state at t + delta t and then a new state mean and state covariance matrix can be estimated from these shifted points.

Unscented Kalman filters also eliminates the need of calculating Jacobian matrices for the linear approximation of the state transition.

For further reading on Unscented Kalman filters I recommended this excellent [notebook](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb) from Roger Labbe.


## The model

For this project we have to fuse Radar and Lidar measurements to get a robust estimation of the position of the run away car, but before we can apply it to estimate the run away car's position we are asked to fine tune the model by detecting a bicycle moving in front of a car.

We will be  working with a sequence of measurements coming from both sensors and the noise parameters from each sensor and we need to accurrately estimate the location of the bicycle. Ground truth values for the position were also provided so we can measure the performance of our algorithm.

The methodology we will follow is represented in the chart below.
![alt text][image2]
Source: Udacity's lesson on Extended Kalman filters.

The chart is actually for the EKF algorithm, but the UKF implementation follows the same steps, but uses a different methodology in each step.


## Fine tuning hyperparameters

The objective function we are trying to minimize is the RMSE of our predicted state (after the state update step) compared to the ground truth.

We can also check the consistency of our algorithm by using a metric called Normalized Innovation Squared (NIS) which is the difference between the predicted measurement and the actual measurement normalized in relation to the state covariance matrix.

These two metrics together can be used to fine tune the longitudinal and yaw acceleration noise parameters as well as a proper initialization of the state vector and state covariance matrix.

The charts below show that after fine tuning all these parameters we have modeled the longitudinal and yaw acceleration noise with 95% confidence.

![alt text][image3]

![alt text][image4]

## Validating results

With proper fine tuning and initialization we have a model that can estimate the state of a bicycle very accurately. Our estimated position is within 8cm of the real location (0.076m for the x axis and 0.0825m for the y axis) and the estimated velocity is within 1.25 km/h of the real velocity (0.3516 m/s and 0.2273 m/s).

The screen shot below shows the last few steps of the simulator used for evaluating our algorithm. The blue dots are radar measurements, the red circles are lidar measurements and the green triangles are the estimated position of the bicycle by our model.

![alt text][image5]

The most impressive part of these results is that they are more accurate than each of these sensors individually. The LIDAR equipment that was used has an accuracy of 15cm, but by efficiently fusing the radar data with the laser data we effectively reduce the uncertainty of our estimations.

The table below compares the RMSE for the position and velocity along the x and y axis for each individual sensor against the sensor fusion approach.



|RMSE | RADAR | LASER  | SENSOR FUSION | IMPROVEMENT FROM BEST |
|-----|-------|--------|---------------|-----------------------|
|Px   | 0.2279| 0.1841 | 0.0760        | 58.7%                 |
|Py   | 0.2971| 0.1495 | 0.0825        | 44.8%                 |
|Vx   | 0.4466| 0.6908 | 0.3516        | 21.3%                 |
|Vy   | 0.3947| 0.2754 | 0.2273        | 17.5%                 |



## Intercept the run away car

Now that we have a model that can accurately estimate the position of a bicycle we can apply it to the real goal of this project, which is to predict and intercept a self-driving car that has malfunctioned and is running in circles.

This would be easily solved if our interceptor could move faster than the run away car, but to make things more interesting, both vehicles have the same maximum speed, so if we just predict the current position of the rogue vehicle we will never catch it.

In order to overcome this limitation we need to use kinematic models to predict the position of the vehicle in the future, using the current estimated state (px, py, v and yaw) to predict it's location in a future step that is close enough to the current observation (so we won't allow the run away car to change it's path) but also far enough in the future that we can get there before the rogue car by taking a different path (since the speed of both vehicles are the same)

The final results can be seen on the video in this [link](./videos/project_video.mp4).

## References
[1] [The Unscented Kalman Filter](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb)


[//]: # (Image References)

[image1]: ./images/sensors.png "Sensor Comparison"
[image2]: ./images/model.png "Sensor fusion model"
[image3]: ./images/radar_nis.png "Radar NIS"
[image4]: ./images/laser_nis.png "Laser NIS"
[image5]: ./images/sim_screenshot.png "Laser NIS"
