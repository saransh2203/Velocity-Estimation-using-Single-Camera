# Velocity-Estimation-using-Single-Camera
A part of driver assist system, we have developed a program to estimate the velocity and distance of an identified vehicle in the view-frame. We have also developed a code to help estimate the lane along with its curve.

This project consists of 3 broad aspects. 
The 1st one being a custom object detector model trained for 6 different class which include HMV, LMV, 2-Wheeler, Auto, Pedestrian & Animal. Over 5,300 images were trained using YOLOv8 for 225 epochs and 80+% validation was achieved. This module also contains another object detection model which is trained for only Car and 2-Wheeler but for separate orientations. 4 orientations including Front/Rear, 45_deg, 90_deg and 135_deg have been trained for with suitable images of each Car and 2-Wheeler. This was done in order to identify the orientation of the vehicle as well along with its class such that the distance can be appropriately calculated irrespective of its inclination with respect to the camera. This dataset consisted of 3,600 images and was trained for 200 epochs. It attained low validation and hence is just used to create a proof of concept. 

The 2nd part of the project consisted of a lane detection model. It includes a program which can identify and track lanes on which the vehicle is being driven on and segment it from the surroundings. 
