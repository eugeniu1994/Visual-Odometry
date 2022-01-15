# Monocular-Visual-Odometry
See 'Monocular_odometry.py'
This is an example of visual localization based on mono camera sensor.
Intrinsic parameters (K matrix is known) .
1. Detect feature descriptors (harris corner, SIFT)
2. Track the features in 2 consecutive images, using optical flow.
3. Estimate Essential matrix (E) between points correspondences, using RANSAC model fitting.
4. Estimate camera rotation (R) and translation (t) based on essential matrix.
5. Update the current pose based on R and t.


<img src="visual.gif" />


# Stereo-Visual-odometry

See 'Stereo_odometry.py'
Camera intrinsic and extrinsic parameters are known
1. Undistort the images
2.Detect features in Left_img_tk, Right_img_tk, Left_img_tk+1, Right_img_tk+1.
3. Estimate features and track them across 4 images
4. Compute the sparse disparity map and estimate the 3D location of the points at time t and t+1
5. In the FrontEnd, estimate the initial transformation between the pointclouds at time t and t+1 using the Iterative closest point cloud (ICP) method.
6. Construct a PoseGraph and optimize the poses.
7. Plot the result

<img src="stereo.gif" />

Bundle Adjustment example, to optimize camera poses, 3D points locations and intrinsic parameters. See 'BA.py' 
