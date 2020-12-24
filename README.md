# Visual-Odometry
This is an example of visual localization based on mono camera sensor.
Intrinsic parameters (K matrix is known) .
1. Detect feature descriptors (harris corner, SIFT)
2. TRack the features in 2 consecutive images, using optical flow.
3. Estimate Essential matrix (E) between points correspondences, using RANSAC model fitting.
4. Estimate camera rotation (R) and translation (t) based on essential matrix.
5. Update the current pose based on R and t.


<img src="visual.gif" />
