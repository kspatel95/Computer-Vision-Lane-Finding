## Advanced Lane Finding

### Project 2 Writeup - Krishan Patel

---

**Advanced Lane Finding Project**

My goals and steps for this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/1.Chessboard_Images/chessboard_comparison2.png "Undistorted"
[image2]: ./output_images/2.Undistorted_Images/test3_undistorted.png "Road Transformed"
[image3]: ./output_images/3.Threshold_Images/Binary/test3_thresholds.png "Binary Example"
[image4]: ./output_images/3.Threshold_Images/Color_Binary/test3_colorthresho.png "Multiple Binary Stacked Example"
[image5]: ./output_images/4.Perspective_Images/Street_View/test3_topview.png "Warp Street View Example"
[image6]: ./output_images/4.Perspective_Images/Threshold_View/test3_threshold_topview.png "Warp Threshold View Example"
[image7]: ./output_images/5.Finding_Lane_Lines/sliding_windows2.png "Fit Visual"
[image8]: ./output_images/6.Original_Perspective_with_Markup/test3_markup.png "Output"
[video1]: ./output_videos/final.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

To Calibrate the camera I began with a single image of a chessboard to first ensure that the corners on the grayscale image were able to be detected on the 9 x 6 layout using `cv2.findChessboardCorners(gray, (9,6), None)`. Then I marked the image with `cv2.drawChessboardCorners(img, (9,6), corners, ret)` which marks the corners and links them in an array configuration (left to right down each row) due to the method they were identified. The corners were added to two empty arrays, one dealing with the object space (3D) and the other with image space (2D). This then gets passed on to the calibration function `cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)` which provides us with matrix and distortion of the camera (along with rvecs and tvecs. Now that the whole process works as expected I switched the input to using glob to add all the calibration images in order to get a more accurate calibration. The matrix and distortion are then saved to a file to use later for our normal images. At this point any image taken with the camera can be undistorted using the undistort function `cv2.undistort(img, mtx, dist, None, mtx)`. Below is an example of the chessboard before and after the undistortion process.

![alt text][image1]
Chessboard original vs undistorted

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Now using the values found from the camera calibration we can apply the matrix and distortion to undistort the images of the road. Using the undistort function `cv2.undistort(img, mtx, dist, None, mtx)` we can correct any distortion that we picked up from the chessboard calibration. I have a side by side comparison of the original vs undistorted image, although it may be minimal it will ultimately end up affecting calculations and the rest of the algorthim if it is not performed early on.

![alt text][image2]
Road image original vs undistorted

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of three different thresholds to be able to have the most coverage. None of the thresholds on their own would identify both lane lines cleanly on its own so a combination of three helps to cover in the less than ideal conditions. The threshold that covered a majority of the left side lane was saturation from HLS color space. The right was covered by luminance from HLS, both S and L channel binaries helped provide solid coverage and I added an additional magnitude of gradient threshold from the square root of the sum of squares in the sobel x and y. This filled in any gaps when the color space thresholds might not have the best coverage, such as shadows. I calculated all the threshold possibilities to try out in my pipeline but settled on the three top performers, I included a summary of the thresholds I did use below. I included two images, one showing the binary image of the combined output of the three thresholds and another coloring each of the individual thresholds. I also later added a region of interest to clear out portions of the image that do not contain any of the lane, this allowed future processes to have a cleaner job in finding the lane when conditions were less than optimal.

```
def thresholds(img,s_thresh=(160,255),l_thresh=(190,255),v_thresh=(200,255),sx_thresh=(60,100),
               mag_thresh=(100,255),dir_thresh=(0.7,1.3),sobel_kernel=5):
    # S Channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # L Channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    # Magnitude of Gradient
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    gradmag_binary = np.zeros_like(gradmag)
    gradmag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    color_binary = np.dstack((gradmag_binary, l_binary, s_binary)) * 255
    combined_binary[(gradmag_binary == 1) | (l_binary == 1) | (s_binary == 1)] = 1
````

![alt text][image3]
Combined binary image

![alt text][image4]
Thresholds on separate color channels

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To alter the perspective from the original to a top view (birds-eye view) I created a funtion that takes in an image and also a toggle for which way you wanted the perspective to change. The source and destination points were in the function so can be altered based on the value of 'view', choosing either 1 or 0 (top view or original perspective). I used percentages based on the image shape dimensions to lock in coordinates for the source and destination points. For the destination I used an offset to assist with lining up the image in the top view perspective on straight_lines test images. By using the straight_lines test images it helped with setting up the transform to be parallel and show straight lines and help visualize how it should be. I created a filter on the lower portion of the code snippet to filter between which type of perspective transform I wanted, since I knew I would be ultimately switching it back I thought this was useful to undo the change later down the road. I attached included the perspective transform with the road visible as well as with the thresholded image for both options.

```
def perspective(img, view=1):
    # Warp your image to a top-down view or original perspective
    offset = 450 # offset for dst points
    # src points
    src = np.float32([(img.shape[1]*.32,img.shape[0]),(img.shape[1]*.47,img.shape[0]*.65),
                      (img.shape[1]*.53,img.shape[0]*.65),(img.shape[1]*.68,img.shape[0])])
    # dst points
    dst = np.float32([[offset,img.shape[0]],[offset,img.shape[0]*0],
                      [img.shape[1]-offset, img.shape[0]*0],[img.shape[1]-offset,img.shape[0]]])
    # Get M, the transform matrix
    if view==1:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    elif view==0:
        M = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return warped
```

![alt text][image5]
Birds-eye view of the road

![alt text][image6]
Birds-eye view of the threshold image

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After changing the perspective I began with using a histogram on the lower portion of the image to locate the lines since they are predominantly vertical on the lower half. This is the starting point for the next process where I place windows on both halves of the image to isolate the left and right lane lines. Once the sliding windows are in place and pinpoint the pixels related to the lane line from top to bottom and polynomial fit line is placed and fits between the windows so that it overlays on the lane pixels. The image attached shows the left and right lanes colored differently with sliding windows in green that center themselves on the lane pixels on the respective halves to plot the polynomial fit line in yellow.

![alt text][image7]
Lane lines identified, sliding windows, and polynomial fit line

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the curvature and the distance from the center of the lane I used information already available in the image processing steps. For curvature I used a function pulling values left_fit, right_fit, and ploty with conversion factors for switching pixels to meters (ym_per_pix and xm_per_pix): left_fit_cr and right_fit_cr. This gives a more accurate representation of the calculations but is making a few assumptions: the road is a level plane, the width of the lanes, the length of a segment of the road ahead. This gives a good estimate but that is about as useful as it is. As for the the distance from the center I took the variables used in the histogram to determine the distance from center of the two x values and finding the difference from the center of the image. By doing so I am able to find the distance from center of lane and also the direction left/right.

```
def measure_curvature_meters(ploty,left_fit_cr,right_fit_cr,ym_per_pix,xm_per_pix):
    # Define y-value where we want radius of curvature
    # Maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Calculation of R_curve (radius of curvature)
    left_curverad_real = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad_real = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I used to fit lines to cast a green polygon inward from both the left and right to create the seemless green lane fill along with the lane markers highlighted in red and blue. Afterwards I used to perspective transform function from early but to send it back to the original perspective along with the lane markings. The output image is a green fill on the lanes that fill the portion between the polynomial fit lines on either side. I have listed the image below.

![alt text][image8]
Lane markup from original perspective

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

My pipeline for video displays the green lane between the polynomial fit lines and follows along with the lane. There is text on the top half displaying the curvature of the road as well as the distance from the center along with the direction.
Here's a [link to my video result](.output_videos/final.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some pain points in my pipeline would be it is highly focused on predominantly straight and subtle freeway curves due to a mask I placed to clean up the threshold images that caused extra noise in portions of the road segment due to shadows, extra lines on the ground, unclear lane markings, and other reasons. With this said it functions fairly well on the project video but would have a much tougher time along a curvier road without opening up the region of interest. Some of the thresholds used also have their own weaknesses such as S and L channels in the color space needing visible lane lines which is difficult to uncover in poorly lit environments. With the region of interest it can help with guiding the lanes in the general direction but it is doing so in a blind fashion since it cannot confirm the lane lines are guaranteed to be ahead in that same direction. Ground truth would be helpful in this case to ensure when all else fails that there is some boundary that the vehicle can follow through other means (IMU, GPS, Lidar). 
