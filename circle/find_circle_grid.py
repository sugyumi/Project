import glob
import numpy as np
import cv2
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()
# Change thresholds
blobParams.minThreshold = 188  # 원래 8
blobParams.maxThreshold = 255  # 원래 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64  # minArea may be adjusted to suit for your experiment -> 원래 값 64
blobParams.maxArea = 2500  # maxArea may be adjusted to suit for your experiment -> 원래 값 2500

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1  # 원래 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87  # 원래 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01  # 원래 0.01

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

###################################################################################################

###################################################################################################

# Original blob coordinates, supposing all blobs are of z-coordinates 0
# And, the distance between every two neighbour blob circle centers is 72 centimetres
# In fact, any number can be used to replace 72.
# Namely, the real size of the circle is pointless while calculating camera calibration parameters.
objp = np.zeros((44, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 72 , 0)
objp[2]  = (0  , 144, 0)
objp[3]  = (0  , 216, 0)
objp[4]  = (36 , 36 , 0)
objp[5]  = (36 , 108, 0)
objp[6]  = (36 , 180, 0)
objp[7]  = (36 , 252, 0)
objp[8]  = (72 , 0  , 0)
objp[9]  = (72 , 72 , 0)
objp[10] = (72 , 144, 0)
objp[11] = (72 , 216, 0)
objp[12] = (108, 36,  0)
objp[13] = (108, 108, 0)
objp[14] = (108, 180, 0)
objp[15] = (108, 252, 0)
objp[16] = (144, 0  , 0)
objp[17] = (144, 72 , 0)
objp[18] = (144, 144, 0)
objp[19] = (144, 216, 0)
objp[20] = (180, 36 , 0)
objp[21] = (180, 108, 0)
objp[22] = (180, 180, 0)
objp[23] = (180, 252, 0)
objp[24] = (216, 0  , 0)
objp[25] = (216, 72 , 0)
objp[26] = (216, 144, 0)
objp[27] = (216, 216, 0)
objp[28] = (252, 36 , 0)
objp[29] = (252, 108, 0)
objp[30] = (252, 180, 0)
objp[31] = (252, 252, 0)
objp[32] = (288, 0  , 0)
objp[33] = (288, 72 , 0)
objp[34] = (288, 144, 0)
objp[35] = (288, 216, 0)
objp[36] = (324, 36 , 0)
objp[37] = (324, 108, 0)
objp[38] = (324, 180, 0)
objp[39] = (324, 252, 0)
objp[40] = (360, 0  , 0)
objp[41] = (360, 72 , 0)
objp[42] = (360, 144, 0)
objp[43] = (360, 216, 0)
###################################################################################################

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Find image files in the Circle_pattern directory
image_files = glob.glob("image*.bmp")
# Check the found image files
print("Image files found:", image_files)

found = 0

for file in image_files:
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints = blobDetector.detect(gray) # Detect blobs.

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    if ret == True:
        objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

        corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
        imgpoints.append(corners2)

        # Draw and display the corners.
        im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
        found += 1

        # Save the image with detected keypoints
        # cv2.imwrite(f"detected_{file}", im_with_keypoints)
        ################
        # im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
        save_name = "detected_" + file  # 새로운 이름 생성
        cv2.imwrite(save_name, im_with_keypoints)  # 이미지 저장
        ################


    re_iwk = cv2.resize(im_with_keypoints, None, fx=0.8, fy=0.8)
    cv2.imshow("img", re_iwk) # display
    # cv2.imshow("img", im_with_keypoints) # display

    # cv2.waitKey(2)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# When everything done, release the capture
cv2.destroyAllWindows()

# Check the length of objpoints and imgpoints
print("Number of images used for calibration:", len(objpoints))

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print camera matrix (Intrinsic parameters)
print("Camera Matrix (Intrinsic Parameters): ", mtx)

# Print distortion coefficients
print("Distortion Coefficients: ", dist)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints_reprojection, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints_reprojection, cv2.NORM_L2) / len(imgpoints_reprojection)
    mean_error += error

mean_error /= len(objpoints)

# Print reprojection error
print("Reprojection Error: ", mean_error)

data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)
with open('calibration.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.SafeLoader)
mtxloaded = loadeddict.get('camera_matrix')
distloaded = loadeddict.get('dist_coeff')