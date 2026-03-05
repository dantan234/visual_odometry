import cv2 as cv
import numpy as np
from dataset import KITTIDataset

class VisualOdometry:
    """
    A Monocular Visual Odometry pipeline for the KITTI dataset.
    """
    def __init__(self, data_path, sequence):
        """
        Initialize the Visual Odometry System
        
        Args:
            data_path (str): Path to the root 'data' folder.
            sequence (str): Sequence number ie. '00', '01'
        """
        self.dataset = KITTIDataset(data_path, sequence)

        # Camera Matrix for (P0) 3x4 matrix
        self.cam_params = self.dataset.calib_matrix
        self.focal_length = self.cam_params[0,0]
        self.pp = (self.cam_params[0,2], self.cam_params[1,2])
        
        # State variables
        self.old_frame = None
        self.current_frame = None
        self.pts_ref = None # "Reference Points" (Points in old frame)
        self.pts_cur = None # "Current Points" (Points in new frame)
        self.curr_R = np.identity(3)
        self.curr_t = np.zeros((3,1))
        
        # Fast Detector initialization
        self.detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    
    def getAbsoluteScale(self, frame_id) -> float:
        """
        Calculates the Euclidean distance between the current and previous ground truth poses to provide the real-world scale.
        
        Args:
            frame_id (int): The index of the current frame

        Returns:
            float: The absolute distance (scale) moved between the previous frame and current frame.
        """
        curr_pose = self.dataset.poses[frame_id]
        x = curr_pose[0,3]
        y = curr_pose[1,3]
        z = curr_pose[2,3]
        prev_pose = self.dataset.poses[frame_id-1]
        x_prev = prev_pose[0,3]
        y_prev = prev_pose[1,3]
        z_prev = prev_pose[2,3]

        if frame_id == 0:
            return 0.0
        else:
            scale = np.sqrt((x-x_prev)**2 + (y-y_prev)**2 + (z-z_prev)**2)
        return scale

    def featureDetection(self, img) -> np.ndarray:
        """
        Detects new features in an image using the detector.

        Args:
            img (np.ndarray): Input grayscale image

        Returns:
            np.ndarray: An array of shape (N,2) containing (x,y) coordinates of detected keypoints.
        """
        # Detect keypoints in image (list of keypoint objects)
        key_points = self.detector.detect(img, None)
        # Convert to numpy array of (x,y) coordinates
        key_points = np.array([x.pt for x in key_points], dtype=np.float32)

        return key_points
    
    def featureTracking(self, img_ref, img_cur, pts_ref) -> tuple[np.ndarray, np.ndarray]:
        """
        Tracks features from reference frame to current frame using the Lucas-Kanade Optical Flow algorithm.
        
        Args: 
            img_ref (np.ndarray): Previous grayscale image
            img_cur (np.ndarray): Current grayscale image
            pts_ref (np.ndarray): Coordinates of features in the previous image

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing (valid_pts_cur, valid_pts_ref)
                valid_pts_cur: Successfully tracked points in the current image 
                valid_pts_ref: The corresponding points from the reference image
        """
        # Find new position of points given old
        pts_cur, status, err = cv.calcOpticalFlowPyrLK(img_ref, img_cur, pts_ref, None, winSize=(21,21), criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        status = status.reshape(-1)

        # Filter out lost points
        valid_pts_cur = pts_cur[status == 1]
        valid_pts_ref = pts_ref[status == 1]

        return valid_pts_cur, valid_pts_ref

    def processFrame(self, frame_id) -> tuple[np.ndarray, np.ndarray]:
        """
        Main Loop: Loads an image, tracks features, and draws the point.

        Args:
            frame_id (int): The index of the frame to process
        
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing (current_frame, display_pts)
                    current_frame (np.ndarray): The grayscale image of the current frame for display.
                    pts_cur (np.ndarray): The (x, y) coordinates of the successfully tracked features in the current frame.
        """
        self.current_frame = self.dataset.get_image(frame_id)
        
        # Initialize for frame 0
        if frame_id == 0:
            # Initialize: Just find points
            self.old_frame = self.current_frame
            self.pts_ref = self.featureDetection(self.current_frame)
            return self.current_frame, self.pts_ref

        # Optical Flow Tracking
        self.pts_cur, self.pts_ref = self.featureTracking(self.old_frame, self.current_frame, self.pts_ref)

        # Pose Estimation
        if len(self.pts_cur) > 10:
            E, mask = cv.findEssentialMat(self.pts_cur, self.pts_ref, self.focal_length, self.pp, method=cv.RANSAC)

            _, R, t, mask = cv.recoverPose(E, self.pts_cur, self.pts_ref, focal=self.focal_length, pp=self.pp)

            absolute_scale = self.getAbsoluteScale(frame_id)

            self.curr_t = self.curr_t + absolute_scale * np.dot(self.curr_R, t)
            self.curr_R = np.dot(R, self.curr_R)

        # Replenish Features
        # If the number of tracked points drops below 2000, detect new ones in the current frame
        if len(self.pts_ref) < 2000:
            new_pts = self.featureDetection(self.current_frame)
            self.pts_cur = np.concatenate((self.pts_cur,new_pts), axis=0)
            self.pts_ref = np.concatenate((self.pts_ref, new_pts), axis=0)

        # Update state for next iteration
        self.old_frame = self.current_frame
        self.pts_ref = self.pts_cur

        return self.current_frame, self.pts_cur
    
def visualize(vo_instance, frame, pts, canvas):
    # Visualization
    x_coord = int(vo_instance.curr_t[0,0]) + 500
    z_coord = 500 - int(vo_instance.curr_t[2,0])
    cv.circle(canvas, (x_coord,z_coord), 1, (0,255,0), 1)
    cv.imshow('Trajectory', canvas)
    
    vis = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    for x, y in pts:
        cv.circle(vis, (int(x), int(y)), radius=2, color=(0,255,0), thickness=-1)
    cv.imshow('VO Pipeline', vis)
    cv.waitKey(1)


if __name__ == "__main__":
    vo = VisualOdometry('data', '00')
    traj_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
    total_frames = len(vo.dataset)

    # Test highway scene
    for i in range(1101):
        frame, pts = vo.processFrame(i)
        visualize(vo, frame, pts, traj_canvas)

    traj_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
    vo.curr_R = np.identity(3)
    vo.curr_t = np.zeros((3,1))
    # Test neighborhood scene
    for i in range(1101,total_frames):
        frame, pts = vo.processFrame(i)
        visualize(vo, frame, pts, traj_canvas)