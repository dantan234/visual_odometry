import cv2 as cv
import numpy as np
from dataset import KITTIDataset

class VisualOdometry:
    """
    A Stereo Visual Odometry pipeline for the KITTI dataset.
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
        self.K = self.dataset.P0[0:3, 0:3]
        self.f = self.K[0,0]
        self.cu = self.K[0,2]
        self.cv = self.K[1,2]
        self.B = abs(self.dataset.P1[0,3]) / self.f

        # Stereo Initialization
        self.stereo = cv.StereoSGBM_create(
            minDisparity = 0,
            numDisparities=64,
            blockSize=7,
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2
        )
        
        # State variables
        self.img_ref = None # Left image from t-1
        self.pts_ref = None # "Reference Points" (Points in t-1)
        self.curr_R = np.identity(3)
        self.curr_t = np.zeros((3,1))
        self.rvec = np.zeros((3,1))
        self.tvec = np.array([[0],[0],[-0.1]], dtype=np.float32)

        # Fast Detector initialization
        self.detector = cv.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
    
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
    
    def get_depth(self, img_left, img_right) -> np.ndarray:
        """
        Computes the depth map (in meters) from a pair of images.

        Args:
            img_left: Grayscale image from camera 0.
            img_right: Grayscale image from camera 1.

        Returns:
            np.ndarray: A 2D float32 depth map where each pixel is depth in meters.
        """
        disp = self.stereo.compute(img_left, img_right)
        disparity = disp.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1

        depth_map = (self.f * self.B) / disparity
        return depth_map
    
    def triangulate(self, pts_2d, depth_map):
        """
        Converts tracked 2D pixels into 3D world coordinates (in meters).

        Args:
            pts_2d: (N,2) array of [u, v] pixel coordinates from the left image.
            depth_map: The float32 dpeth map (in meters) from get_depth.
        """
        pts_3d = []
        valid_idx = []

        for i, (u, v) in enumerate(pts_2d):
            u_int, v_int = int(round(u)), int(round(v))
            
            # Boundary check
            if v_int >= depth_map.shape[0] or u_int >= depth_map.shape[1] or v_int < 0 or u_int < 0:
                continue

            z = depth_map[v_int, u_int]
            
            # Filter out points beyond 80m or less than 5m
            if z > 100.0 or z < 2.0:
                continue

            x = (u - self.cu) * z / self.f
            y = (v - self.cv) * z / self.f

            pts_3d.append([x,y,z])
            valid_idx.append(i)
        return np.array(pts_3d, dtype=np.float32), valid_idx

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
    
    def gridFeatureDetection(self, img, m=10, n=20, qty=10) -> np.ndarray:
        """
        Detects new features in an image using the detector by splitting the image into a m x n grid and picking points in each grid cell.

        Args:
            img (np.ndarray): Input grayscale image
            m (int): number of grid rows
            n (int): number of grid columns
            qty (int): Max points to take in each grid cell
        Returns:
            np.ndarray: An array of shape (N,2) containing (x,y) coordinates of detected keypoints.
        """

        h, w = img.shape
        cell_h, cell_w = h // m, w // n
        feature_points = []

        for i in range(m):
            for j in range(n):
                cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

                key_points = self.detector.detect(cell, None)
                key_points = sorted(key_points, key=lambda x: x.response, reverse=True)[:qty]

                for key_point in key_points:
                    feature_points.append([key_point.pt[0] + j*cell_w, key_point.pt[1] + i*cell_h])
        feature_points = np.array(feature_points, dtype=np.float32)

        return feature_points
    
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
        pts_cur, status, err = cv.calcOpticalFlowPyrLK(img_ref, img_cur, pts_ref, None, winSize=(21,21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        status = status.reshape(-1)

        # Filter out lost points
        valid_pts_cur = pts_cur[status == 1]
        valid_pts_ref = pts_ref[status == 1]

        return valid_pts_cur, valid_pts_ref

    def processFrame(self, frame_id):
        """
        Main Loop: Loads an image, tracks features, and draws the point.

        Args:
            frame_id (int): The index of the frame to process
        """
        img_cur = self.dataset.get_image_left(frame_id)
        
        # Initialize for frame 0
        if frame_id == 0:
            # Initialize: Just find points
            self.img_ref = img_cur
            self.pts_ref = self.gridFeatureDetection(img_cur, qty=20)
            return

        # Optical Flow Tracking
        pts_cur, pts_ref_valid = self.featureTracking(self.img_ref, img_cur, self.pts_ref)

        # Depth Estimation of ref points
        img_ref_right = self.dataset.get_image_right(frame_id - 1)
        depth_ref = self.get_depth(self.img_ref, img_ref_right)

        # Triangulation of ref points
        pts_3d_ref, valid_idx = self.triangulate(pts_ref_valid, depth_ref)
        pts_2d_cur = pts_cur[valid_idx]

        R_rel, t_rel = None, None

        # Pose Estimation (PnP)
        if len(pts_3d_ref) > 10:
            success, rvec_new, tvec_new, inliers = cv.solvePnPRansac(pts_3d_ref, pts_2d_cur, self.K, None, 
                                                    rvec=self.rvec,
                                                    tvec=self.tvec,
                                                    useExtrinsicGuess=True,
                                                    iterationsCount=200, 
                                                    reprojectionError=2.0, 
                                                    confidence=0.99)
            
            if success and inliers is not None:
                R_mat, _ = cv.Rodrigues(rvec_new)
                R_cand = R_mat.T
                t_cand = -R_mat.T @ tvec_new

                # Sanity Check for forward motion
                if t_cand[2] > 0.0 and t_cand[2] < 5.0:
                    R_rel, t_rel = R_cand, t_cand
                    self.rvec, self.tvec = rvec_new, tvec_new

        if R_rel is None or t_rel is None:
            R_mat_prev, _ = cv.Rodrigues((self.rvec))
            R_rel = R_mat_prev.T
            t_rel = -R_mat_prev.T @ self.tvec
        
        # Update Pose
        self.curr_t = self.curr_t + self.curr_R @ t_rel
        self.curr_R = self.curr_R @ R_rel

        # Replenish Features
        # If the number of tracked points drops below 2000, detect new ones in the current frame
        if len(pts_cur) < 4000:
            new_pts = self.gridFeatureDetection(img_cur)
            pts_cur = np.concatenate((pts_cur,new_pts), axis=0)


        # Update state for next iteration
        self.img_ref = img_cur
        self.pts_ref = pts_cur
    
def visualize(vo_instance, canvas):
    # Visualization
    x_coord = int(vo_instance.curr_t[0,0]*0.75) + 500
    z_coord = 500 - int(vo_instance.curr_t[2,0]*0.75)
    cv.circle(canvas, (x_coord,z_coord), 1, (0,255,0), 1)
    cv.imshow('Trajectory', canvas)
    
    
    vis = cv.cvtColor(vo.img_ref, cv.COLOR_GRAY2BGR)
    for x, y in vo.pts_ref:
        cv.circle(vis, (int(x), int(y)), radius=2, color=(0,255,0), thickness=-1)
    cv.imshow('VO Pipeline', vis)
    cv.waitKey(1)


if __name__ == "__main__":
    vo = VisualOdometry('data', '00')
    traj_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
    total_frames = len(vo.dataset)

    # Test highway scene
    for i in range(1101):
        vo.processFrame(i)
        visualize(vo, traj_canvas)

    # Reset canvas
    traj_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
    vo.curr_R = np.identity(3)
    vo.curr_t = np.zeros((3,1))

    # Test neighborhood scene
    for i in range(1101,total_frames):
        vo.processFrame(i)
        visualize(vo, traj_canvas)