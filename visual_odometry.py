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
        self.prev_frame = None # Left image from t-1
        self.prev_pts = None # "Reference Points" (Points in t-1)
        self.prev_depth = None # Depth map from t-1
        self.global_R = np.identity(3)
        self.global_t = np.zeros((3,1))

        # Initialize Pnp Extrinsics
        self.rvec = np.zeros((3,1))
        self.tvec = np.array([[0],[0],[-0.1]], dtype=np.float32)

        # Fast Detector initialization
        self.detector = cv.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
    
    # def getAbsoluteScale(self, frame_id) -> float:
    #     """
    #     Calculates the Euclidean distance between the current and previous ground truth poses to provide the real-world scale.
        
    #     Args:
    #         frame_id (int): The index of the current frame

    #     Returns:
    #         float: The absolute distance (scale) moved between the previous frame and current frame.
    #     """
    #     curr_pose = self.dataset.poses[frame_id]
    #     x = curr_pose[0,3]
    #     y = curr_pose[1,3]
    #     z = curr_pose[2,3]
    #     prev_pose = self.dataset.poses[frame_id-1]
    #     x_prev = prev_pose[0,3]
    #     y_prev = prev_pose[1,3]
    #     z_prev = prev_pose[2,3]

    #     if frame_id == 0:
    #         return 0.0
    #     else:
    #         scale = np.sqrt((x-x_prev)**2 + (y-y_prev)**2 + (z-z_prev)**2)
    #     return scale
    
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
        Converts tracked 2D pixels into 3D world coordinates (in meters) given depth map.

        Args:
            pts_2d: (N,2) array of [u, v] pixel coordinates from the left image.
            depth_map: The float32 dpeth map (in meters) from get_depth.

        Returns:
            (np.ndarray, List[int]): Returns a tuple containing:
                pts_3d (np.ndarray): A (N,3) array of 3D coordinates [x,y,z]
                valid_idx (List[int]): List of indices of the 2D points that had valid depths

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
            # if z > 100.0 or z < 2.0:
            #     continue

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
    
    def processFrame(self, frame_id) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads an image, tracks features, and estimates pose. Returns current global translation and rotation matrix.

        Args:
            frame_id (int): The index of the frame to process

        Returns:
            Tuple containing:
                global_t (np.ndarray): (3,1) Current position in world coordinates (meters)
                global_R (np.ndarray): (3,3) Current orientation matrix in world coordinates
                t_rel (np.ndarray): (3,1) Instantaneous translation vector in meters
                R_Rel (np.ndarray): (3,3) Instantaneous rotation matrix
                curr_depth_map (np.ndarray): (H,W) Dense depth map for current frame in meters
        """
        curr_frame = self.dataset.get_image_left(frame_id)
        curr_frame_right = self.dataset.get_image_right(frame_id)
        curr_depth_map = self.get_depth(curr_frame, curr_frame_right)
        
        # Initialize for frame 0
        if frame_id == 0 or frame_id == 1101:
            # Initialize: Just find points
            self.prev_frame = curr_frame
            self.prev_pts = self.gridFeatureDetection(curr_frame, qty=10)
            self.prev_depth = curr_depth_map
            return self.global_t, self.global_R, np.zeros((3,1)), np.identity(3), curr_depth_map

        # Optical Flow Tracking
        curr_pts, prev_pts = self.featureTracking(self.prev_frame, curr_frame, self.prev_pts)

        # Triangulation of points in previous frame
        pts_3d_prev, valid_idx = self.triangulate(prev_pts, self.prev_depth)
        pts_2d_curr = curr_pts[valid_idx]

        # Initially assume motion is the same as last frame
        R_mat_prev, _ = cv.Rodrigues(self.rvec)
        R_rel, t_rel = R_mat_prev.T, -R_mat_prev.T @ self.tvec

        # Pose Estimation RANSAC (PnP)
        if len(pts_3d_prev) > 10:
            success, rvec, tvec, inliers = cv.solvePnPRansac(pts_3d_prev, pts_2d_curr, self.K, None, 
                                                    rvec=self.rvec,
                                                    tvec=self.tvec,
                                                    useExtrinsicGuess=True,
                                                    iterationsCount=200, 
                                                    reprojectionError=2.0, 
                                                    confidence=0.99)
            
            # If RANSAC finds a solution, use that relative motion vector
            if success and inliers is not None:
                R_mat, _ = cv.Rodrigues(rvec)
                R_cand = R_mat.T
                t_cand = -R_mat.T @ tvec

                # Sanity Check for forward motion
                if t_cand[2] > 0.0 and t_cand[2] < 5.0:
                    R_rel, t_rel = R_cand, t_cand
                    self.rvec, self.tvec = rvec, tvec
        
        # Update Global Pose
        self.global_t = self.global_t + self.global_R @ t_rel
        self.global_R = self.global_R @ R_rel

        # Replenish Features
        # If the number of tracked points drops below 2000, detect new ones in the current frame
        if len(curr_pts) < 2000:
            new_pts = self.gridFeatureDetection(curr_frame)
            curr_pts = np.concatenate((curr_pts,new_pts), axis=0)

        # Update state for next iteration
        self.prev_frame = curr_frame
        self.prev_pts = curr_pts

        return self.global_t, self.global_R, t_rel, R_rel, curr_depth_map

def draw_trajectory(t_vec, canvas, color=(0, 255, 0), r_mat=None):
    draw_scale = 0.75
    x_coord = int(t_vec[0,0]*draw_scale) + 500
    z_coord = 500 - int(t_vec[2,0]*draw_scale)
    cv.circle(canvas, (x_coord,z_coord), 1, color, 1)
    cv.imshow('Trajectory', canvas)

def draw_feature_feed(vo_instance):
    vis = cv.cvtColor(vo_instance.prev_frame, cv.COLOR_GRAY2BGR)
    for x, y in vo_instance.prev_pts:
        cv.circle(vis, (int(x), int(y)), radius=2, color=(0,255,0), thickness=-1)
    cv.imshow('Left Camera Feed', vis)

if __name__ == "__main__":
    vo = VisualOdometry('data', '00')
    traj_canvas = np.zeros((1200, 2000, 3), dtype=np.uint8)
    total_frames = len(vo.dataset)

    # Test sequence 00
    for i in range(total_frames):
        t_vec, r_mat, t_rel, R_Rel, depth_map = vo.processFrame(i)
        gt_pose = vo.dataset.poses[i]
        gt_t_vec = gt_pose[:,3].reshape(3,1)

        draw_trajectory(t_vec, traj_canvas, color=(0,255,0))
        draw_trajectory(gt_t_vec, traj_canvas, color=(0,0,255))
        draw_feature_feed(vo)
        cv.waitKey(1)

    # # Reset canvas
    # traj_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # vo.curr_R = np.identity(3)
    # vo.curr_t = np.zeros((3,1))
    # vo.rvec = np.zeros((3,1))
    # vo.tvec = np.array([[0],[0],[-0.1]], dtype=np.float32)

    # # Test neighborhood scene
    # for i in range(1101,total_frames):
    #     t_vec, r_mat = vo.processFrame(i)
    #     gt_pose = vo.dataset.poses[i]
    #     gt_t_vec = gt_pose[:,3].reshape(3,1)

    #     draw_trajectory(t_vec, traj_canvas, color=(0,255,0))
    #     draw_trajectory(gt_t_vec, traj_canvas, color=(0,0,255))
    #     draw_feature_feed(vo)
    #     cv.waitKey(1)