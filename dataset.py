import os
import cv2 as cv
import numpy as np

class KITTIDataset:
    """
    Data loader for KITTI Visual Odometry dataset
    """
    def __init__(self, data_path, sequence='00'):
        """
        Initializes paths and loads data.

        Args: 
            data_path (str): Path to the root 'data' folder.
            sequence (str): Sequence number ie. '00', '01'
        """
        self.data_path = data_path
        self.sequence_path = os.path.join(self.data_path, 'sequences', sequence)
        self.pose_path = os.path.join(self.data_path, 'poses', f'{sequence}.txt')

        self.image_path = os.path.join(self.sequence_path, 'image_0')
        self.calib_path = os.path.join(self.sequence_path, 'calib.txt')

        self.image_files = sorted(os.listdir(self.image_path))
        self.calib_matrix = self.load_calib()
        self.poses = self.load_poses()

    def __len__(self):
        """
        Returns the length of images_files, which is the total number of images in the datapath.
        """
        return len(self.image_files)

    def load_calib(self) -> np.ndarray:
        """
        Parses the calibration file to extract the left camera projection matrix (P0)

        Returns:
            np.ndarray: (3, 4) projection matrix
        """
        with open(self.calib_path, 'r') as f:
            lines = f.readlines()
        line_p0 = lines[0]
        p0_data = line_p0.split()[1:]
        calib_array_p0 = np.array(p0_data, dtype=np.float32)
        calib_array_p0 = calib_array_p0.reshape(3,4)
        return calib_array_p0
        
    def load_poses(self) -> np.ndarray:
        """
        Parses the ground truth pose text file
        """
        with open(self.pose_path, 'r') as f:
            lines = f.readlines()
        poses = []
        for line in lines:
            pose = [float(x) for x in line.split()]
            pose_matrix = np.array(pose).reshape(3,4)
            poses.append(pose_matrix)

        poses = np.array(poses)
        return poses
    
    def get_image(self, index) -> np.ndarray:
        """
        Loads the grayscale image at the specified frame index
        
        Args:
            index (int): frame index

        Returns:
            np.ndarray: The image array 
        """
        # Load and return grayscale image at the given index
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_path, img_name)
        image = cv.imread(img_path, 0)
        return image

if __name__ == "__main__":
    print("hello world")
    kitti = KITTIDataset(data_path='data')
    img = kitti.get_image(0)
    print(f"Image Loaded. Shape: {img.shape}")
    
    print(f"Calibration Matrix:\n{kitti.calib_matrix}")
    
    print(f"First Pose:\n{kitti.poses[0]}")

    print(f"Length of dataset: {len(kitti)}")
    
    cv.imshow("Test", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
