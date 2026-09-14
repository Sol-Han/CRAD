"""
Pohang Canal Dataset camera calibration loader.

Standalone utility (no CROCS dependency) for reading the Pohang Canal
Dataset's calibration/intrinsics.json and calibration/extrinsics.json files.
Extracted as-is from the authors' internal CROCS codebase (core/camera.py)
since it is generic dataset I/O, not part of any CROCS-specific algorithm.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class CameraCalibration:
    """Camera calibration utilities for loading and managing camera parameters"""
    
    def __init__(self):
        self.left_intrinsics = None
        self.right_intrinsics = None
        self.K_left = None
        self.K_right = None
        self.R_left = None
        self.t_left = None
        self.R_right = None
        self.t_right = None
        self.R_radar_to_ahrs = None
        self.t_radar_to_ahrs = None
        self.R_radar_to_camera = None
        self.t_radar_to_camera = None
    
    @staticmethod
    def load_images(folder: str) -> list:
        """Load image paths from folder"""
        import glob
        paths = sorted(glob.glob(os.path.join(folder, '*.png')))
        if len(paths) == 0:
            raise FileNotFoundError(f'No PNGs in {folder}')
        return paths
    
    @staticmethod
    def load_intrinsics(filename: str, camera_name: str = "stereo_left") -> Tuple[Dict, np.ndarray]:
        """Load camera intrinsics from JSON file"""
        with open(filename, 'r') as f:
            intrinsics_data = json.load(f)
        
        camera_intrinsics = intrinsics_data[camera_name]
        K = np.array([
            [camera_intrinsics['focal_length'], 0, camera_intrinsics['cc_x']],
            [0, camera_intrinsics['focal_length'], camera_intrinsics['cc_y']],
            [0, 0, 1]
        ])
        return camera_intrinsics, K
    
    @staticmethod
    def load_extrinsics(data_dir: str, extrinsics_file: str = 'calibration/extrinsics.json') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load extrinsic parameters from stereo cameras to AHRS"""
        with open(os.path.join(data_dir, extrinsics_file), 'r') as f:
            cfg = json.load(f)
        
        # Load stereo_left camera extrinsics (stereo_left to AHRS)
        stereo_left = cfg['stereo_left']
        R_left_to_ahrs = CameraCalibration.quaternion_to_rotation_matrix(stereo_left['quaternion'])
        t_left_to_ahrs = np.array(stereo_left['translation'], dtype=np.float64)
        
        # Load stereo_right camera extrinsics (stereo_right to AHRS)
        stereo_right = cfg['stereo_right']
        R_right_to_ahrs = CameraCalibration.quaternion_to_rotation_matrix(stereo_right['quaternion'])
        t_right_to_ahrs = np.array(stereo_right['translation'], dtype=np.float64)
        
        # Load infrared camera extrinsics (infrared to AHRS)
        infrared = cfg['infrared']
        R_infrared_to_ahrs = CameraCalibration.quaternion_to_rotation_matrix(infrared['quaternion'])
        t_infrared_to_ahrs = np.array(infrared['translation'], dtype=np.float64)
        
        return R_left_to_ahrs, t_left_to_ahrs, R_right_to_ahrs, t_right_to_ahrs, R_infrared_to_ahrs, t_infrared_to_ahrs
    
    @staticmethod
    def load_gps_to_ahrs_extrinsics(data_dir: str, extrinsics_file: str = 'calibration/extrinsics.json') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load GPS to AHRS extrinsic parameters
        
        Returns:
            R_gps_to_ahrs: Rotation from GPS to AHRS (3, 3)
            t_gps_to_ahrs: Translation from GPS to AHRS (3,)
        """
        with open(os.path.join(data_dir, extrinsics_file), 'r') as f:
            cfg = json.load(f)
        
        # Load GPS to AHRS extrinsics
        if 'gps' in cfg:
            gps = cfg['gps']
            R_gps_to_ahrs = CameraCalibration.quaternion_to_rotation_matrix(gps['quaternion'])
            t_gps_to_ahrs = np.array(gps['translation'], dtype=np.float64)
            return R_gps_to_ahrs, t_gps_to_ahrs
        else:
            # If GPS extrinsics not found, return identity transformation
            print("Warning: GPS to AHRS extrinsics not found in extrinsics.json, using identity transformation")
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    
    @staticmethod
    def load_radar_to_ahrs_extrinsics(data_dir: str, radar_high_file: str = 'calibration/extrinsics.json') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load radar to AHRS extrinsic parameters from radar_high.json
        
        Returns:
            R_radar_to_ahrs: Rotation from radar to AHRS (3, 3)
            t_radar_to_ahrs: Translation from radar to AHRS (3,)
        """
        radar_high_path = os.path.join(data_dir, radar_high_file)
        if not os.path.exists(radar_high_path):
            print(f"Warning: radar_high file not found at {radar_high_path}")
            return None, None
        
        with open(radar_high_path, 'r') as f:
            cfg = json.load(f)
        
        # Load radar to AHRS extrinsics
        if 'radar_high' in cfg:
            radar = cfg['radar_high']
            R_radar_to_ahrs = CameraCalibration.quaternion_to_rotation_matrix(radar['quaternion'])
            t_radar_to_ahrs = np.array(radar['translation'], dtype=np.float64)
            return R_radar_to_ahrs, t_radar_to_ahrs
        elif 'ahrs_radar' in cfg:
            # Alternative key name
            radar = cfg['ahrs_radar']
            R_radar_to_ahrs = CameraCalibration.quaternion_to_rotation_matrix(radar['quaternion'])
            t_radar_to_ahrs = np.array(radar['translation'], dtype=np.float64)
            return R_radar_to_ahrs, t_radar_to_ahrs
        else:
            print("Warning: radar to AHRS extrinsics not found in radar_high.json")
            return None, None

    @staticmethod
    def decode_infrared_image(image_path: str, K: np.ndarray, distortion_coeffs: np.ndarray) -> np.ndarray:
        """Decode 16-bit PNG infrared image with 14-bit thermal data and undistort"""
        # Load 16-bit PNG image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_array = np.array(img, dtype=np.uint16)
        image_size = (img.shape[1], img.shape[0])
        
        # Convert to temperature (celsius)
        temperature = 0.04 * img_array - 273.15
        
        # Normalize temperature to 0-255 range for visualization
        # Adjust temperature range for better visualization
        temp_min = np.percentile(temperature, 5)  # 5th percentile
        temp_max = np.percentile(temperature, 95)  # 95th percentile
        
        # Clip temperature to range
        temperature_clipped = np.clip(temperature, temp_min, temp_max)
        
        # Normalize to 0-255
        normalized = ((temperature_clipped - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        
        # Undistort the image if camera parameters are provided
        if K is not None and distortion_coeffs is not None:
            # Get optimal new camera matrix
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, distortion_coeffs, image_size, 1, image_size)
            
            # Undistort the image
            undistorted = cv2.undistort(normalized, K, distortion_coeffs, None, new_K)
            
            # Crop the image to remove black borders if needed
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
            
            return undistorted, new_K
        
        return normalized
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: list) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix"""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        return R
    
    @staticmethod
    def image_rectify(image: np.ndarray, camera_intrinsics: Dict, K: np.ndarray) -> np.ndarray:
        """Apply image rectification"""
        image_width = camera_intrinsics['image_width']
        image_height = camera_intrinsics['image_height']
        wh = (image_width, image_height)
        dist_coeff = np.array(camera_intrinsics["distortion_coefficients"])

        optimal_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeff, wh, 1, wh)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, None, optimal_camera_matrix, wh, cv2.CV_16SC2)
        rectified_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

        return rectified_image, optimal_camera_matrix
    
    def create_rectify_function(self, camera_intrinsics: Dict, K: np.ndarray) -> callable:
        """Create a rectification function with pre-bound camera_intrinsics and K"""
        def rectify_func(image: np.ndarray) -> np.ndarray:
            return self.image_rectify(image, camera_intrinsics, K)
        return rectify_func
    
    def load_calibration_data(self, data_dir: str, intrinsics_file: str = 'calibration/intrinsics.json', 
                            extrinsics_file: str = 'calibration/extrinsics.json', mode: str = 'stereo') -> None:
        """Load all camera calibration data"""
        intrinsics_path = os.path.join(data_dir, intrinsics_file)
        
        # Load intrinsics
        self.left_intrinsics, self.K_left = self.load_intrinsics(intrinsics_path, "stereo_left")
        self.right_intrinsics, self.K_right = self.load_intrinsics(intrinsics_path, "stereo_right")
        self.infrared_intrinsics, self.K_infrared = self.load_intrinsics(intrinsics_path, "infrared")
        
        # Load extrinsics
        self.R_left, self.t_left, self.R_right, self.t_right, self.R_infrared_to_ahrs, self.t_infrared_to_ahrs = self.load_extrinsics(data_dir, extrinsics_file)
        
        # Load GPS to AHRS extrinsics
        self.R_gps_to_ahrs, self.t_gps_to_ahrs = self.load_gps_to_ahrs_extrinsics(data_dir, extrinsics_file)
        
        # Load radar to AHRS extrinsics
        self.R_radar_to_ahrs, self.t_radar_to_ahrs = self.load_radar_to_ahrs_extrinsics(data_dir, extrinsics_file)
        
        # Compute radar to camera transformation
        # Radar -> AHRS -> Camera (stereo_left)
        if mode == 'stereo' and self.R_radar_to_ahrs is not None and self.R_left is not None:
            # R_radar_to_camera = R_left_to_ahrs^T @ R_radar_to_ahrs
            # t_radar_to_camera = R_left_to_ahrs^T @ (t_radar_to_ahrs - t_left)
            self.R_radar_to_camera = self.R_left.T @ self.R_radar_to_ahrs
            self.t_radar_to_camera = self.R_left.T @ (self.t_radar_to_ahrs - self.t_left)
        elif mode == 'infrared' and self.R_radar_to_ahrs is not None and self.R_infrared_to_ahrs is not None:
            # R_radar_to_camera = R_infrared_to_ahrs^T @ R_radar_to_ahrs
            # t_radar_to_camera = R_infrared_to_ahrs^T @ (t_radar_to_ahrs - t_infrared_to_ahrs)
            self.R_radar_to_camera = self.R_infrared_to_ahrs.T @ self.R_radar_to_ahrs
            self.t_radar_to_camera = self.R_infrared_to_ahrs.T @ (self.t_radar_to_ahrs - self.t_infrared_to_ahrs)
        else:
            self.R_radar_to_camera = None
            self.t_radar_to_camera = None
    
    def get_stereo_baseline(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get stereo baseline parameters"""
        R_LR = self.R_left.T @ self.R_right  # Left to right rotation
        t_LR = self.R_left.T @ (self.t_right - self.t_left)  # Left to right translation
        return R_LR, t_LR
    
    def get_camera_height(self) -> float:
        """Get estimated camera height"""
        return -(self.t_left[-1] + self.t_right[-1]) / 2.0 + 1.5
