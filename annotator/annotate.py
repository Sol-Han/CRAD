#!/usr/bin/env python3
"""
CRAD Annotation Tool
====================
Camera-radar roll/pitch/height reference labels: the manual annotation
interface used to build the CRAD dataset.

Requires the raw Pohang Canal Dataset sequence directory (calibration/,
stereo/left_images or infrared/images, radar/ -- see ../README.md and
https://sites.google.com/view/pohang-canal-dataset). Not needed if you only
want to use the already-published labels in ../labels/.

[ Usage ]
    python annotate.py --data_dir pohang/pohang03

[ Layout ]
    Left  : camera image + detected water boundary
    Right : normalized raw radar intensity + camera boundary projected onto it
    Sliders: roll / pitch / height, updated live
    Buttons: <Prev / Next> / Save GT / Reset

[ Save format ]
    {save_dir}/gt_rph.json
    {
      "013332": {"roll": 0.05, "pitch": -0.10, "height": 3.20},
      ...
    }
    (roll/pitch in radians here; the published labels/ folder in this repo
    is the same data converted to degrees -- see ../README.md)
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import cv2
from math import pi
from typing import Optional, Tuple

from PIL import Image as PILImage

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

# ── Pohang Canal Dataset calibration loader (standalone, no CROCS dependency) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pohang_calibration import CameraCalibration

# ── SAM3 ──────────────────────────────────────────────────────────────────────
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print('[CRAD Annotator] WARNING: sam3 module not found -- falling back to Canny edges.')


# ============================================================================
# Data loading utilities (standalone reimplementation, no CROCS dependency)
# ============================================================================

def load_images(folder: str):
    paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not paths:
        raise FileNotFoundError(f'No PNGs in {folder}')
    return paths


def load_timestamp_file(timestamp_path: str):
    """Each line: unix_time image_name"""
    timestamps = []
    with open(timestamp_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamps.append((float(parts[0]), parts[1]))
    return timestamps


def extract_image_number(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def find_closest_radar_image(stereo_ts, radar_ts, image_number: str):
    stereo_time = None
    for t, name in stereo_ts:
        if name == image_number:
            stereo_time = t
            break
    if stereo_time is None:
        return None, None
    best, best_diff = None, float('inf')
    for t, name in radar_ts:
        diff = abs(t - stereo_time)
        if diff < best_diff:
            best_diff = diff
            best = name
    return best, best_diff


def load_radar_image_path(radar_dir: str, radar_name: str) -> Optional[str]:
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(radar_dir, f'{radar_name}{ext}')
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(radar_dir, f'{radar_name}.*'))
    return matches[0] if matches else None


# ============================================================================
# Radar display (raw-intensity, no inverse sensor model)
# ============================================================================
#
# CRAD's own evaluation metrics (Projection MAE / mIoU) are computed directly
# from normalized raw radar intensity, not from an inverse sensor model (ISM)
# -- see README.md. This annotation tool matches that convention: it displays
# a simple normalized version of the raw radar image, with no probabilistic
# occlusion/plausibility modeling. (CROCS, the separate follow-on algorithm,
# does use a full ISM internally, but that is not part of CRAD.)

class RadarDisplay:
    """Holds the radar image's display geometry and a simple normalization
    for visualization -- no ISM, no torch dependency."""
    def __init__(self, image_shape=(2048, 2048), radar_range=1655.0):
        self.H, self.W = image_shape
        self.n_r = min(self.H, self.W) // 2
        self.grid_size = radar_range / self.n_r   # meters per pixel (radial)
        self.radar_range = radar_range
        self.cx = (self.W - 1) / 2.0
        self.cy = (self.H - 1) / 2.0

    def process(self, img_np: np.ndarray) -> np.ndarray:
        """Raw grayscale radar image -> normalized intensity (0-1, float32)."""
        img = img_np.astype(np.float32)
        lo, hi = np.percentile(img, [1, 99])
        if hi <= lo:
            lo, hi = img.min(), img.max() + 1e-6
        return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


# ============================================================================
# Water boundary extraction
# ============================================================================

def _sample_boundary_points(boundary_points: np.ndarray,
                             camera_center: Tuple[float, float],
                             num_samples: int,
                             W: int, H: int,
                             margin: int) -> np.ndarray:
    """
    Same sampling logic as the authors' CROCS boundary-point extraction.
    num_samples >= 10: linear interpolation along X; < 10: even rho spacing per side.
    """
    c_x, c_y = camera_center

    # margin / slope filtering
    valid_x = (boundary_points[:, 0] >= margin) & (boundary_points[:, 0] < W - margin)
    valid_y = (boundary_points[:, 1] >= margin) & (boundary_points[:, 1] < H - margin)
    pts = boundary_points[valid_x & valid_y]
    if len(pts) == 0:
        return None

    # sort by X, then filter by slope
    sort_idx = np.argsort(pts[:, 0])
    pts = pts[sort_idx]
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    slopes = np.abs(dy / (dx + 1e-6))
    flat_mask = slopes < 1.5
    target = pts[flat_mask] if flat_mask.sum() >= num_samples else pts

    if len(target) == 0:
        return None

    if num_samples < 10:
        # even rho spacing, left/right side
        d_x = target[:, 0] - c_x
        d_y = target[:, 1] - c_y
        rhos = np.sqrt(d_x**2 + d_y**2)
        left_m = target[:, 0] < c_x
        right_m = ~left_m

        def _sample_side(p, r, n):
            if len(p) == 0:
                return np.empty((0, 2))
            if len(p) <= n:
                return p
            idx_s = np.argsort(r)
            p, r = p[idx_s], r[idx_s]
            targets = np.linspace(r.min(), r.max(), n)
            indices = np.clip(np.searchsorted(r, targets), 0, len(r) - 1)
            return p[np.unique(indices)]

        n_L = num_samples // 2 + num_samples % 2
        n_R = num_samples // 2
        pts_L = _sample_side(target[left_m], rhos[left_m], n_L)
        pts_R = _sample_side(target[right_m], rhos[right_m], n_R)
        merged = []
        for i in range(max(len(pts_L), len(pts_R))):
            if i < len(pts_L):
                merged.append(pts_L[i])
            if i < len(pts_R) and i != 0:
                merged.append(pts_R[i])
        return np.array(merged) if merged else None
    else:
        # linear interpolation along X
        sort_idx2 = np.argsort(target[:, 0])
        target = target[sort_idx2]
        new_x = np.linspace(target[:, 0].min(), target[:, 0].max(), num_samples)
        new_y = np.interp(new_x, target[:, 0], target[:, 1])
        return np.stack((new_x, new_y), axis=1)


def extract_water_boundary_sam3(image_rgb: np.ndarray,
                                 camera_center: Tuple[float, float],
                                 processor,
                                 water_text_prompt: str = 'water',
                                 num_samples: int = 150,
                                 margin: int = 200
                                 ) -> Tuple[Optional[np.ndarray],
                                            Optional[np.ndarray],
                                            Optional[np.ndarray]]:
    """
    Water mask via SAM3 -> Canny edge -> boundary point sampling.

    Same logic as the authors' CROCS boundary-point extraction (mode='seg').

    Returns:
        boundary_points : (N, 2) pixel coordinates, or None
        edges           : Canny edge image, or None
        water_mask      : binary mask, or None
    """
    H, W = image_rgb.shape[:2]
    pil_image = PILImage.fromarray(image_rgb)

    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)

    results = processor.set_text_prompt(state=inference_state,
                                        prompt=water_text_prompt)
    masks = results['masks']
    scores = results['scores']

    if len(scores) == 0:
        return None, None, None

    best_idx = int(np.argmax(scores.cpu().numpy()))
    water_mask = masks[best_idx].squeeze(0).cpu().numpy()

    if water_mask.sum() == 0:
        return None, None, None

    mask_uint8 = (water_mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, edges, water_mask.astype(bool)

    main_contour = max(contours, key=len)
    boundary_points = main_contour.reshape(-1, 2).astype(float)

    sampled = _sample_boundary_points(
        boundary_points, camera_center, num_samples, W, H, margin)

    return sampled, edges, water_mask.astype(bool)


def extract_water_boundary_canny(image_rgb: np.ndarray,
                                  camera_center: Tuple[float, float],
                                  num_samples: int = 150,
                                  margin: int = 80
                                  ) -> Tuple[Optional[np.ndarray], None, None]:
    """Canny-based fallback when SAM3 is unavailable."""
    H, W = image_rgb.shape[:2]
    cy = int(camera_center[1])
    roi_top = max(0, cy - H // 8)
    roi = image_rgb[roi_top:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None

    main_contour = max(contours, key=len)
    pts = main_contour.reshape(-1, 2).astype(float)
    pts[:, 1] += roi_top

    sampled = _sample_boundary_points(
        pts, camera_center, num_samples, W, H, margin)
    return sampled, None, None


# ============================================================================
# Camera -> radar polar projection (same as the authors' CROCS implementation)
# ============================================================================

def project_camera_to_radar(camera_points: np.ndarray,
                             focal_length: float,
                             camera_center: Tuple[float, float],
                             pitch: float, h: float, roll: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Camera pixel coordinates -> ground projection -> radar polar (r[m], theta[rad]).

    Returns:
        r_cam    : (N,) range [m]
        theta_cam: (N,) angle [rad, 0-2pi]
    """
    if camera_points is None or len(camera_points) == 0:
        return np.array([]), np.array([])

    c_x, c_y = camera_center
    dx = camera_points[:, 0] - c_x
    dy = camera_points[:, 1] - c_y
    rho = np.sqrt(dx**2 + dy**2) / focal_length
    theta_orig = np.arctan2(dy, dx)

    sin_a = np.sin(pitch)
    cos_a = np.cos(pitch)
    theta_shifted = theta_orig - roll
    sin_t = np.sin(theta_shifted)
    cos_t = np.cos(theta_shifted)

    den = sin_a + rho * cos_a * sin_t
    valid = den > 1e-6
    if not valid.any():
        return np.array([]), np.array([])

    t_D = h / den[valid]
    rho_v = rho[valid]
    cos_tv = cos_t[valid]
    sin_tv = sin_t[valid]

    u_D = rho_v * t_D * cos_tv
    use_cos = abs(sin_a) < abs(cos_a)
    if use_cos:
        v_D = (h * sin_a - t_D) / cos_a
    else:
        y_D = rho_v * t_D * sin_tv
        v_D = (y_D - h * cos_a) / sin_a

    x_r = u_D
    y_r = -v_D
    fwd = y_r > 0
    x_r, y_r = x_r[fwd], y_r[fwd]

    r_cam = np.sqrt(x_r**2 + y_r**2)
    theta_cam = (np.arctan2(x_r, y_r) + 2 * np.pi) % (2 * np.pi)
    return r_cam, theta_cam


def polar_to_cart_pixels(r: np.ndarray, theta: np.ndarray,
                         cx: float, cy: float,
                         grid_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polar (r[m], theta[rad]) -> radar image pixel coordinates.
    RadarDisplay's convention: src_x = R_pix * sin(theta) + cx
                                  src_y = -R_pix * cos(theta) + cy
    """
    r_pix = r / grid_size
    px = r_pix * np.sin(theta) + cx
    py = -r_pix * np.cos(theta) + cy
    return px, py


# ============================================================================
# Main annotator class
# ============================================================================

class GTAnnotator:
    # -- default slider ranges --
    ROLL_RANGE   = (-np.deg2rad(15), np.deg2rad(15))   # ±15°
    PITCH_RANGE  = (-np.deg2rad(15), np.deg2rad(15))   # ±15°
    HEIGHT_RANGE = (0.5, 10.0)                         # m

    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        data_name = os.path.basename(os.path.normpath(self.data_dir))
        self.save_path = os.path.join(args.save_dir, data_name, 'gt_rph.json')

        os.makedirs(os.path.join(args.save_dir, data_name), exist_ok=True)  

        # -- load GT dict (resume from existing file if present) --
        self.gt_dict = {}
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.gt_dict = json.load(f)
            print(f'[GT Annotator] Loaded existing GT: {self.save_path} '
                  f'({len(self.gt_dict)} entries)')

        # -- camera calibration --
        use_infrared = ('pohang01' in self.data_dir or
                        'pohang05' in self.data_dir)
        self.use_infrared = use_infrared
        mode = 'infrared' if use_infrared else 'stereo'

        self.cal = CameraCalibration()
        self.cal.load_calibration_data(self.data_dir, mode=mode)

        if use_infrared:
            intr = self.cal.infrared_intrinsics
            self.K = self.cal.K_infrared
            left_dir = 'infrared/images'
        else:
            intr = self.cal.left_intrinsics
            self.K = self.cal.K_left
            left_dir = args.left_dir

        self.focal_length = float(intr['focal_length'])
        self.camera_center = (float(intr['cc_x']), float(intr['cc_y']))
        self.dist_coeffs = np.array(intr['distortion_coefficients'])
        self._build_rectify(intr)

        # -- load images / timestamps --
        stereo_folder = os.path.dirname(left_dir)
        self.img_paths = load_images(
            os.path.join(self.data_dir, left_dir))
        self.stereo_ts = load_timestamp_file(
            os.path.join(self.data_dir, stereo_folder, 'timestamp.txt'))
        self.radar_ts = load_timestamp_file(
            os.path.join(self.data_dir, args.radar_dir, 'timestamp.txt'))
        self.radar_img_dir = os.path.join(
            self.data_dir, args.radar_dir, 'images')

        # build the frame-index list after applying offset / interval
        all_idx = list(range(args.offset,
                             min(len(self.img_paths), args.end_idx),
                             args.interval))
        self.frame_indices = all_idx
        self.frame_cursor = 0   # position within frame_indices

        # -- radar display --
        radar_shape = (2048, 2048)
        self.radar_proc = RadarDisplay(
            image_shape=radar_shape, radar_range=args.grid_range)

        # -- SAM3 initialization --
        self.sam3_processor = None
        use_sam3 = SAM3_AVAILABLE and not getattr(args, 'no_sam3', False)
        if use_sam3:
            print('[CRAD Annotator] Loading SAM3 model...')
            sam3_model = build_sam3_image_model()
            self.sam3_processor = Sam3Processor(sam3_model)
            print('[CRAD Annotator] SAM3 loaded.')
        else:
            print('[CRAD Annotator] SAM3 disabled -- using Canny fallback.')

        # -- current frame data (updated on load) --
        self.cam_image_rgb: Optional[np.ndarray] = None
        self.radar_phi: Optional[np.ndarray] = None
        self.boundary_pts: Optional[np.ndarray] = None
        self.water_mask: Optional[np.ndarray] = None   # SAM3 water mask
        self.water_edges: Optional[np.ndarray] = None  # Canny edge on mask
        self.image_number: str = ''

        # -- build the matplotlib window --
        self._build_ui()

        # -- load the first frame --
        self._load_current_frame()

    # -- helper methods --

    def _build_rectify(self, intr):
        """Build the undistortion/rectify maps."""
        w = intr['image_width']
        h = intr['image_height']
        wh = (w, h)
        dist = np.array(intr['distortion_coefficients'])
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, dist, wh, 1, wh)
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K, dist, None, new_K, wh, cv2.CV_16SC2)
        self._rmap1, self._rmap2 = map1, map2
        self._new_K = new_K

    def _rectify(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self._rmap1, self._rmap2, cv2.INTER_LINEAR)

    # -- build UI --

    def _build_ui(self):
        self.fig = plt.figure(figsize=(18, 9))
        try:
            self.fig.canvas.manager.set_window_title('CRAD Annotation Tool')
        except Exception:
            pass

        gs = gridspec.GridSpec(
            3, 2,
            figure=self.fig,
            height_ratios=[7, 1, 1],
            hspace=0.35,
            wspace=0.1
        )

        # -- camera / radar axes --
        self.ax_cam = self.fig.add_subplot(gs[0, 0])
        self.ax_rad = self.fig.add_subplot(gs[0, 1])
        self.ax_cam.set_title('Camera (water boundary)', fontsize=11)
        self.ax_rad.set_title('Radar (raw) + Projection', fontsize=11)
        self.ax_cam.axis('off')
        self.ax_rad.axis('off')

        # -- slider axes --
        ax_roll   = self.fig.add_axes([0.10, 0.30, 0.35, 0.03])
        ax_pitch  = self.fig.add_axes([0.10, 0.24, 0.35, 0.03])
        ax_height = self.fig.add_axes([0.10, 0.18, 0.35, 0.03])

        self.sl_roll = Slider(
            ax_roll, 'Roll  [°]',
            np.rad2deg(self.ROLL_RANGE[0]),
            np.rad2deg(self.ROLL_RANGE[1]),
            valinit=0.0, valstep=0.1, color='steelblue')
        self.sl_pitch = Slider(
            ax_pitch, 'Pitch [°]',
            np.rad2deg(self.PITCH_RANGE[0]),
            np.rad2deg(self.PITCH_RANGE[1]),
            valinit=0.0, valstep=0.1, color='tomato')
        self.sl_height = Slider(
            ax_height, 'Height [m]',
            self.HEIGHT_RANGE[0], self.HEIGHT_RANGE[1],
            valinit=3.0, valstep=0.05, color='forestgreen')

        self.sl_roll.on_changed(self._on_slider)
        self.sl_pitch.on_changed(self._on_slider)
        self.sl_height.on_changed(self._on_slider)

        # -- button axes --
        ax_prev  = self.fig.add_axes([0.55, 0.26, 0.08, 0.05])
        ax_next  = self.fig.add_axes([0.65, 0.26, 0.08, 0.05])
        ax_save  = self.fig.add_axes([0.75, 0.26, 0.10, 0.05])
        ax_reset = self.fig.add_axes([0.87, 0.26, 0.08, 0.05])

        self.btn_prev  = Button(ax_prev,  '◀ Prev',
                                color='0.85', hovercolor='0.70')
        self.btn_next  = Button(ax_next,  'Next ▶',
                                color='0.85', hovercolor='0.70')
        self.btn_save  = Button(ax_save,  '💾 Save GT',
                                color='#d4edda', hovercolor='#a8d5b5')
        self.btn_reset = Button(ax_reset, 'Reset',
                                color='#ffeeba', hovercolor='#ffd966')

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_save.on_clicked(self._on_save)
        self.btn_reset.on_clicked(self._on_reset)

        # -- keyboard shortcuts --
        # left/right: prev/next frame,  s: save,  r: reset
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # -- status text --
        self.status_text = self.fig.text(
            0.55, 0.33, '', ha='left', va='bottom',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # -- frame loading --

    def _load_current_frame(self):
        """Load the camera+radar pair for the current frame_cursor."""
        if not self.frame_indices:
            print('[CRAD Annotator] frame_indices is empty.')
            return

        img_idx = self.frame_indices[self.frame_cursor]
        img_path = self.img_paths[img_idx]
        self.image_number = extract_image_number(img_path)
        print(f'[GT Annotator] Loading frame {self.frame_cursor + 1}/'
              f'{len(self.frame_indices)} : {self.image_number}')

        # -- load & rectify the camera image --
        if self.use_infrared:
            img_gray, new_K = CameraCalibration.decode_infrared_image(
                img_path, self.K, self.dist_coeffs)
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            self.focal_length = float(new_K[0, 0])
            self.camera_center = (float(new_K[0, 2]), float(new_K[1, 2]))
        else:
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = self._rectify(img_rgb)
            self.focal_length = float(self._new_K[0, 0])
            self.camera_center = (float(self._new_K[0, 2]),
                                  float(self._new_K[1, 2]))

        self.cam_image_rgb = img_rgb

        # -- load radar & normalize --
        radar_name, dt = find_closest_radar_image(
            self.stereo_ts, self.radar_ts, self.image_number)
        if radar_name is None:
            print(f'[CRAD Annotator] No matching radar frame: {self.image_number}')
            self.radar_phi = None
        else:
            rp = load_radar_image_path(self.radar_img_dir, radar_name)
            if rp is None:
                print(f'[CRAD Annotator] Radar file not found: {radar_name}')
                self.radar_phi = None
            else:
                raw = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
                self.radar_phi = self.radar_proc.process(raw)

        # -- water boundary extraction --
        margin = 30 if self.use_infrared else 200
        if self.sam3_processor is not None:
            pts, edges, mask = extract_water_boundary_sam3(
                self.cam_image_rgb, self.camera_center,
                self.sam3_processor,
                num_samples=150, margin=margin)
        else:
            pts, edges, mask = extract_water_boundary_canny(
                self.cam_image_rgb, self.camera_center,
                num_samples=150, margin=80)

        self.boundary_pts  = pts
        self.water_edges   = edges
        self.water_mask    = mask

        if pts is None:
            print(f'[CRAD Annotator]  boundary extraction failed: {self.image_number}')

        # -- restore sliders if a saved GT value exists --
        if self.image_number in self.gt_dict:
            entry = self.gt_dict[self.image_number]
            self.sl_roll.set_val(np.rad2deg(entry['roll']))
            self.sl_pitch.set_val(np.rad2deg(entry['pitch']))
            self.sl_height.set_val(entry['height'])
        # (restoring a slider auto-triggers _on_slider, so no extra update call is needed)
        else:
            self._update_display()

    # -- display update --

    def _update_display(self):
        roll   = np.deg2rad(self.sl_roll.val)
        pitch  = np.deg2rad(self.sl_pitch.val)
        height = self.sl_height.val

        self.ax_cam.cla()
        self.ax_rad.cla()
        self.ax_cam.axis('off')
        self.ax_rad.axis('off')

        total = len(self.frame_indices)
        saved_mark = ' ✓' if self.image_number in self.gt_dict else ''
        self.ax_cam.set_title(
            f'Camera  [{self.frame_cursor + 1}/{total}]  '
            f'{self.image_number}{saved_mark}',
            fontsize=10)
        self.ax_rad.set_title(
            f'Radar (raw) + Projection  '
            f'(roll={np.rad2deg(roll):.1f}°  '
            f'pitch={np.rad2deg(pitch):.1f}°  '
            f'h={height:.2f}m)',
            fontsize=10)

        # -- display the camera image --
        if self.cam_image_rgb is not None:
            self.ax_cam.imshow(self.cam_image_rgb)

            # semi-transparent water-mask overlay
            if self.water_mask is not None:
                H_img, W_img = self.cam_image_rgb.shape[:2]
                overlay = np.zeros((H_img, W_img, 4), dtype=np.float32)
                overlay[self.water_mask, 0] = 0.2   # R
                overlay[self.water_mask, 1] = 0.6   # G
                overlay[self.water_mask, 2] = 1.0   # B
                overlay[self.water_mask, 3] = 0.3   # alpha
                self.ax_cam.imshow(overlay)

            # display the Canny edge (mask boundary)
            if self.water_edges is not None:
                edge_rgba = np.zeros(
                    (self.water_edges.shape[0], self.water_edges.shape[1], 4),
                    dtype=np.float32)
                edge_mask = self.water_edges > 0
                edge_rgba[edge_mask] = [1.0, 1.0, 0.0, 0.9]  # yellow
                self.ax_cam.imshow(edge_rgba)

            # scatter of the sampled boundary points
            if self.boundary_pts is not None:
                self.ax_cam.scatter(
                    self.boundary_pts[:, 0], self.boundary_pts[:, 1],
                    c='lime', s=6, alpha=0.9, linewidths=0, label='boundary pts')
            cx, cy = self.camera_center
            self.ax_cam.plot(cx, cy, 'r+', ms=14, mew=2.5, label='cam center')
            self.ax_cam.legend(loc='upper right', fontsize=8, markerscale=1.5)

        # -- display radar --
        if self.radar_phi is not None:
            self.ax_rad.imshow(self.radar_phi, cmap='hot', origin='upper',
                               vmin=0.0, vmax=1.0)

            # -- compute & overlay the projection --
            if self.boundary_pts is not None and len(self.boundary_pts) > 0:
                r_cam, theta_cam = project_camera_to_radar(
                    self.boundary_pts, self.focal_length,
                    self.camera_center, pitch, height, roll)

                if len(r_cam) > 0:
                    px, py = polar_to_cart_pixels(
                        r_cam, theta_cam,
                        self.radar_proc.cx, self.radar_proc.cy,
                        self.radar_proc.grid_size)

                    # keep only points within the image bounds
                    H, W = self.radar_phi.shape
                    in_range = ((px >= 0) & (px < W) &
                                (py >= 0) & (py < H))
                    if in_range.any():
                        self.ax_rad.scatter(
                            px[in_range], py[in_range],
                            c='cyan', s=6, alpha=0.8,
                            linewidths=0, label='cam proj')
                        self.ax_rad.legend(loc='upper right', fontsize=8,
                                           markerscale=2)
        else:
            self.ax_rad.text(0.5, 0.5, 'No radar data',
                             ha='center', va='center',
                             transform=self.ax_rad.transAxes,
                             color='gray', fontsize=14)

        # -- status info text --
        n_proj = 0
        if self.boundary_pts is not None:
            r_t, _ = project_camera_to_radar(
                self.boundary_pts, self.focal_length,
                self.camera_center, pitch, height, roll)
            n_proj = len(r_t)

        saved_str = (f'Saved roll={np.rad2deg(self.gt_dict[self.image_number]["roll"]):.2f}°'
                     if self.image_number in self.gt_dict else 'Not saved yet')
        self.status_text.set_text(
            f'Image : {self.image_number}\n'
            f'Boundary pts : {len(self.boundary_pts) if self.boundary_pts is not None else 0}\n'
            f'Projected pts: {n_proj}\n'
            f'GT status    : {saved_str}')

        self.fig.canvas.draw_idle()

    # -- callback methods --

    def _on_slider(self, _val):
        self._update_display()

    def _on_prev(self, _event):
        if self.frame_cursor > 0:
            self.frame_cursor -= 1
            self._load_current_frame()

    def _on_next(self, _event):
        if self.frame_cursor < len(self.frame_indices) - 1:
            self.frame_cursor += 1
            self._load_current_frame()

    def _on_reset(self, _event):
        self.sl_roll.set_val(0.0)
        self.sl_pitch.set_val(0.0)
        self.sl_height.set_val(3.0)

    def _on_key(self, event):
        if event.key == 'left':
            self._on_prev(event)
        elif event.key == 'right':
            self._on_next(event)
        elif event.key == 's':
            self._on_save(event)
        elif event.key == 'r':
            self._on_reset(event)

    def _on_save(self, _event):
        roll   = np.deg2rad(self.sl_roll.val)
        pitch  = np.deg2rad(self.sl_pitch.val)
        height = self.sl_height.val

        self.gt_dict[self.image_number] = {
            'roll'  : float(roll),
            'pitch' : float(pitch),
            'height': float(height),
        }
        with open(self.save_path, 'w') as f:
            json.dump(self.gt_dict, f, indent=2)

        print(f'[GT Annotator] Saved  {self.image_number}  '
              f'roll={np.rad2deg(roll):.3f}°  '
              f'pitch={np.rad2deg(pitch):.3f}°  '
              f'h={height:.3f}m  →  {self.save_path}')

        # refresh the saved-status display
        self._update_display()

    # -- run --

    def run(self):
        plt.show()


# ============================================================================
# Entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='CRAD Annotation Tool — manual water-boundary/radar projection alignment')
    ap.add_argument('--data_dir',    default='pohang/pohang03',
                    help='data root directory')
    ap.add_argument('--left_dir',    default='stereo/left_images',
                    help='stereo left image folder (relative to data_dir)')
    ap.add_argument('--radar_dir',   default='radar',
                    help='radar folder (relative to data_dir)')
    ap.add_argument('--offset',      type=int, default=None,
                    help='start frame index')
    ap.add_argument('--end_idx',     type=int, default=None,
                    help='end frame index')
    ap.add_argument('--interval',    type=int, default=100,
                    help='frame interval')
    ap.add_argument('--grid_range',  type=float, default=1655.0,
                    help='maximum radar range [m]')
    ap.add_argument('--save_dir',    default='gt_annotations',
                    help='folder to save the GT JSON output')
    ap.add_argument('--no_sam3',     action='store_true',
                    help='disable SAM3 -- use Canny edge fallback')
    args = ap.parse_args()

    SEQ_PARAMS = {
        'pohang00': {'offset': 5800, 'end_idx': 21700},
        'pohang01': {'offset': 6900, 'end_idx': 26400},
        'pohang02': {'offset': 8400, 'end_idx': 22400},
        'pohang03': {'offset': 8000, 'end_idx': 24200},
        'pohang04': {'offset': 6500, 'end_idx': 22000},
        'pohang05': {'offset': 4700, 'end_idx': 21700},
    }
    for _seq_key, _seq_vals in SEQ_PARAMS.items():
        if _seq_key in args.data_dir:
            for _pk, _pv in _seq_vals.items():
                if getattr(args, _pk) is None:
                    setattr(args, _pk, _pv)
            break

    annotator = GTAnnotator(args)
    annotator.run()


if __name__ == '__main__':
    main()
