"""
Kinematics module for coordinate conversion and inverse kinematics.
Converts pixel coordinates to table coordinates and generates servo commands.
Optimized for landscape view () setup.
"""
import json
import os

# Calibration parameters (can be saved/loaded from file)
CALIBRATION_FILE = "calibration.json"

# Default calibration for bird's eye view
DEFAULT_CALIBRATION = {
    "origin_px": [320, 240],  # Pixel coordinates of table origin (center)
    "scale_x": 0.0015,        # Meters per pixel in X direction
    "scale_y": 0.0015,        # Meters per pixel in Y direction
    "flip_x": False,           # Flip X axis (for mirrored cameras)
    "flip_y": True,            # Flip Y axis (camera Y is inverted)
    "table_width_m": 0.60,     # Table width in meters (for reference)
    "table_height_m": 0.40,    # Table height in meters (for reference)
    "arm_base_x": 0.0,         # Robot arm base X position in table coordinates
    "arm_base_y": 0.0,         # Robot arm base Y position in table coordinates
    # Arm geometry (4-DOF)
    "base_height_m": 0.0612,   # 61.2mm - from table to base
    "shoulder_height_m": 0.095, # 95mm - from table to shoulder joint
    "upper_arm_length_m": 0.12, # 12cm - shoulder to elbow
    "lower_arm_length_m": 0.09, # 9cm - elbow to wrist
    "hand_length_m": 0.06,     # 6cm - wrist to grabbing point (hand extends downward)
}

_calibration = None


def load_calibration():
    """Load calibration from file or use defaults."""
    global _calibration
    if _calibration is not None:
        return _calibration
    
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                _calibration = json.load(f)
                print(f"Loaded calibration from {CALIBRATION_FILE}")
                return _calibration
        except Exception as e:
            print(f"Error loading calibration: {e}, using defaults")
    
    _calibration = DEFAULT_CALIBRATION.copy()
    return _calibration


def save_calibration(calibration=None):
    """Save calibration to file."""
    if calibration is None:
        calibration = _calibration or load_calibration()
    
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"Calibration saved to {CALIBRATION_FILE}")
        return True
    except Exception as e:
        print(f"Error saving calibration: {e}")
        return False

landscapeConfig ={
    "frame_width": 640,
    "frame_height": 480,
    "x_scale":0.0008,   # Meters per pixel in X direction
    "fixed_y":0.2,      # Fixed Y coordinate for grabbing point (0.2m forward from arm base)
    "fixed_z":0.02      # Fixed Z coordinate for grabbing point (0.02m above table)
}

def px_to_table_landscape(cx, cy, ultrasonic_y = None, config=None):
    """
    Convert landscape camera pixel coordinates to world coordinates.
    
    cx, cy          : YOLO bounding box centre pixels
    ultrasonic_y    : optional real depth reading in metres from sensor
                      if None, uses fixed_y from config
    """

    if config is None:
        config = landscapeConfig
    
    # Convert pixel X to meters (right is positive)
    x = (cx - config["frame_width"] / 2) * config["x_scale"]
    
    # Use ultrasonic reading for Y if available, otherwise use fixed Y
    y = ultrasonic_y if ultrasonic_y is not None else config["fixed_y"]
    
    # Z is fixed height above table
    z = config["fixed_z"]
    
    return x, y, z


def table_to_px(x, y, calibration=None):
    """
    Convert table coordinates (meters) to pixel coordinates.
    Inverse of px_to_table.
    
    Args:
        x, y: Table coordinates in meters
        calibration: Calibration dict (uses loaded calibration if None)
    
    Returns:
        tuple: (cx, cy) pixel coordinates
    """
    if calibration is None:
        calibration = load_calibration()
    
    origin_px = calibration["origin_px"]
    scale_x = calibration["scale_x"]
    scale_y = calibration["scale_y"]
    flip_x = calibration.get("flip_x", False)
    flip_y = calibration.get("flip_y", True)
    
    # Apply flips
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    
    # Convert meters to pixel offset
    cx = origin_px[0] + (x / scale_x)
    cy = origin_px[1] + (y / scale_y)
    
    return int(cx), int(cy)


def calculate_arm_angles(x, y, z=0.02, calibration=None):
    """
    Calculate robot arm joint angles for a given target position.
    
    This is IK calculation for a 4-DOF arm:
    - Base rotation (yaw) - D5 (MG996R 180°)
    - Shoulder pitch - D18 (MG996R 180°)
    - Elbow pitch - D22 (MG996R 180°)
    - Wrist pitch (x-axis rotation, up/down) - D19 (9g servo 180°)
    - No gripper (hand is locked in position)
    
    Args:
        x: X position in meters (right is positive) - grabbing point position
        y: Y position in meters (forward is positive) - grabbing point position
        z: Z position in meters (height above table, default: 0.02) - grabbing point position
        calibration: Calibration dict (uses loaded calibration if None)
    
    Returns:
        dict: Joint angles in degrees
    
    Note:
        The input (x, y, z) is the desired grabbing point position.
        Since the hand extends downward from the wrist joint, the IK calculates
        the wrist position at (x, y, z + hand_length).
    """
    if calibration is None:
        calibration = load_calibration()
    
    import math
    
    # Get arm base position
    arm_base_x = calibration.get("arm_base_x", 0.0)
    arm_base_y = calibration.get("arm_base_y", 0.0)
    
    # Get arm geometry from calibration or use defaults
    base_height = calibration.get("base_height_m", 0.0612)  # 61.2mm
    shoulder_height = calibration.get("shoulder_height_m", 0.095)  # 95mm
    upper_arm_length = calibration.get("upper_arm_length_m", 0.12)  # 12cm
    lower_arm_length = calibration.get("lower_arm_length_m", 0.09)  # 9cm
    hand_length = calibration.get("hand_length_m", 0.06)  # 6cm - hand extends downward
    
    # Adjust target position: hand extends downward from wrist
    # If we want grabbing point at z, wrist needs to be at z + hand_length
    wrist_z = z + hand_length
    
    # Calculate relative position from arm base
    rel_x = x - arm_base_x
    rel_y = y - arm_base_y
    
    # Calculate base rotation (yaw) - angle to target in XY plane
    base_angle_deg = math.degrees(math.atan2(rel_y, rel_x))
    
    # Calculate distance in XY plane from base
    xy_dist = math.sqrt(rel_x**2 + rel_y**2)
    
    # Calculate target distance from shoulder joint
    # Shoulder is at height = base_height + shoulder_height
    shoulder_z = base_height + shoulder_height
    # Target is the wrist position (grabbing point + hand length upward)
    target_z_relative = wrist_z - shoulder_z  # Height relative to shoulder
    
    # Calculate distance from shoulder to wrist in 3D
    target_dist = math.sqrt(xy_dist**2 + target_z_relative**2)
    
    # Check if target is reachable
    # Max reach includes upper arm + lower arm (hand length is already accounted in wrist_z)
    max_reach = upper_arm_length + lower_arm_length
    if target_dist > max_reach:
        print(f"Warning: Target at {target_dist:.3f}m exceeds max reach {max_reach:.3f}m")
        # Scale down to max reach
        scale = max_reach / target_dist
        xy_dist *= scale
        target_z_relative *= scale
        target_dist = max_reach
    
    # Calculate angle from shoulder to target in vertical plane
    # Angle from horizontal (0 = horizontal forward, 90 = straight up)
    target_angle_rad = math.atan2(target_z_relative, xy_dist)
    target_angle_deg = math.degrees(target_angle_rad)
    
    # Use law of cosines for 2-link arm (shoulder-elbow-wrist)
    # Calculate elbow angle first
    cos_elbow = (upper_arm_length**2 + lower_arm_length**2 - target_dist**2) / (2 * upper_arm_length * lower_arm_length)
    cos_elbow = max(-1, min(1, cos_elbow))  # Clamp to valid range
    elbow_angle_deg = math.degrees(math.acos(cos_elbow))
    
    # Calculate shoulder angle
    # Use law of cosines to find angle between upper arm and target direction
    # cos(angle_between) = (upper^2 + target^2 - lower^2) / (2 * upper * target)
    cos_angle_between = (upper_arm_length**2 + target_dist**2 - lower_arm_length**2) / (2 * upper_arm_length * target_dist)
    cos_angle_between = max(-1, min(1, cos_angle_between))
    angle_between_deg = math.degrees(math.acos(cos_angle_between))
    
    # Shoulder angle = target angle - angle between upper arm and target
    # This gives us the angle the upper arm makes with horizontal
    shoulder_angle_deg = target_angle_deg - angle_between_deg
    
    # Wrist angle (x-axis rotation, up/down)
    # Keep wrist perpendicular to ground for grabbing
    # Wrist compensates for shoulder + elbow to keep hand level
    # Total arm angle = shoulder + elbow, wrist keeps hand vertical
    total_arm_angle = shoulder_angle_deg + elbow_angle_deg
    wrist_angle_deg = 90 - total_arm_angle
    
    return {
        "base_deg": base_angle_deg,
        "shoulder_deg": shoulder_angle_deg,
        "elbow_deg": elbow_angle_deg,
        "wrist_deg": wrist_angle_deg,
    }


def fake_ik_to_us(x, y, z=0.02, calibration=None):
    """
    Convert table coordinates to servo microsecond values.
    
    This maps joint angles to servo positions for 4-DOF arm:
    - Base (D5): MG996R 180°
    - Shoulder (D18): MG996R 180°
    - Elbow (D22): MG996R 180°
    - Wrist (D19): 9g servo 180° (x-axis rotation, up/down)
    - No gripper (hand is locked)
    
    Args:
        x: X position in meters (right is positive)
        y: Y position in meters (forward is positive)
        z: Z position in meters (height above table, default: 0.02)
        calibration: Calibration dict (uses loaded calibration if None)
    
    Returns:
        list: [base_us, shoulder_us, elbow_us, wrist_us]
              Servo microsecond values (clamped to 900-2100 range)
    """
    angles = calculate_arm_angles(x, y, z, calibration)
    
    # Map angles to servo microseconds
    # MG996R: 1500 = center, 900 = -90deg, 2100 = +90deg
    # ±90deg = ±600us from center
    # Formula: us = 1500 + (angle_deg / 90) * 600
    
    base_us = 1500 + int(angles["base_deg"] * 600 / 90)
    shoulder_us = 1500 + int(angles["shoulder_deg"] * 600 / 90)
    elbow_us = 1500 + int(angles["elbow_deg"] * 600 / 90)
    wrist_us = 1500 + int(angles["wrist_deg"] * 600 / 90)
    
    clamp = lambda u: max(900, min(2100, u))
    return list(map(clamp, [base_us, shoulder_us, elbow_us, wrist_us]))


def get_arm_orientation_info(x, y, z=0.02, calibration=None):
    """
    Get detailed orientation information for debugging.
    
    Returns:
        dict: Complete orientation info including angles and servo values
    """
    angles = calculate_arm_angles(x, y, z, calibration)
    servo_us = fake_ik_to_us(x, y, z, calibration)
    
    return {
        "position_m": {"x": x, "y": y, "z": z},
        "angles_deg": angles,
        "servo_us": servo_us,
        "servo_names": ["base", "shoulder", "elbow", "wrist"],
        "gpio_pins": {"base": 5, "shoulder": 18, "elbow": 22, "wrist": 19}
    }

