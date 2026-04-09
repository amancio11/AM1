"""
camera_controller.py
====================
Simulates drone camera perspectives for building facade inspection.

The camera mimics a DJI-style drone hovering in front of a building facade:
  - Variable altitude (floor level to rooftop)
  - Variable horizontal distance
  - Pitch angle to simulate looking up, level, or slightly down
  - Subtle roll jitter for realism
  - Variable focal length (zoom range)

The controller also computes the camera intrinsic matrix for exporting
to metadata, enabling reprojection in downstream analysis.
"""

import bpy
import math
import random
from typing import Dict, Any, Tuple
from mathutils import Matrix, Vector, Euler


class CameraController:
    """Controls drone camera placement and parameters."""

    CAMERA_NAME = "Drone_Camera"

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.cfg = config["camera"]
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, facade_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Position and configure the scene camera as a drone viewpoint.

        Parameters
        ----------
        facade_metadata : dict
            Must include 'total_height' and 'facade_width'.

        Returns
        -------
        dict
            Camera metadata (position, rotation, intrinsics).
        """
        total_height = facade_metadata.get("total_height", 20.0)
        facade_width = facade_metadata.get("facade_width", 20.0)

        altitude = self.rng.uniform(
            self.cfg["drone_altitude_min"],
            min(self.cfg["drone_altitude_max"], total_height * 1.2),
        )
        distance = self.rng.uniform(
            self.cfg["distance_min"], self.cfg["distance_max"]
        )
        lateral_offset = self.rng.uniform(-facade_width * 0.25, facade_width * 0.25)

        pitch_deg = self.rng.uniform(
            self.cfg["pitch_min"], self.cfg["pitch_max"]
        )
        roll_deg = self.rng.uniform(-self.cfg["roll_max"], self.cfg["roll_max"])

        focal_length = self.rng.uniform(
            self.cfg["focal_length_min"], self.cfg["focal_length_max"]
        )
        sensor_width = self.cfg["sensor_width"]

        cam_x = lateral_offset
        cam_y = -distance            # negative Y = in front of facade (facade is at Y≈0)
        cam_z = altitude

        camera_obj = self._get_or_create_camera()
        camera_obj.location = Vector((cam_x, cam_y, cam_z))

        # Camera points roughly towards the center of the facade
        # Yaw: horizontal angle to point at facade center (X=0)
        target = Vector((0.0, 0.0, total_height / 2.0))
        direction = (target - camera_obj.location).normalized()

        # Base rotation from direction
        yaw = math.atan2(direction.x, direction.y)        # rotation around Z
        # Pitch override from config (allow looking up/down from direction)
        base_pitch = math.asin(direction.z)
        pitch_rad = base_pitch + math.radians(pitch_deg)
        roll_rad = math.radians(roll_deg)

        camera_obj.rotation_euler = Euler(
            (math.pi / 2 + pitch_rad, roll_rad, yaw), "XYZ"
        )

        # Camera intrinsics
        scene = bpy.context.scene
        camera_obj.data.lens = focal_length
        camera_obj.data.sensor_width = sensor_width
        camera_obj.data.clip_start = 0.1
        camera_obj.data.clip_end = 500.0

        scene.camera = camera_obj

        # Render resolution
        render = scene.render
        img_w = render.resolution_x
        img_h = render.resolution_y

        fx = focal_length / sensor_width * img_w
        fy = focal_length / sensor_width * img_w  # square pixels
        cx = img_w / 2.0
        cy = img_h / 2.0

        metadata = {
            "camera_position": [cam_x, cam_y, cam_z],
            "camera_rotation_euler_deg": [
                math.degrees(math.pi / 2 + pitch_rad),
                math.degrees(roll_rad),
                math.degrees(yaw),
            ],
            "focal_length_mm": focal_length,
            "sensor_width_mm": sensor_width,
            "drone_altitude_m": altitude,
            "drone_distance_m": distance,
            "lateral_offset_m": lateral_offset,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": img_w,
                "height": img_h,
            },
        }
        return metadata

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create_camera(self) -> bpy.types.Object:
        """Retrieve existing drone camera or create a new one."""
        if self.CAMERA_NAME in bpy.data.objects:
            return bpy.data.objects[self.CAMERA_NAME]

        cam_data = bpy.data.cameras.new(self.CAMERA_NAME)
        cam_data.type = "PERSP"
        cam_obj = bpy.data.objects.new(self.CAMERA_NAME, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        return cam_obj

    def remove_camera(self) -> None:
        if self.CAMERA_NAME in bpy.data.objects:
            obj = bpy.data.objects[self.CAMERA_NAME]
            bpy.data.objects.remove(obj, do_unlink=True)
