"""
dirt_simulator.py
=================
Simulates dirt, grime, dust, and water stains on glass surfaces.

Each glass pane receives an individual dirt texture parameterised by:
  - Pattern type: Perlin noise, Voronoi cells, vertical streaks,
                  dust spots, water stains
  - Density (overall amount of dirt)
  - Texture scale and distortion
  - Color tint (brown grime vs grey dust)

The simulator also generates a ground-truth dirt intensity map as a
NumPy array that is saved alongside each render as a PNG dirt map.
"""

import bpy
import random
import math
import numpy as np
from typing import Dict, Any, List, Tuple


class DirtSimulator:
    """
    Applies procedural dirt patterns to glass materials and generates
    pixel-accurate ground-truth dirt maps.
    """

    PATTERN_TYPES = ["perlin", "voronoi", "streaks", "dust_spots", "water_stains"]

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.cfg = config["dirt"]
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)  # for numpy operations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def randomize_dirt(
        self, glass_objects: List[bpy.types.Object]
    ) -> Dict[str, float]:
        """
        Assign random dirt parameters to each glass object's material.

        Returns
        -------
        dict
            Maps glass object name → dirt density (float [0, 1]).
        """
        dirt_map: Dict[str, float] = {}

        for glass_obj in glass_objects:
            if not self.cfg.get("enabled", True):
                dirt_map[glass_obj.name] = 0.0
                continue

            density = self.rng.uniform(
                self.cfg["density_min"], self.cfg["density_max"]
            )
            pattern = self.rng.choice(self.cfg["pattern_types"])
            scale = self.rng.uniform(
                self.cfg["texture_scale_min"], self.cfg["texture_scale_max"]
            )

            self._apply_dirt_to_material(glass_obj, density, pattern, scale)
            dirt_map[glass_obj.name] = density

        return dirt_map

    def generate_ground_truth_dirt_map(
        self,
        glass_objects: List[bpy.types.Object],
        image_width: int,
        image_height: int,
        scene_metadata: Dict[str, Any],
    ) -> np.ndarray:
        """
        Generate a full-image dirt intensity map (float32, range [0, 1]).

        This map is rendered by projecting each glass pane's dirt texture
        onto the image plane using the camera parameters in scene_metadata.

        Parameters
        ----------
        glass_objects : list
            List of glass pane Blender objects.
        image_width, image_height : int
            Output image dimensions.
        scene_metadata : dict
            Must contain 'camera' key with projection info.

        Returns
        -------
        np.ndarray of shape (H, W) float32
        """
        dirt_map = np.zeros((image_height, image_width), dtype=np.float32)

        camera = bpy.context.scene.camera
        if camera is None:
            return dirt_map

        render = bpy.context.scene.render
        scene = bpy.context.scene

        for glass_obj in glass_objects:
            mat = glass_obj.data.materials[0] if glass_obj.data.materials else None
            if mat is None:
                continue

            dirt_density = mat.get("dirt_intensity", 0.0)
            if dirt_density < 0.01:
                continue

            # Project glass pane vertices onto image plane
            pane_mask = self._project_object_to_image(
                glass_obj, camera, image_width, image_height, scene
            )

            # Generate procedural dirt texture for this pane
            pane_dirt = self._generate_pane_dirt_texture(
                glass_obj, image_width, image_height, pane_mask, dirt_density
            )

            # Accumulate onto global dirt map
            dirt_map = np.maximum(dirt_map, pane_dirt)

        return np.clip(dirt_map, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Blender node manipulation
    # ------------------------------------------------------------------

    def _apply_dirt_to_material(
        self,
        glass_obj: bpy.types.Object,
        density: float,
        pattern: str,
        scale: float,
    ) -> None:
        if not glass_obj.data.materials:
            return

        mat = glass_obj.data.materials[0]
        mat["dirt_intensity"] = density
        mat["dirt_pattern"] = pattern

        nodes = mat.node_tree.nodes
        noise_node = nodes.get("DirtNoiseTexture")
        ramp_node = nodes.get("DirtRamp")
        mapping_node = nodes.get("DirtMapping")

        if noise_node is None:
            return

        if pattern == "perlin":
            noise_node.inputs["Scale"].default_value = scale
            noise_node.inputs["Detail"].default_value = self.rng.uniform(4.0, 12.0)
            noise_node.inputs["Roughness"].default_value = self.rng.uniform(0.5, 0.9)
            noise_node.inputs["Distortion"].default_value = self.rng.uniform(0.0, 1.0)

        elif pattern == "voronoi":
            # Use Voronoi texture by swapping node type (re-create)
            self._swap_to_voronoi(mat, nodes, scale)

        elif pattern == "streaks":
            # Stretch noise vertically for streak effect
            if mapping_node:
                mapping_node.inputs["Scale"].default_value[0] = scale * 0.2
                mapping_node.inputs["Scale"].default_value[2] = scale * 3.0
            noise_node.inputs["Distortion"].default_value = self.rng.uniform(0.5, 2.0)

        elif pattern == "dust_spots":
            noise_node.inputs["Scale"].default_value = scale * 2.0
            noise_node.inputs["Detail"].default_value = 2.0
            noise_node.inputs["Roughness"].default_value = 0.2

        elif pattern == "water_stains":
            noise_node.inputs["Scale"].default_value = scale * 1.5
            noise_node.inputs["Detail"].default_value = self.rng.uniform(8.0, 15.0)
            noise_node.inputs["Distortion"].default_value = self.rng.uniform(1.5, 3.0)

        # Adjust ramp thresholds based on density
        if ramp_node:
            # High density → lower threshold → more dirt appears
            low = max(0.0, 0.8 - density * 0.9)
            high = min(1.0, low + 0.3)
            ramp_node.color_ramp.elements[0].position = low
            ramp_node.color_ramp.elements[1].position = high

        # Increase glass roughness where dirty
        glass_bsdf = nodes.get("GlassBSDF")
        if glass_bsdf:
            base_roughness = glass_bsdf.inputs["Roughness"].default_value
            glass_bsdf.inputs["Roughness"].default_value = min(
                1.0,
                base_roughness + density * self.cfg.get("dirt_roughness_boost", 0.4),
            )

    def _swap_to_voronoi(
        self,
        mat: bpy.types.Material,
        nodes,
        scale: float,
    ) -> None:
        """Replace the Noise texture node with a Voronoi for cell-like dirt."""
        old_noise = nodes.get("DirtNoiseTexture")
        if old_noise is None:
            return

        voronoi = nodes.new("ShaderNodeTexVoronoi")
        voronoi.location = old_noise.location
        voronoi.name = "DirtNoiseTexture"  # keep name for compatibility
        voronoi.inputs["Scale"].default_value = scale
        voronoi.inputs["Randomness"].default_value = self.rng.uniform(0.7, 1.0)
        voronoi.feature = "F2"
        voronoi.distance = "EUCLIDEAN"

        links = mat.node_tree.links
        ramp_node = nodes.get("DirtRamp")
        mapping_node = nodes.get("DirtMapping")
        if ramp_node:
            links.new(voronoi.outputs["Distance"], ramp_node.inputs["Fac"])
        if mapping_node:
            links.new(mapping_node.outputs["Vector"], voronoi.inputs["Vector"])

        nodes.remove(old_noise)

    # ------------------------------------------------------------------
    # Ground-truth map generation
    # ------------------------------------------------------------------

    def _project_object_to_image(
        self,
        obj: bpy.types.Object,
        camera: bpy.types.Object,
        img_w: int,
        img_h: int,
        scene: bpy.types.Scene,
    ) -> np.ndarray:
        """
        Project object bounding box vertices onto image plane.
        Returns a boolean mask (H, W) where the object is visible.
        """
        from bpy_extras.object_utils import world_to_camera_view

        mask = np.zeros((img_h, img_w), dtype=np.float32)
        verts_world = [obj.matrix_world @ v.co for v in obj.data.vertices]

        # Get 2D bounding box in image space
        coords_2d = [
            world_to_camera_view(scene, camera, v) for v in verts_world
        ]
        xs = [c.x for c in coords_2d if 0.0 <= c.x <= 1.0 and 0.0 <= c.y <= 1.0 and c.z > 0]
        ys = [c.y for c in coords_2d if 0.0 <= c.x <= 1.0 and 0.0 <= c.y <= 1.0 and c.z > 0]

        if not xs or not ys:
            return mask

        x_min = int(min(xs) * img_w)
        x_max = int(max(xs) * img_w)
        # Blender Y=0 is bottom, image Y=0 is top
        y_min = int((1.0 - max(ys)) * img_h)
        y_max = int((1.0 - min(ys)) * img_h)

        x_min = max(0, x_min)
        x_max = min(img_w - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(img_h - 1, y_max)

        if x_max > x_min and y_max > y_min:
            mask[y_min:y_max, x_min:x_max] = 1.0

        return mask

    def _generate_pane_dirt_texture(
        self,
        glass_obj: bpy.types.Object,
        img_w: int,
        img_h: int,
        pane_mask: np.ndarray,
        density: float,
    ) -> np.ndarray:
        """
        Generate a NumPy dirt texture for a single glass pane using the
        same pattern parameters as the Blender material.
        """
        pattern = glass_obj.data.materials[0].get("dirt_pattern", "perlin") if glass_obj.data.materials else "perlin"

        # Generate pattern on full canvas, then mask
        yy, xx = np.mgrid[0:img_h, 0:img_w].astype(np.float32)
        yy /= img_h
        xx /= img_w

        if pattern == "perlin":
            dirt = self._np_perlin_approx(xx, yy, scale=4.0)
        elif pattern == "voronoi":
            dirt = self._np_voronoi_approx(img_w, img_h)
        elif pattern == "streaks":
            dirt = self._np_streaks(xx, yy)
        elif pattern == "dust_spots":
            dirt = self._np_dust_spots(img_w, img_h)
        elif pattern == "water_stains":
            dirt = self._np_water_stains(xx, yy)
        else:
            dirt = self._np_perlin_approx(xx, yy)

        # Threshold and scale by density
        threshold = 1.0 - density
        dirt = np.clip((dirt - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)
        dirt = dirt * pane_mask

        return dirt.astype(np.float32)

    # ------------------------------------------------------------------
    # NumPy texture generators (approximate procedural patterns)
    # ------------------------------------------------------------------

    def _np_perlin_approx(
        self, xx: np.ndarray, yy: np.ndarray, scale: float = 4.0
    ) -> np.ndarray:
        """Approximate Perlin noise using layered sin/cos."""
        noise = np.zeros_like(xx)
        amplitude = 1.0
        frequency = scale
        for _ in range(5):
            angle = self._np_rng.uniform(0, 2 * math.pi)
            noise += amplitude * np.sin(
                frequency * (xx * math.cos(angle) + yy * math.sin(angle))
                + self._np_rng.uniform(0, 2 * math.pi)
            )
            amplitude *= 0.5
            frequency *= 2.0
        return (noise - noise.min()) / ((noise.max() - noise.min()) + 1e-8)

    def _np_voronoi_approx(self, img_w: int, img_h: int) -> np.ndarray:
        """Approximate Voronoi distance field."""
        num_points = self._np_rng.integers(20, 80)
        px = self._np_rng.uniform(0, img_w, num_points)
        py = self._np_rng.uniform(0, img_h, num_points)
        yy, xx = np.mgrid[0:img_h, 0:img_w].astype(np.float32)
        dist = np.full((img_h, img_w), np.inf, dtype=np.float32)
        for i in range(num_points):
            d = np.sqrt((xx - px[i]) ** 2 + (yy - py[i]) ** 2)
            dist = np.minimum(dist, d)
        dist /= dist.max() + 1e-8
        return 1.0 - dist  # invert: bright = near cell centers

    def _np_streaks(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """Vertical streaks pattern."""
        angle_rad = math.radians(
            self.rng.uniform(-self.cfg.get("streak_direction_range", 20), self.cfg.get("streak_direction_range", 20))
        )
        rotated_x = xx * math.cos(angle_rad) - yy * math.sin(angle_rad)
        streak = np.abs(np.sin(rotated_x * self._np_rng.uniform(3, 10) * math.pi))
        return streak ** 3  # sharpen streaks

    def _np_dust_spots(self, img_w: int, img_h: int) -> np.ndarray:
        """Random circular dust spots."""
        map_ = np.zeros((img_h, img_w), dtype=np.float32)
        yy, xx = np.mgrid[0:img_h, 0:img_w].astype(np.float32)
        num_spots = self._np_rng.integers(5, 50)
        for _ in range(num_spots):
            cx = self._np_rng.uniform(0, img_w)
            cy = self._np_rng.uniform(0, img_h)
            radius = self._np_rng.uniform(5, 60)
            intensity = self._np_rng.uniform(0.3, 1.0)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            spot = intensity * np.exp(-(dist ** 2) / (2 * radius ** 2))
            map_ = np.maximum(map_, spot)
        return map_

    def _np_water_stains(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """Trailing water stain patterns."""
        stain = np.zeros_like(xx)
        for _ in range(self._np_rng.integers(3, 15)):
            cx = self._np_rng.uniform(0.1, 0.9)
            freq = self._np_rng.uniform(2.0, 8.0)
            stain += np.exp(-((xx - cx) ** 2) / 0.01) * np.maximum(0, np.sin(yy * freq * math.pi * 2 + self._np_rng.uniform(0, math.pi)))
        return np.clip(stain / (stain.max() + 1e-8), 0.0, 1.0)
