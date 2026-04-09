"""
render_pipeline.py
==================
Orchestrates the full per-scene rendering pipeline.

Per-scene steps:
  1. Generate building facade (BuildingGenerator)
  2. Setup materials (MaterialManager)
  3. Simulate dirt (DirtSimulator)
  4. Place drone camera (CameraController)
  5. Randomize lighting (LightingRandomizer)
  6. Render RGB image
  7. Render glass mask
  8. Generate/render dirt map
  9. Export metadata JSON
"""

import bpy
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

from building_generator import BuildingGenerator
from material_manager import MaterialManager
from dirt_simulator import DirtSimulator
from lighting_randomizer import LightingRandomizer
from camera_controller import CameraController
from mask_exporter import MaskExporter


class RenderPipeline:
    """
    Full synthetic data generation pipeline for one scene ID.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.render_cfg = config["render"]
        self.export_cfg = config["export"]
        self._setup_render_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_scene(self, scene_id: int) -> Dict[str, Any]:
        """
        Render a single complete scene and export all outputs.

        Parameters
        ----------
        scene_id : int
            Unique identifier for this scene (also used as RNG seed offset).

        Returns
        -------
        dict
            Full metadata for this scene.
        """
        seed = self.config["generation"]["seed"] + scene_id
        t_start = time.perf_counter()

        out_root = Path(self.render_cfg["output_dir"])
        out_images = out_root / "images"
        out_masks = out_root / "glass_masks"
        out_dirt = out_root / "dirt_maps"
        out_meta = out_root / "metadata"
        for d in [out_images, out_masks, out_dirt, out_meta]:
            d.mkdir(parents=True, exist_ok=True)

        scene_name = f"scene_{scene_id:06d}"
        rgb_path = str(out_images / f"{scene_name}.png")
        mask_path = str(out_masks / f"{scene_name}.png")
        dirt_path = str(out_dirt / f"{scene_name}.png")
        meta_path = str(out_meta / f"{scene_name}.json")

        # --- 1. Generate building ---
        building_gen = BuildingGenerator(self.config, seed=seed)
        facade_meta = building_gen.generate()
        glass_objects = building_gen.glass_objects
        facade_objects = building_gen.facade_objects

        # --- 2. Setup materials ---
        mat_mgr = MaterialManager(self.config, seed=seed + 1000)
        mat_mgr.setup_all_facade_materials()

        # --- 3. Simulate dirt ---
        dirt_sim = DirtSimulator(self.config, seed=seed + 2000)
        dirt_density_map = dirt_sim.randomize_dirt(glass_objects)

        # Assign glass materials (with dirt)
        for glass_obj in glass_objects:
            density = dirt_density_map.get(glass_obj.name, 0.0)
            mat_mgr.create_glass_material(glass_obj, dirt_intensity=density)

        # --- 4. Camera ---
        cam_ctrl = CameraController(self.config, seed=seed + 3000)
        camera_meta = cam_ctrl.setup(facade_meta)

        # --- 5. Lighting ---
        light_rand = LightingRandomizer(self.config, seed=seed + 4000)
        lighting_meta = light_rand.randomize()

        # --- 6. Render RGB ---
        scene = bpy.context.scene
        scene.render.filepath = rgb_path
        bpy.ops.render.render(write_still=True)

        # --- 7. Render glass mask ---
        mask_exp = MaskExporter(self.config)
        if self.export_cfg.get("glass_masks", True):
            mask_exp.render_glass_mask(glass_objects, facade_objects, mask_path)

        # --- 8. Generate dirt map ---
        if self.export_cfg.get("dirt_maps", True):
            render_w = scene.render.resolution_x
            render_h = scene.render.resolution_y
            dirt_array = dirt_sim.generate_ground_truth_dirt_map(
                glass_objects,
                render_w,
                render_h,
                {"camera": camera_meta},
            )
            mask_exp.save_dirt_map_from_array(dirt_array, dirt_path)

        # --- 9. Metadata ---
        t_end = time.perf_counter()
        metadata = {
            "scene_id": scene_id,
            "scene_name": scene_name,
            "seed": seed,
            "render_time_s": round(t_end - t_start, 2),
            "building": facade_meta,
            "camera": camera_meta,
            "lighting": lighting_meta,
            "dirt_density_per_glass": dirt_density_map,
            "outputs": {
                "rgb": rgb_path,
                "glass_mask": mask_path,
                "dirt_map": dirt_path,
            },
        }

        if self.export_cfg.get("metadata_json", True):
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        print(f"[RenderPipeline] Scene {scene_id:06d} done in {t_end-t_start:.1f}s")
        return metadata

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_render_settings(self) -> None:
        scene = bpy.context.scene
        render = scene.render

        render.engine = self.render_cfg.get("engine", "CYCLES")
        render.resolution_x = self.render_cfg.get("resolution_x", 1280)
        render.resolution_y = self.render_cfg.get("resolution_y", 720)
        render.resolution_percentage = 100
        render.image_settings.file_format = self.render_cfg.get("file_format", "PNG")
        render.image_settings.color_depth = str(self.render_cfg.get("color_depth", 8))
        render.image_settings.color_mode = "RGB"
        render.use_file_extension = True
        render.use_overwrite = True

        if render.engine == "CYCLES":
            cycles = scene.cycles
            device = self.render_cfg.get("device", "GPU")
            cycles.device = device
            cycles.samples = self.render_cfg.get("samples", 128)
            cycles.use_denoising = True
            cycles.tile_size = self.render_cfg.get("tile_size", 256)

            if device == "GPU":
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = "CUDA"
                prefs.get_devices()
                for dev in prefs.devices:
                    dev.use = True
