"""
mask_exporter.py
================
Exports glass segmentation masks and dirt intensity maps as PNG images.

Strategy
--------
Two render passes are used to generate ground-truth annotations:

1. Glass mask (binary)
   - Temporarily assign a flat white emission material to all glass panes
   - Assign flat black to everything else
   - Render at 1 sample (instant, no noise)
   - Result: white pixels = glass, black pixels = non-glass

2. Dirt map (greyscale heatmap)
   - Assign a flat grayscale emission based on the per-pane dirt density
   - Or: use the pre-computed NumPy dirt map from DirtSimulator and save directly
   - Result: pixel intensity ∈ [0, 255] encodes dirt level

Both outputs are 8-bit PNG files matching the RGB render resolution.
"""

import bpy
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


class MaskExporter:
    """Handles generation and saving of segmentation masks and dirt maps."""

    MASK_MATERIAL_NAME = "__MASK_TMP__"
    DIRT_MATERIAL_PREFIX = "__DIRT_TMP_"

    def __init__(self, config: Dict[str, Any]):
        self.render_cfg = config["render"]
        self.export_cfg = config["export"]
        self._original_materials: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Glass Mask
    # ------------------------------------------------------------------

    def render_glass_mask(
        self,
        glass_objects: List[bpy.types.Object],
        facade_objects: List[bpy.types.Object],
        output_path: str,
    ) -> None:
        """
        Render a binary glass segmentation mask.

        Glass pixels → white (255)
        Non-glass pixels → black (0)
        """
        scene = bpy.context.scene
        render = scene.render

        # Save original settings
        orig_samples = scene.cycles.samples if scene.render.engine == "CYCLES" else 1
        orig_path = render.filepath
        orig_engine = render.engine

        # Switch to fast render
        render.engine = "CYCLES"
        scene.cycles.samples = self.export_cfg.get("mask_render_samples", 1)
        scene.cycles.use_denoising = False

        # Stash original materials and apply mask materials
        self._stash_and_apply_mask_materials(glass_objects, facade_objects)

        # Disable world lighting (flat render)
        orig_world_nodes = self._disable_world_lighting()

        render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        # Restore everything
        self._restore_materials(glass_objects, facade_objects)
        self._restore_world_lighting(orig_world_nodes)
        scene.cycles.samples = orig_samples
        render.filepath = orig_path
        render.engine = orig_engine

    def save_dirt_map_from_array(
        self,
        dirt_array: np.ndarray,
        output_path: str,
    ) -> None:
        """
        Save a pre-computed NumPy dirt map (float32 H×W) directly as PNG.
        Values are scaled to uint8 [0, 255].
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dirt_uint8 = (np.clip(dirt_array, 0, 1) * 255).astype(np.uint8)

        # Use Blender image API to save (avoids external OpenCV dependency here)
        img_name = "__DIRT_EXPORT__"
        h, w = dirt_uint8.shape
        if img_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[img_name])

        blender_img = bpy.data.images.new(img_name, width=w, height=h, alpha=False)
        blender_img.colorspace_settings.name = "Non-Color"

        # Blender pixel order: bottom-to-top, RGBA float
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, 0] = dirt_array[::-1]  # flip vertical
        rgba[:, :, 1] = dirt_array[::-1]
        rgba[:, :, 2] = dirt_array[::-1]
        rgba[:, :, 3] = 1.0
        blender_img.pixels = rgba.flatten().tolist()

        blender_img.filepath_raw = output_path
        blender_img.file_format = "PNG"
        blender_img.save()
        bpy.data.images.remove(blender_img)

    def render_dirt_map_blender(
        self,
        glass_objects: List[bpy.types.Object],
        facade_objects: List[bpy.types.Object],
        output_path: str,
    ) -> None:
        """
        Alternative: render dirt map through Blender by assigning flat
        emission materials whose brightness encodes the dirt density.
        More expensive but gives spatially accurate results.
        """
        scene = bpy.context.scene
        render = scene.render

        orig_samples = scene.cycles.samples
        orig_path = render.filepath

        scene.cycles.samples = self.export_cfg.get("mask_render_samples", 1)
        scene.cycles.use_denoising = False
        orig_world_nodes = self._disable_world_lighting()

        self._stash_and_apply_dirt_materials(glass_objects, facade_objects)

        render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        self._restore_materials(glass_objects, facade_objects)
        self._restore_world_lighting(orig_world_nodes)
        scene.cycles.samples = orig_samples
        render.filepath = orig_path

    # ------------------------------------------------------------------
    # Material swapping helpers
    # ------------------------------------------------------------------

    def _stash_and_apply_mask_materials(
        self,
        glass_objects: List[bpy.types.Object],
        facade_objects: List[bpy.types.Object],
    ) -> None:
        white_mat = self._make_emission_material("__GLASS_WHITE__", color=(1.0, 1.0, 1.0), strength=1.0)
        black_mat = self._make_emission_material("__FACADE_BLACK__", color=(0.0, 0.0, 0.0), strength=0.0)

        for obj in glass_objects:
            self._stash_materials(obj)
            obj.data.materials.clear()
            obj.data.materials.append(white_mat)

        for obj in facade_objects:
            self._stash_materials(obj)
            obj.data.materials.clear()
            obj.data.materials.append(black_mat)

    def _stash_and_apply_dirt_materials(
        self,
        glass_objects: List[bpy.types.Object],
        facade_objects: List[bpy.types.Object],
    ) -> None:
        black_mat = self._make_emission_material("__FACADE_BLACK__", color=(0.0, 0.0, 0.0), strength=0.0)

        for obj in facade_objects:
            self._stash_materials(obj)
            obj.data.materials.clear()
            obj.data.materials.append(black_mat)

        for obj in glass_objects:
            self._stash_materials(obj)
            orig_mat = obj.data.materials[0] if obj.data.materials else None
            dirt_level = orig_mat.get("dirt_intensity", 0.0) if orig_mat else 0.0
            c = dirt_level
            dirt_mat = self._make_emission_material(
                f"__DIRT_{obj.name}__",
                color=(c, c, c),
                strength=1.0,
            )
            obj.data.materials.clear()
            obj.data.materials.append(dirt_mat)

    def _stash_materials(self, obj: bpy.types.Object) -> None:
        self._original_materials[obj.name] = [m for m in obj.data.materials]

    def _restore_materials(
        self,
        glass_objects: List[bpy.types.Object],
        facade_objects: List[bpy.types.Object],
    ) -> None:
        all_objs = glass_objects + facade_objects
        for obj in all_objs:
            if obj.name in self._original_materials:
                obj.data.materials.clear()
                for mat in self._original_materials[obj.name]:
                    obj.data.materials.append(mat)
        self._original_materials.clear()

        # Cleanup temp materials
        for mat in list(bpy.data.materials):
            if mat.name.startswith("__") and mat.name.endswith("__"):
                if mat.users == 0:
                    bpy.data.materials.remove(mat)

    def _make_emission_material(
        self,
        name: str,
        color: tuple,
        strength: float = 1.0,
    ) -> bpy.types.Material:
        if name in bpy.data.materials:
            return bpy.data.materials[name]
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*color, 1.0)
        emission.inputs["Strength"].default_value = strength
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return mat

    # ------------------------------------------------------------------
    # World lighting helpers
    # ------------------------------------------------------------------

    def _disable_world_lighting(self) -> Any:
        """Set world background to pure black for mask renders."""
        world = bpy.context.scene.world
        if world is None:
            return None
        orig = world.node_tree.nodes.copy() if world.use_nodes else None
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputWorld")
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0
        links.new(bg.outputs["Background"], output.inputs["Surface"])
        return True  # indicates "was modified"

    def _restore_world_lighting(self, orig: Any) -> None:
        """Caller should re-run LightingRandomizer to restore proper world."""
        pass  # Lighting is re-randomized per scene anyway
