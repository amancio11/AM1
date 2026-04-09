"""
building_generator.py
=====================
Procedural building facade generation using Blender Python API.

Generates a realistic multi-floor building facade with:
  - Configurable number of floors and windows per floor
  - Procedural facade materials (concrete, brick, metal panel, stone)
  - Frame geometry around each window
  - Glass pane objects tagged for material assignment and mask export

All dimensions are in meters. The facade is placed in the XZ plane
(X = horizontal, Z = vertical, Y = depth away from camera).
"""

import bpy
import bmesh
import random
import math
from mathutils import Vector
from typing import List, Tuple, Dict, Any


class BuildingGenerator:
    """Procedurally generates a building facade scene."""

    # Object name prefixes used throughout the pipeline
    FACADE_PREFIX = "Facade_Wall"
    WINDOW_FRAME_PREFIX = "Window_Frame"
    GLASS_PREFIX = "Glass_Pane"
    COLLECTION_NAME = "Building_Facade"

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.cfg = config["building"]
        self.rng = random.Random(seed)
        self._glass_objects: List[bpy.types.Object] = []
        self._facade_objects: List[bpy.types.Object] = []
        self._collection: bpy.types.Collection = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Dict[str, Any]:
        """
        Generate a complete facade scene.

        Returns
        -------
        dict
            Metadata about the generated scene (floors, windows, dimensions, etc.)
        """
        self._clear_scene()
        self._collection = self._get_or_create_collection(self.COLLECTION_NAME)

        num_floors = self.rng.randint(
            self.cfg["floors_min"], self.cfg["floors_max"]
        )
        windows_per_floor = self.rng.randint(
            self.cfg["windows_per_floor_min"], self.cfg["windows_per_floor_max"]
        )
        floor_height = self.rng.uniform(
            self.cfg["floor_height_min"], self.cfg["floor_height_max"]
        )
        facade_width = self.rng.uniform(
            self.cfg["facade_width_min"], self.cfg["facade_width_max"]
        )
        facade_depth = self.rng.uniform(0.2, 0.6)          # wall thickness
        facade_material_name = self.rng.choice(
            self.cfg["facade_materials"]
        )

        total_height = num_floors * floor_height
        window_data = []

        # --- Floor slab geometry (base wall) ---
        wall_obj = self._create_wall(
            width=facade_width,
            height=total_height,
            depth=facade_depth,
            material_name=facade_material_name,
        )
        self._facade_objects.append(wall_obj)

        # --- Generate windows grid ---
        for floor_idx in range(num_floors):
            floor_z_base = floor_idx * floor_height
            win_w = self.rng.uniform(
                self.cfg["window_width_min"], self.cfg["window_width_max"]
            )
            win_h = self.rng.uniform(
                self.cfg["window_height_min"], self.cfg["window_height_max"]
            )
            margin_h = self.rng.uniform(
                self.cfg["window_margin_min"], self.cfg["window_margin_max"]
            )
            margin_v = (floor_height - win_h) / 2.0

            # Distribute windows evenly across facade width
            total_win_w = windows_per_floor * win_w
            spacing = (facade_width - total_win_w) / (windows_per_floor + 1)

            if spacing < 0.1:
                # Not enough space — reduce window count
                windows_per_floor = max(1, windows_per_floor - 1)
                spacing = max(0.1, (facade_width - windows_per_floor * win_w) / (windows_per_floor + 1))

            for win_idx in range(windows_per_floor):
                win_x = -facade_width / 2.0 + spacing + win_idx * (win_w + spacing) + win_w / 2.0
                win_z = floor_z_base + margin_v + win_h / 2.0

                # Window niche (recess into wall)
                niche_depth = self.rng.uniform(0.05, 0.15)

                # Frame
                frame_obj = self._create_window_frame(
                    center=(win_x, 0.0, win_z),
                    width=win_w,
                    height=win_h,
                    depth=niche_depth,
                    frame_thickness=self.rng.uniform(0.04, 0.10),
                    name=f"{self.WINDOW_FRAME_PREFIX}_F{floor_idx}_W{win_idx}",
                )
                self._facade_objects.append(frame_obj)

                # Glass pane
                glass_obj = self._create_glass_pane(
                    center=(win_x, -niche_depth / 2.0, win_z),
                    width=win_w - 0.08,
                    height=win_h - 0.08,
                    name=f"{self.GLASS_PREFIX}_F{floor_idx}_W{win_idx}",
                )
                self._glass_objects.append(glass_obj)

                window_data.append(
                    {
                        "floor": floor_idx,
                        "window": win_idx,
                        "center_x": win_x,
                        "center_z": win_z,
                        "width": win_w,
                        "height": win_h,
                        "object_name": glass_obj.name,
                    }
                )

        return {
            "num_floors": num_floors,
            "windows_per_floor": windows_per_floor,
            "floor_height": floor_height,
            "facade_width": facade_width,
            "total_height": total_height,
            "facade_material": facade_material_name,
            "windows": window_data,
            "glass_object_names": [o.name for o in self._glass_objects],
        }

    @property
    def glass_objects(self) -> List[bpy.types.Object]:
        return self._glass_objects

    @property
    def facade_objects(self) -> List[bpy.types.Object]:
        return self._facade_objects

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear_scene(self) -> None:
        """Remove all mesh objects from the current scene."""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        # Remove orphaned meshes / materials
        for block in list(bpy.data.meshes):
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in list(bpy.data.materials):
            if block.users == 0:
                bpy.data.materials.remove(block)

    def _get_or_create_collection(self, name: str) -> bpy.types.Collection:
        if name in bpy.data.collections:
            return bpy.data.collections[name]
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        return col

    def _link_to_collection(self, obj: bpy.types.Object) -> None:
        """Link object to building collection, unlink from scene root."""
        for col in obj.users_collection:
            col.objects.unlink(obj)
        self._collection.objects.link(obj)

    def _create_wall(
        self,
        width: float,
        height: float,
        depth: float,
        material_name: str,
    ) -> bpy.types.Object:
        mesh = bpy.data.meshes.new("Wall_Mesh")
        obj = bpy.data.objects.new(self.FACADE_PREFIX, mesh)
        self._link_to_collection(obj)
        bpy.context.view_layer.objects.active = obj

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()

        obj.scale = (width, depth, height)
        obj.location = (0.0, depth / 2.0, height / 2.0)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.transform_apply(scale=True, location=True)

        mat = self._get_or_create_facade_material(material_name)
        obj.data.materials.append(mat)
        return obj

    def _create_window_frame(
        self,
        center: Tuple[float, float, float],
        width: float,
        height: float,
        depth: float,
        frame_thickness: float,
        name: str,
    ) -> bpy.types.Object:
        """Create a window frame (rectangular border) around the glass opening."""
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        obj = bpy.data.objects.new(name, mesh)
        self._link_to_collection(obj)

        bm = bmesh.new()

        # Outer rect
        ox, oy, oz = center
        hw_o = width / 2.0
        hh_o = height / 2.0
        hw_i = hw_o - frame_thickness
        hh_i = hh_o - frame_thickness
        front_y = oy
        back_y = oy + depth

        # Create frame as 4 rectangular bars
        def _rect_bar(x0, x1, z0, z1):
            verts = [
                bm.verts.new((x0, front_y, z0)),
                bm.verts.new((x1, front_y, z0)),
                bm.verts.new((x1, front_y, z1)),
                bm.verts.new((x0, front_y, z1)),
                bm.verts.new((x0, back_y, z0)),
                bm.verts.new((x1, back_y, z0)),
                bm.verts.new((x1, back_y, z1)),
                bm.verts.new((x0, back_y, z1)),
            ]
            faces = [
                [verts[0], verts[1], verts[2], verts[3]],  # front
                [verts[7], verts[6], verts[5], verts[4]],  # back
                [verts[0], verts[4], verts[5], verts[1]],  # bottom
                [verts[3], verts[2], verts[6], verts[7]],  # top
                [verts[0], verts[3], verts[7], verts[4]],  # left
                [verts[1], verts[5], verts[6], verts[2]],  # right
            ]
            for f in faces:
                bm.faces.new(f)

        _cx = ox
        _cz = oz
        # Bottom bar
        _rect_bar(_cx - hw_o, _cx + hw_o, _cz - hh_o, _cz - hh_i)
        # Top bar
        _rect_bar(_cx - hw_o, _cx + hw_o, _cz + hh_i, _cz + hh_o)
        # Left bar
        _rect_bar(_cx - hw_o, _cx - hw_i, _cz - hh_i, _cz + hh_i)
        # Right bar
        _rect_bar(_cx + hw_i, _cx + hw_o, _cz - hh_i, _cz + hh_i)

        bm.to_mesh(mesh)
        bm.free()

        mat = self._get_or_create_facade_material("window_frame")
        obj.data.materials.append(mat)
        return obj

    def _create_glass_pane(
        self,
        center: Tuple[float, float, float],
        width: float,
        height: float,
        name: str,
        thickness: float = 0.01,
    ) -> bpy.types.Object:
        """Create a thin glass pane object, tagged with custom property."""
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        obj = bpy.data.objects.new(name, mesh)
        self._link_to_collection(obj)

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()

        cx, cy, cz = center
        obj.scale = (width, thickness, height)
        obj.location = (cx, cy, cz)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(scale=True, location=True)

        # Custom property: used by mask exporter to identify glass objects
        obj["is_glass"] = True
        obj["glass_id"] = name

        return obj

    # ------------------------------------------------------------------
    # Material stubs (actual node setup handled by material_manager.py)
    # ------------------------------------------------------------------

    def _get_or_create_facade_material(self, name: str) -> bpy.types.Material:
        mat_name = f"FAC_{name}"
        if mat_name in bpy.data.materials:
            return bpy.data.materials[mat_name]
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        # Material nodes are configured by MaterialManager — stub here
        mat["facade_type"] = name
        return mat
