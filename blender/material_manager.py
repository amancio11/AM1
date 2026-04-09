"""
material_manager.py
===================
Manages all Blender materials for the synthetic data pipeline.

Creates physically-based materials using Blender's node editor:
  - Glass pane material (with optional dirt layer)
  - Facade materials (concrete, brick, metal panel, stone)
  - Window frame material
  - Dirt overlay material

Each glass pane material stores a reference to its dirt texture node
so dirt_simulator.py can update the dirt pattern per render.
"""

import bpy
import random
import math
from typing import Dict, Any, List, Tuple, Optional


class MaterialManager:
    """Creates and manages PBR materials for the building scene."""

    GLASS_MAT_PREFIX = "Glass_Mat"
    FACADE_MAT_PREFIX = "Facade_Mat"

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.glass_cfg = config["glass"]
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Glass Material
    # ------------------------------------------------------------------

    def create_glass_material(
        self, obj: bpy.types.Object, dirt_intensity: float = 0.0
    ) -> bpy.types.Material:
        """
        Build a PBR glass material with an optional dirt overlay.

        The material uses Blender's Principled BSDF with transmission for
        the glass layer, mixed with a Diffuse BSDF for the dirt layer.

        Parameters
        ----------
        obj : bpy.types.Object
            The glass pane object to assign the material to.
        dirt_intensity : float
            Scalar [0, 1] controlling how visible the dirt is initially.
            The DirtSimulator updates the texture map later.

        Returns
        -------
        bpy.types.Material
        """
        mat_name = f"{self.GLASS_MAT_PREFIX}_{obj.name}"
        if mat_name in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[mat_name])

        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        mat.blend_method = "HASHED"      # for alpha transparency
        mat.shadow_method = "HASHED"

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # --- Output ---
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (800, 0)

        # --- Mix Shader: glass + dirt ---
        mix_shader = nodes.new("ShaderNodeMixShader")
        mix_shader.location = (600, 0)
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

        # --- Glass BSDF (Principled with transmission) ---
        glass_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        glass_bsdf.location = (300, 100)
        glass_bsdf.name = "GlassBSDF"

        ior = self.rng.uniform(
            self.glass_cfg["ior_min"], self.glass_cfg["ior_max"]
        )
        roughness = self.rng.uniform(
            self.glass_cfg["roughness_min"], self.glass_cfg["roughness_max"]
        )
        tint = self.rng.choice(self.glass_cfg["tint_colors"])

        glass_bsdf.inputs["Base Color"].default_value = (*tint, 1.0)
        glass_bsdf.inputs["Roughness"].default_value = roughness
        glass_bsdf.inputs["IOR"].default_value = ior
        glass_bsdf.inputs["Transmission Weight"].default_value = 0.95
        glass_bsdf.inputs["Specular IOR Level"].default_value = 0.5
        glass_bsdf.inputs["Alpha"].default_value = 1.0

        links.new(glass_bsdf.outputs["BSDF"], mix_shader.inputs[1])

        # --- Dirt BSDF ---
        dirt_bsdf = nodes.new("ShaderNodeBsdfDiffuse")
        dirt_bsdf.location = (300, -150)
        dirt_bsdf.name = "DirtBSDF"
        dirt_color = self.rng.choice(self.glass_cfg.get("dirt_colors", [
            [0.18, 0.15, 0.12],
            [0.45, 0.42, 0.38],
        ]))
        dirt_bsdf.inputs["Color"].default_value = (*dirt_color, 1.0)
        dirt_bsdf.inputs["Roughness"].default_value = 0.8

        links.new(dirt_bsdf.outputs["BSDF"], mix_shader.inputs[2])

        # --- Dirt texture (procedural noise — updated by DirtSimulator) ---
        dirt_tex = nodes.new("ShaderNodeTexNoise")
        dirt_tex.location = (0, -200)
        dirt_tex.name = "DirtNoiseTexture"
        dirt_tex.inputs["Scale"].default_value = 4.0
        dirt_tex.inputs["Detail"].default_value = 8.0
        dirt_tex.inputs["Roughness"].default_value = 0.7
        dirt_tex.inputs["Distortion"].default_value = 0.5

        # Remap noise to sharper dirt mask
        color_ramp = nodes.new("ShaderNodeValToRGB")
        color_ramp.location = (200, -200)
        color_ramp.name = "DirtRamp"
        color_ramp.color_ramp.interpolation = "EASE"
        color_ramp.color_ramp.elements[0].position = max(0.0, 1.0 - dirt_intensity * 1.5)
        color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        color_ramp.color_ramp.elements[1].position = min(1.0, 1.0 - dirt_intensity * 0.5)
        color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

        links.new(dirt_tex.outputs["Fac"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], mix_shader.inputs["Fac"])

        # --- UV mapping ---
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-400, -200)
        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (-200, -200)
        mapping.name = "DirtMapping"
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], dirt_tex.inputs["Vector"])

        # Store node references in custom properties for DirtSimulator
        mat["dirt_noise_node"] = "DirtNoiseTexture"
        mat["dirt_ramp_node"] = "DirtRamp"
        mat["dirt_mapping_node"] = "DirtMapping"
        mat["dirt_intensity"] = dirt_intensity

        obj.data.materials.clear()
        obj.data.materials.append(mat)
        return mat

    def create_glass_material_with_image_dirt(
        self,
        obj: bpy.types.Object,
        dirt_image: bpy.types.Image,
        dirt_intensity: float = 1.0,
    ) -> bpy.types.Material:
        """
        Variant that uses an image texture node for the dirt map.
        Enables pixel-accurate dirt placement from a pre-generated texture.
        """
        mat = self.create_glass_material(obj, dirt_intensity=0.0)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Replace noise texture with image texture
        noise_node = nodes.get("DirtNoiseTexture")
        if noise_node:
            nodes.remove(noise_node)

        img_tex = nodes.new("ShaderNodeTexImage")
        img_tex.location = (0, -200)
        img_tex.name = "DirtImageTexture"
        img_tex.image = dirt_image
        img_tex.interpolation = "Linear"

        ramp_node = nodes.get("DirtRamp")
        mapping_node = nodes.get("DirtMapping")
        if ramp_node and mapping_node:
            links.new(mapping_node.outputs["Vector"], img_tex.inputs["Vector"])
            links.new(img_tex.outputs["Color"], ramp_node.inputs["Fac"])

        # Scale dirt by intensity
        math_node = nodes.new("ShaderNodeMath")
        math_node.operation = "MULTIPLY"
        math_node.location = (400, -250)
        math_node.inputs[1].default_value = dirt_intensity
        mix_shader = nodes.get("Mix Shader") or nodes.get("ShaderNodeMixShader")
        if ramp_node and mix_shader:
            links.new(ramp_node.outputs["Color"], math_node.inputs[0])
            links.new(math_node.outputs["Value"], mix_shader.inputs["Fac"])

        mat["dirt_image_node"] = "DirtImageTexture"
        return mat

    # ------------------------------------------------------------------
    # Facade Material
    # ------------------------------------------------------------------

    def setup_facade_material(
        self, material_name: str, mat: bpy.types.Material
    ) -> None:
        """
        Configure node tree for a facade material.
        Called by BuildingGenerator after stubbing the material.
        """
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Keep existing output if present
        output = next(
            (n for n in nodes if n.type == "OUTPUT_MATERIAL"), None
        )
        if output is None:
            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (600, 0)

        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (600, 0)

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (300, 0)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        noise = nodes.new("ShaderNodeTexNoise")
        noise.location = (-200, 0)

        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (0, 0)
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        facade_type = mat.get("facade_type", "concrete")
        self._configure_facade_type(facade_type, bsdf, noise, ramp)

    def _configure_facade_type(
        self,
        facade_type: str,
        bsdf: bpy.types.ShaderNode,
        noise: bpy.types.ShaderNode,
        ramp: bpy.types.ShaderNode,
    ) -> None:
        presets = {
            "concrete": {
                "color1": (0.50, 0.50, 0.50, 1),
                "color2": (0.65, 0.65, 0.65, 1),
                "roughness": 0.85,
                "noise_scale": 25.0,
                "metallic": 0.0,
            },
            "brick": {
                "color1": (0.55, 0.25, 0.15, 1),
                "color2": (0.70, 0.35, 0.20, 1),
                "roughness": 0.90,
                "noise_scale": 40.0,
                "metallic": 0.0,
            },
            "metal_panel": {
                "color1": (0.60, 0.62, 0.65, 1),
                "color2": (0.70, 0.72, 0.75, 1),
                "roughness": 0.30,
                "noise_scale": 5.0,
                "metallic": 0.9,
            },
            "stone": {
                "color1": (0.45, 0.42, 0.38, 1),
                "color2": (0.60, 0.57, 0.52, 1),
                "roughness": 0.88,
                "noise_scale": 20.0,
                "metallic": 0.0,
            },
            "window_frame": {
                "color1": (0.15, 0.15, 0.15, 1),
                "color2": (0.25, 0.25, 0.25, 1),
                "roughness": 0.4,
                "noise_scale": 2.0,
                "metallic": 0.5,
            },
        }
        p = presets.get(facade_type, presets["concrete"])
        ramp.color_ramp.elements[0].color = p["color1"]
        ramp.color_ramp.elements[1].color = p["color2"]
        bsdf.inputs["Roughness"].default_value = p["roughness"]
        bsdf.inputs["Metallic"].default_value = p["metallic"]
        noise.inputs["Scale"].default_value = p["noise_scale"]
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.6

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def setup_all_facade_materials(self) -> None:
        """Iterate all stub facade materials and apply proper node setups."""
        for mat in bpy.data.materials:
            if "facade_type" in mat:
                self.setup_facade_material(mat["facade_type"], mat)
