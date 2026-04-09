"""
lighting_randomizer.py
======================
Domain randomization for lighting conditions in the synthetic data pipeline.

Supports multiple weather/lighting scenarios:
  - Clear sunny sky (HDRI or Nishita sky)
  - Overcast (uniform ambient)
  - Golden hour (warm sun at low angle)
  - Harsh noon (strong top-down sun)
  - Foggy (volumetric scatter)

Uses Blender's world shader node tree for sky/environment setup
and a Sun lamp for directional lighting.
"""

import bpy
import math
import random
from typing import Dict, Any, Tuple


class LightingRandomizer:
    """Randomizes scene lighting with domain randomization strategies."""

    SUN_LAMP_NAME = "Sun_Drone"
    FILL_LAMP_NAME = "Fill_Ambient"

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.cfg = config["lighting"]
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def randomize(self) -> Dict[str, Any]:
        """
        Apply a random lighting setup to the current scene.

        Returns
        -------
        dict
            Metadata about the applied lighting configuration.
        """
        weather = self._sample_weather_condition()
        time_of_day = self.rng.uniform(
            self.cfg["time_of_day_min"], self.cfg["time_of_day_max"]
        )
        sun_energy = self.rng.uniform(
            self.cfg["sun_energy_min"], self.cfg["sun_energy_max"]
        )
        sun_angle = self.rng.uniform(
            self.cfg["sun_angle_min"], self.cfg["sun_angle_max"]
        )
        ambient_strength = self.rng.uniform(
            self.cfg["ambient_strength_min"], self.cfg["ambient_strength_max"]
        )
        turbidity = self.rng.uniform(
            self.cfg["sky_turbidity_min"], self.cfg["sky_turbidity_max"]
        )

        # Apply weather-specific overrides
        if weather["name"] == "overcast":
            sun_energy *= 0.3
            ambient_strength = self.rng.uniform(0.4, 0.7)
            turbidity = self.rng.uniform(4.0, 6.0)
        elif weather["name"] == "golden_hour":
            time_of_day = self.rng.choice([
                self.rng.uniform(6.0, 7.5),
                self.rng.uniform(17.5, 19.5),
            ])
            sun_energy = self.rng.uniform(1.0, 4.0)
        elif weather["name"] == "harsh_noon":
            time_of_day = self.rng.uniform(11.5, 13.5)
            sun_energy = self.rng.uniform(5.0, 8.0)
        elif weather["name"] == "foggy":
            sun_energy *= 0.4
            ambient_strength = self.rng.uniform(0.2, 0.4)

        sun_azimuth, sun_elevation = self._time_to_sun_angles(time_of_day)

        # Remove old lights
        self._remove_existing_lights()

        # Setup sky world shader
        self._setup_sky_world(
            turbidity=turbidity,
            sun_elevation=sun_elevation,
            sun_azimuth=sun_azimuth,
            sun_intensity=ambient_strength,
        )

        # Add sun lamp
        sun_obj = self._create_sun_lamp(
            elevation=sun_elevation,
            azimuth=sun_azimuth,
            energy=sun_energy,
            angle_deg=sun_angle,
        )

        # Optional: fill light for shadows
        fill_energy = ambient_strength * 0.5
        fill_obj = self._create_fill_light(fill_energy)

        # Foggy: add volumetric scatter
        fog_density = 0.0
        if weather["name"] == "foggy":
            fog_density = weather.get("fog_density", 0.05)
            self._add_volumetric_fog(fog_density)

        metadata = {
            "weather": weather["name"],
            "time_of_day": time_of_day,
            "sun_energy": sun_energy,
            "sun_angle_deg": sun_angle,
            "sun_elevation_deg": math.degrees(sun_elevation),
            "sun_azimuth_deg": math.degrees(sun_azimuth),
            "ambient_strength": ambient_strength,
            "turbidity": turbidity,
            "fog_density": fog_density,
        }
        return metadata

    # ------------------------------------------------------------------
    # Sky & World
    # ------------------------------------------------------------------

    def _setup_sky_world(
        self,
        turbidity: float,
        sun_elevation: float,
        sun_azimuth: float,
        sun_intensity: float,
    ) -> None:
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world

        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputWorld")
        output.location = (400, 0)

        bg = nodes.new("ShaderNodeBackground")
        bg.location = (200, 0)
        bg.inputs["Strength"].default_value = sun_intensity
        links.new(bg.outputs["Background"], output.inputs["Surface"])

        sky_tex = nodes.new("ShaderNodeTexSky")
        sky_tex.location = (-100, 0)
        sky_tex.sky_type = "NISHITA"
        sky_tex.turbidity = turbidity
        sky_tex.sun_elevation = sun_elevation
        sky_tex.sun_rotation = sun_azimuth
        sky_tex.sun_intensity = 1.0

        links.new(sky_tex.outputs["Color"], bg.inputs["Color"])

    # ------------------------------------------------------------------
    # Sun Lamp
    # ------------------------------------------------------------------

    def _create_sun_lamp(
        self,
        elevation: float,
        azimuth: float,
        energy: float,
        angle_deg: float,
    ) -> bpy.types.Object:
        bpy.ops.object.light_add(type="SUN", location=(0, -50, 50))
        sun = bpy.context.active_object
        sun.name = self.SUN_LAMP_NAME
        sun.data.energy = energy
        sun.data.angle = math.radians(angle_deg)

        # Point sun towards origin from computed direction
        sun_dir = self._angle_to_direction(elevation, azimuth)
        # Rotation: sun should point in direction (-sun_dir)
        inv_dir = (-sun_dir[0], -sun_dir[1], -sun_dir[2])
        sun.rotation_euler = self._direction_to_euler(inv_dir)

        return sun

    def _create_fill_light(self, energy: float) -> bpy.types.Object:
        bpy.ops.object.light_add(type="AREA", location=(20, -30, 30))
        fill = bpy.context.active_object
        fill.name = self.FILL_LAMP_NAME
        fill.data.energy = energy * 200.0
        fill.data.size = 10.0
        # Face the facade
        fill.rotation_euler = (math.radians(45), 0, math.radians(15))
        return fill

    # ------------------------------------------------------------------
    # Fog
    # ------------------------------------------------------------------

    def _add_volumetric_fog(self, density: float) -> None:
        """Add a volume scatter cube to simulate fog/haze."""
        bpy.ops.mesh.primitive_cube_add(size=200, location=(0, 0, 50))
        fog_cube = bpy.context.active_object
        fog_cube.name = "VolumetricFog"
        fog_cube.display_type = "WIRE"

        mat = bpy.data.materials.new("FogMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        vol_scatter = nodes.new("ShaderNodeVolumeScatter")
        vol_abs = nodes.new("ShaderNodeVolumeAbsorption")
        add_shader = nodes.new("ShaderNodeAddShader")

        vol_scatter.inputs["Density"].default_value = density
        vol_scatter.inputs["Anisotropy"].default_value = 0.3
        vol_abs.inputs["Density"].default_value = density * 0.2
        vol_abs.inputs["Color"].default_value = (0.9, 0.9, 0.95, 1.0)

        links.new(vol_scatter.outputs["Volume"], add_shader.inputs[0])
        links.new(vol_abs.outputs["Volume"], add_shader.inputs[1])
        links.new(add_shader.outputs["Shader"], output.inputs["Volume"])

        fog_cube.data.materials.append(mat)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_weather_condition(self) -> Dict[str, Any]:
        weather_list = self.cfg["weather_conditions"]
        probs = [w["probability"] for w in weather_list]
        total = sum(probs)
        r = self.rng.uniform(0, total)
        cumulative = 0.0
        for w in weather_list:
            cumulative += w["probability"]
            if r <= cumulative:
                return w
        return weather_list[-1]

    def _time_to_sun_angles(
        self, time_of_day: float
    ) -> Tuple[float, float]:
        """Convert time of day (hours) to sun elevation and azimuth (radians)."""
        # Approximate: solar noon at 12:00, elevation follows sin curve
        hour_angle = (time_of_day - 12.0) * 15.0  # degrees
        declination = 23.45  # spring equinox approximate

        # Simplified: elevation based on hour angle
        elevation_deg = max(0, 90.0 - abs(hour_angle) * 0.8)
        azimuth_deg = (180.0 + hour_angle) % 360.0

        return math.radians(elevation_deg), math.radians(azimuth_deg)

    def _angle_to_direction(
        self, elevation: float, azimuth: float
    ) -> Tuple[float, float, float]:
        x = math.cos(elevation) * math.sin(azimuth)
        y = math.cos(elevation) * math.cos(azimuth)
        z = math.sin(elevation)
        return (x, y, z)

    def _direction_to_euler(
        self, direction: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        from mathutils import Vector
        vec = Vector(direction).normalized()
        # Point -Z axis towards direction
        pitch = math.asin(-vec.z)
        yaw = math.atan2(vec.x, vec.y)
        return (pitch + math.pi / 2, 0.0, yaw)

    def _remove_existing_lights(self) -> None:
        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj, do_unlink=True)
