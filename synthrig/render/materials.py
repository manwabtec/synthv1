"""Small material helpers for scenes (bpy)."""
from __future__ import annotations

import bpy


def principled(name: str, color=(0.5, 0.5, 0.5), roughness=0.6, metallic=0.0,
               noise_scale: float = 0.0, noise_contrast: float = 0.0, noise_seed: float = 0.0):
    """Principled BSDF, optionally with procedural noise modulating base colour.
    Texture matters: stereo matching needs it, and laser dots need something to land on."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    if noise_scale > 0 and noise_contrast > 0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = (noise_seed * 13.1, noise_seed * 7.7, noise_seed * 3.3)
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = noise_scale
        nz.inputs["Detail"].default_value = 8.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        lo = [max(c * (1 - noise_contrast), 0.0) for c in color[:3]]
        hi = [min(c * (1 + noise_contrast), 1.0) for c in color[:3]]
        ramp.color_ramp.elements[0].color = (*lo, 1)
        ramp.color_ramp.elements[1].color = (*hi, 1)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m
