"""Shared railway asset library.

profiles.py and classes.py are renderer-agnostic (no bpy) so the layout tool can use the
same geometry; the bpy builders (wheel.py, track.py, materials.py) are imported explicitly.
"""
