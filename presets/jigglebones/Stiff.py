"""
Stiff preset
For stiff objects
High angle elasticity to prevent bending.
High root and length elasticity to keep it from stretching.
Reduced gravity to make it resist forces more.
High friction for reduced oscillation
"""

import bpy
b = bpy.context.active_pose_bone

b.jiggle_root_elasticity = 0.95
b.jiggle_angle_elasticity = 0.95
b.jiggle_length_elasticity = 0.95
b.jiggle_elasticity_soften = 0.0
b.jiggle_gravity = 0.5
b.jiggle_blend = 1.0
b.jiggle_air_drag = 0
b.jiggle_friction = 0.5
