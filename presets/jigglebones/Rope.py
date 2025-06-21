"""
Rope preset
For ropes and other soft objects
Low angle elasticity, for drooping
High elasticity softening, as it doesn't retain its shape well
High gravity, to pull it down
Air drag to have it fall behind in motion, but not severely.
High friction for little to no oscillation
"""

import bpy
b = bpy.context.active_pose_bone

b.jiggle_root_elasticity = 0.9
b.jiggle_angle_elasticity = 0.55
b.jiggle_length_elasticity = 0.9
b.jiggle_elasticity_soften = 1.0
b.jiggle_gravity = 1.0
b.jiggle_blend = 1.0
b.jiggle_air_drag = 0.1
b.jiggle_friction = 0.5
