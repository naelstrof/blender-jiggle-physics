"""
Breast preset
High length and root elasticity to prevent stretching.
Zero gravity as breasts are almost always posed as if they're already under the effects of gravity.
No air drag so they don't lag behind the character.
"""

import bpy
b = bpy.context.active_pose_bone

b.jiggle_root_elasticity = 0.9
b.jiggle_angle_elasticity = 0.8
b.jiggle_length_elasticity = 0.9
b.jiggle_elasticity_soften = 0.0
b.jiggle_gravity = 0.0
b.jiggle_blend = 1.0
b.jiggle_air_drag = 0.0
b.jiggle_friction = 0.1
