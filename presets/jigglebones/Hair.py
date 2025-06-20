"""
Hair preset
Low gravity as hair is typically posed as if it's already under the effects of gravity. Yet should point down if the head tilts.
High root and length elasticity to prevent stretching.
High friction to prevent hair from oscillating.
Non-zero air drag to let the hair flow behind the character like a trace of their movement.
Reduced blend to prevent the hair from deforming too much.
"""

import bpy
b = bpy.context.active_pose_bone

b.jiggle_root_elasticity = 0.9
b.jiggle_angle_elasticity = 0.6
b.jiggle_length_elasticity = 0.9
b.jiggle_elasticity_soften = 0
b.jiggle_gravity = 0.25
b.jiggle_blend = 0.7
b.jiggle_air_drag = 0.2
b.jiggle_friction = 0.5
