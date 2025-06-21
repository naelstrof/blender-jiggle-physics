"""
Tail preset
High-friction and low angle-elasticity makes for a flowy swishy tail.
Gravity is set to 0.5 as some tails are typically posed as if they're already under the effects of gravity.
A tiny bit of elasticity softening as tails don't typically perfectly return to their original shape.
Air drag lets the tail flow behind the character like a trace of their movement.
"""

import bpy
b = bpy.context.active_pose_bone

b.jiggle_root_elasticity = 0.9
b.jiggle_angle_elasticity = 0.6
b.jiggle_length_elasticity = 0.9
b.jiggle_elasticity_soften = 0.1
b.jiggle_gravity = 0.5
b.jiggle_blend = 1.0
b.jiggle_air_drag = 0.2
b.jiggle_friction = 0.4
