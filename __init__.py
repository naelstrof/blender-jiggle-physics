import bpy, math, cProfile, pstats, gpu, os, shutil
from bpy.types import Scene, Panel, Operator, Menu, PoseBone, SpaceView3D, Object, PropertyGroup, Collection
from bpy.props import StringProperty, EnumProperty, BoolProperty, IntProperty, FloatProperty, FloatVectorProperty, PointerProperty
from bl_operators.presets import AddPresetBase
from bl_ui.utils import PresetPanel
from mathutils import Vector, Matrix, Euler, Quaternion, geometry
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

ZERO_VEC = Vector((0,0,0))
IDENTITY_MAT = Matrix.Identity(4)
IDENTITY_QUAT = Quaternion()
# We merge bones that are closer than this as bones perfectly on top of each other don't work well with jiggle physics.
MERGE_BONE_THRESHOLD = 0.01

class AreaProperties:
    def __init__(self):
        self.overlay_pose = False
        self.overlay_simulation = False

class JiggleGlobals:
    def __init__(self):
        self.is_rendering = False
        self.is_preroll = False
        self.is_animation_playing = False
        self.profiler = cProfile.Profile()
        self.area_overlays = {}
        self.physics_resetting = False
        self.jiggle_object_virtual_point_cache = {}
        self.jiggle_scene_virtual_point_cache = []
        self.jiggle_baking = False
        self.overlay_handler = None
        self.propagating_props = False
    def get_area_overlay_properties(self, area):
        area_pointer = area.as_pointer()
        if area_pointer not in self.area_overlays:
            self.area_overlays[area_pointer] = AreaProperties()
        return self.area_overlays[area_pointer]
    def update_overlay_draw_handler(self):
        if self.overlay_handler is not None:
            SpaceView3D.draw_handler_remove(self.overlay_handler, 'WINDOW')

        self.overlay_handler = None
        # FIXME: This doesn't handle areas being destroyed
        for area_pointer, area_props in self.area_overlays.items():
            if area_props.overlay_simulation or area_props.overlay_pose:
                self.overlay_handler = SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_VIEW')
                break
    def clear_per_object_caches(self):
        self.jiggle_object_virtual_point_cache.clear()
        self.jiggle_scene_virtual_point_cache.clear()
    def clear_per_frame_caches(self):
        self.jiggle_scene_virtual_point_cache.clear()
    def on_unregister(self):
        self.is_rendering = False
        self.is_preroll = False
        self.profiler = None
        self.area_overlays.clear()
        self.physics_resetting = False
        self.jiggle_object_virtual_point_cache.clear()
        self.jiggle_scene_virtual_point_cache.clear()
        self.propagating_props = False
        if self.overlay_handler is not None:
            SpaceView3D.draw_handler_remove(self.overlay_handler, 'WINDOW')


_jiggle_globals = JiggleGlobals()

class JiggleSettings:
    def __init__(self, root_elasticity, angle_elasticity, length_elasticity, elasticity_soften, gravity, blend, air_drag, friction, collision_radius):
        self.angle_elasticity = angle_elasticity
        self.length_elasticity = length_elasticity
        self.root_elasticity = root_elasticity
        self.elasticity_soften = elasticity_soften
        self.gravity = gravity
        self.blend = blend
        self.air_drag = air_drag
        self.friction = friction
        self.collision_radius = collision_radius
    @classmethod
    def from_bone(cls, bone):
        return cls(bone.jiggle_root_elasticity, bone.jiggle_angle_elasticity, bone.jiggle_length_elasticity, bone.jiggle_elasticity_soften, bone.jiggle_gravity, bone.jiggle_blend, bone.jiggle_air_drag, bone.jiggle_friction, bone.jiggle_collision_radius)

STATIC_JIGGLE_SETTINGS = JiggleSettings(1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,0.1)

def get_jiggle_settings(bone, static=False):
    if static:
        while not bone.jiggle.enable:
            if not bone.parent:
                return STATIC_JIGGLE_SETTINGS
            bone = bone.parent
        return JiggleSettings(1.0,1.0,1.0,0.0,0.0,bone.jiggle_blend,0.0,1.0,0.1)
    return JiggleSettings.from_bone(bone)

class VirtualParticle:
    def read(self):
        self.obj_world_matrix = self.obj.matrix_world
        self.bone = self.obj.pose.bones[self.bone_name]

        match self.particleType:
            case 'normal':
                self.position = self.bone.jiggle.position1.copy()
                self.position_last = self.bone.jiggle.position_last1
                self.rest_pose_position = self.bone.jiggle.rest_pose_position1
                self.pose = (self.obj_world_matrix@self.bone.head)
                self.working_position = self.position.copy()
                self.jiggle_settings = get_jiggle_settings(self.bone, self.static)
            case 'backProject':
                self.position = self.bone.jiggle.position0.copy()
                self.position_last = self.bone.jiggle.position_last0
                self.rest_pose_position = self.bone.jiggle.rest_pose_position0
                self.jiggle_settings = STATIC_JIGGLE_SETTINGS
                headpos = self.bone.head
                tailpos = self.bone.tail
                diff = (headpos-tailpos)
                if diff.length < MERGE_BONE_THRESHOLD:
                    diff = diff.normalized()*MERGE_BONE_THRESHOLD*2
                self.pose = self.obj_world_matrix@(diff+headpos)
                self.parent_pose = self.obj_world_matrix@((diff*2.0)+headpos)
                self.working_position = self.pose.copy()
            case 'forwardProject':
                self.position = self.bone.jiggle.position2.copy()
                self.position_last = self.bone.jiggle.position_last2
                self.rest_pose_position = self.bone.jiggle.rest_pose_position2
                headpos = self.bone.head
                tailpos = self.bone.tail
                diff = tailpos - headpos
                if diff.length < MERGE_BONE_THRESHOLD:
                    diff = diff.normalized()*MERGE_BONE_THRESHOLD*2
                self.pose = self.obj_world_matrix@(headpos+diff)
                self.working_position = self.position.copy()
                self.jiggle_settings = get_jiggle_settings(self.bone, self.static)
        if self.parent:
            self.parent_pose = self.parent.pose
        self.desired_length_to_parent = max((self.pose - self.parent_pose).length, MERGE_BONE_THRESHOLD)
        self.needs_collision = self.get_needs_collision()

    def __init__(self, obj, bone, particleType, static=False):
        self.obj = obj
        self.obj_world_matrix = obj.matrix_world
        self.bone = bone
        self.bone_name = bone.name
        self.particleType = particleType
        self.static = static
        self.parent = None
        self.pose = ZERO_VEC
        self.parent_pose = ZERO_VEC
        self.rolling_error = IDENTITY_QUAT
        self.desired_length_to_parent = 1
        self.children = []
        self.jiggle_settings = None
        self.desired_constrain = ZERO_VEC
        self.needs_collision = False
        self.read()

    def set_parent(self, parent):
        self.parent = parent
        parent.set_child(self)
        self.parent_pose = parent.pose
        self.desired_length_to_parent = max((self.pose - self.parent_pose).length, MERGE_BONE_THRESHOLD)

    def set_child(self, child):
        self.children.append(child)

    def write(self):
        match self.particleType:
            case 'backProject':
                self.bone.jiggle.position0 = self.position
                self.bone.jiggle.position_last0 = self.position_last
                self.bone.jiggle.rest_pose_position0 = self.rest_pose_position
            case 'normal':
                self.bone.jiggle.position1 = self.position
                self.bone.jiggle.position_last1 = self.position_last
                self.bone.jiggle.rest_pose_position1 = self.rest_pose_position
            case 'forwardProject':
                self.bone.jiggle.position2 = self.position
                self.bone.jiggle.position_last2 = self.position_last
                self.bone.jiggle.rest_pose_position2 = self.rest_pose_position

    def verlet_integrate(self, dt2, gravity):
        if not self.parent:
            return
        delta = self.position - self.position_last
        local_space_velocity = delta - (self.parent.position - self.parent.position_last)
        velocity = delta - local_space_velocity
        if self.parent.parent:
            self.working_position = self.position + velocity * (1.0-self.parent.jiggle_settings.air_drag) + local_space_velocity * (1.0-self.parent.jiggle_settings.friction) + gravity * self.parent.jiggle_settings.gravity * dt2
        else:
            self.working_position = self.position + velocity * (1.0-self.jiggle_settings.air_drag) + local_space_velocity * (1.0-self.jiggle_settings.friction) + gravity * self.jiggle_settings.gravity * dt2

    def mesh_collide(self, collider, depsgraph, position):
        collider_matrix = collider.matrix_world
        local_working_position = collider_matrix.inverted() @ position
        result, local_location, local_normal, _ = collider.closest_point_on_mesh(local_working_position, depsgraph=depsgraph)
        if not result:
            return position
        location = collider_matrix @ local_location
        normal = collider_matrix.to_quaternion() @ local_normal
        diff = position-location

        local_radius = self.parent.jiggle_settings.collision_radius
        bone_matrix_world = (self.bone.id_data.matrix_world @ self.bone.matrix)
        world_radius = sum(bone_matrix_world.to_scale()) / 3.0 * local_radius

        if (diff).length > world_radius:
            return position
        return location + diff.normalized() * world_radius

    def empty_collide(self, collider, position):
        collider_matrix = collider.matrix_world

        local_radius = self.parent.jiggle_settings.collision_radius
        bone_matrix_world = (self.bone.id_data.matrix_world @ self.bone.matrix)
        world_radius = sum(bone_matrix_world.to_scale()) / 3.0 * local_radius

        world_vec = (position-collider_matrix.translation).normalized()*world_radius;
        local_vec = collider_matrix.inverted().to_3x3() @ world_vec

        local_working_position = collider_matrix.inverted() @ position
        local_radius = local_vec.length

        diff = local_working_position
        empty_radius = 1.0
        if diff.length-local_radius > empty_radius:
            return position
        local_working_position = diff.normalized() * (empty_radius+local_radius)
        return collider_matrix @ local_working_position

    def get_needs_collision(self):
        if not self.bone.jiggle.collider_type:
            return False
        if self.bone.jiggle.collider_type == 'Object':
            if not self.bone.jiggle.collider:
                return False
        else:
            if not self.bone.jiggle.collider_collection:
                return False
        return True

    def solve_collisions(self, depsgraph, position):
        if self.bone.jiggle.collider_type == 'Object':
            collider = self.bone.jiggle.collider
            if collider.type == 'MESH':
                return self.mesh_collide(collider, depsgraph, position)
            if collider.type == 'EMPTY':
                return self.empty_collide(collider, position)
        else:
            collider_collection = self.bone.jiggle.collider_collection
            for collider in collider_collection.objects:
                if collider.type == 'MESH':
                    position = self.mesh_collide(collider, depsgraph, position)
                if collider.type == 'EMPTY':
                    position = self.empty_collide(collider, position)
        return position

    def constrain_angle(self):
        parent_aim_pose = (self.parent_pose - self.parent.parent_pose).normalized()
        if not self.parent.parent:
            parent_aim = (self.parent.desired_constrain - self.parent.parent_pose).normalized()
        else:
            parent_aim = (self.parent.desired_constrain - self.parent.parent.desired_constrain).normalized()

        current_length = (self.working_position - self.parent.desired_constrain).length
        from_to_rot = parent_aim_pose.rotation_difference(parent_aim)
        current_pose_dir = (self.pose - self.parent_pose).normalized()
        constraintTarget = from_to_rot @ (current_pose_dir * current_length)

        error = (self.working_position - (self.parent.desired_constrain + constraintTarget)).length
        error /= self.desired_length_to_parent
        error = min(error, 1.0)
        error = pow(error, self.parent.jiggle_settings.elasticity_soften*self.parent.jiggle_settings.elasticity_soften)
        self.desired_constrain = self.working_position.lerp(self.parent.desired_constrain + constraintTarget, self.parent.jiggle_settings.angle_elasticity * self.parent.jiggle_settings.angle_elasticity * error)
        return self.parent.desired_constrain + constraintTarget

    def constrain(self, depsgraph):
        if not self.parent:
            return

        if not self.parent.parent:
            self.desired_constrain = self.working_position = self.working_position.lerp(self.pose, self.jiggle_settings.root_elasticity*self.jiggle_settings.root_elasticity)

            headpos = self.bone.head
            tailpos = self.bone.tail
            diff = (headpos-tailpos)
            self.parent.desired_constrain = self.desired_constrain + (self.obj_world_matrix.to_3x3()@diff)
            return

        # constrain angle
        forward_constraint = self.constrain_angle()

        if self.needs_collision:
            self.desired_constrain = self.solve_collisions(depsgraph, self.desired_constrain)

        length_elasticity = self.parent.jiggle_settings.length_elasticity * self.parent.jiggle_settings.length_elasticity
        if self.bone.bone.use_connect:
            length_elasticity = 1

        # constrain length
        diff = self.desired_constrain - self.parent.desired_constrain
        dir = diff.normalized()
        forward_constraint = self.desired_constrain.lerp(self.parent.desired_constrain + dir * self.desired_length_to_parent, length_elasticity)
        self.desired_constrain = forward_constraint

        if not self.needs_collision or self.parent.jiggle_settings.angle_elasticity == 1 and self.parent.jiggle_settings.length_elasticity == 1:
            self.working_position = forward_constraint
            return

        if len(self.children) > 0:
            child = self.children[0]

            aim_pose = (child.pose - self.parent_pose).normalized()
            aim = (child.working_position - self.parent.working_position).normalized()
            from_to_rot = aim_pose.rotation_difference(aim)
            parent_to_self = (self.pose - self.parent_pose).normalized()
            real_length = (self.working_position - self.parent.working_position).length
            targetPos = (from_to_rot@(parent_to_self*real_length)) + self.parent.working_position

            error = (self.working_position - targetPos).length
            error /= self.desired_length_to_parent
            error = min(error, 1.0)
            error = pow(error, self.parent.jiggle_settings.elasticity_soften*self.parent.jiggle_settings.elasticity_soften)
            backward_constraint = self.working_position.lerp(targetPos, (self.parent.jiggle_settings.angle_elasticity * self.parent.jiggle_settings.angle_elasticity * error))

            child_length_elasticity = self.jiggle_settings.length_elasticity * self.jiggle_settings.length_elasticity
            if child.bone.bone.use_connect:
                child_length_elasticity = 1

            cdiff = backward_constraint - child.working_position
            cdir = cdiff.normalized()
            backward_constraint = backward_constraint.lerp(child.working_position + cdir * child.desired_length_to_parent, child_length_elasticity*0.5)
            self.working_position = forward_constraint.lerp(backward_constraint, 0.5)
        else:
            self.working_position = forward_constraint

    def finish_step(self):
        self.position_last = self.position
        self.position = self.working_position

    def apply_pose(self):
        if len(self.children) == 0:
            self.rest_pose_position = self.pose
            return
        child = self.children[0]

        inverted_obj_matrix = self.obj_world_matrix.inverted()

        local_pose = (inverted_obj_matrix@self.pose)
        local_child_pose = (inverted_obj_matrix@child.pose)
        local_child_working_position = (inverted_obj_matrix@child.working_position)
        local_working_position = (inverted_obj_matrix@self.working_position)

        self.rest_pose_position = self.pose

        if not self.parent:
            self.rolling_error = IDENTITY_QUAT
            return

        if len(self.children) == 1:
            cachedAnimatedVector = (local_child_pose - local_pose).normalized()
            simulatedVector = (local_child_working_position - local_working_position).normalized()
        else:
            cachedAnimatedVectorSum = ZERO_VEC.copy() 
            simulatedVectorSum = ZERO_VEC.copy()
            for child in self.children:
                local_child_pose = (inverted_obj_matrix@child.pose)
                local_child_working_position = (inverted_obj_matrix@child.working_position)
                cachedAnimatedVectorSum += (local_child_pose - local_pose).normalized()
                simulatedVectorSum += (local_child_working_position - local_working_position).normalized()
            cachedAnimatedVector = (cachedAnimatedVectorSum * (1.0/len(self.children))).normalized()
            simulatedVector = (simulatedVectorSum * (1.0/len(self.children))).normalized()
        animPoseToPhysicsPose = cachedAnimatedVector.rotation_difference(simulatedVector).slerp(IDENTITY_QUAT, 1-self.jiggle_settings.blend).normalized()

        loc, rot, scale = self.bone.matrix.decompose()
        if self.bone.bone.use_inherit_rotation:
            prot = self.parent.rolling_error.inverted().slerp(IDENTITY_QUAT, 1-self.jiggle_settings.blend)
        else:
            prot = IDENTITY_QUAT


        parent_pose_aim = local_pose - (inverted_obj_matrix@self.parent_pose) 
        adjusted_pose = (inverted_obj_matrix@self.parent.working_position) + (self.parent.rolling_error@parent_pose_aim)
        diff = (inverted_obj_matrix@self.working_position)-adjusted_pose

        loc = loc + (prot@diff) * self.jiggle_settings.blend
        self.bone.matrix = Matrix.Translation(loc) @ prot.to_matrix().to_4x4() @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
        self.rolling_error = self.parent.rolling_error.slerp(IDENTITY_QUAT, self.jiggle_settings.blend)@animPoseToPhysicsPose

def get_virtual_particles_obj(obj):
    global _jiggle_globals
    jiggle_object_virtual_point_cache = _jiggle_globals.jiggle_object_virtual_point_cache

    if obj in jiggle_object_virtual_point_cache:
        virtual_particles = jiggle_object_virtual_point_cache[obj]
        for particle in virtual_particles:
            particle.read()
        return jiggle_object_virtual_point_cache[obj]

    virtual_particles_cache = []
    bones = obj.pose.bones
    def visit(bone, last_particle=None):
        bone_jiggle = bone.jiggle
        static = not bone_jiggle.enable
        match bone.jiggle.mode:
            case 'normal':
                particle = VirtualParticle(obj, bone, 'normal', static)
                particle.set_parent(last_particle)
                virtual_particles_cache.append(particle)
                last_particle = particle
            case 'root':
                back_particle = VirtualParticle(obj, bone, 'backProject', static)
                virtual_particles_cache.append(back_particle)
                particle = VirtualParticle(obj, bone, 'normal', static)
                particle.set_parent(back_particle)
                virtual_particles_cache.append(particle)
                last_particle = particle
            case 'tip':
                particle = VirtualParticle(obj, bone, 'normal', static)
                particle.set_parent(last_particle)
                virtual_particles_cache.append(particle)
                tip = VirtualParticle(obj, bone, 'forwardProject', static)
                tip.set_parent(particle)
                virtual_particles_cache.append(tip)
                return
            case 'solo':
                back_particle = VirtualParticle(obj, bone, 'backProject', static)
                virtual_particles_cache.append(back_particle)
                particle = VirtualParticle(obj, bone, 'normal', static)
                particle.set_parent(back_particle)
                virtual_particles_cache.append(particle)
                tip = VirtualParticle(obj, bone, 'forwardProject', static)
                tip.set_parent(particle)
                virtual_particles_cache.append(tip)
                return
        for child in bone.children:
            if child.jiggle.mode == 'none':
                continue
            visit(child, last_particle)
    for bone in bones:
        if bone.jiggle.mode == 'root' or bone.jiggle.mode == 'solo':
            visit(bone)
    jiggle_object_virtual_point_cache[obj] = virtual_particles_cache
    return virtual_particles_cache

def get_virtual_particles(scene):
    global _jiggle_globals
    jiggle_baking = _jiggle_globals.jiggle_baking
    jiggle_scene_virtual_point_cache = _jiggle_globals.jiggle_scene_virtual_point_cache

    if len(jiggle_scene_virtual_point_cache) > 0:
        return jiggle_scene_virtual_point_cache 
    jiggle_objs = [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.jiggle.enable and not obj.jiggle.mute and (not obj.jiggle.freeze or jiggle_baking)]
    for obj in jiggle_objs:
        jiggle_scene_virtual_point_cache.extend(get_virtual_particles_obj(obj))
    return jiggle_scene_virtual_point_cache 

def lerp(a, b, t):
    return a + (b - a) * t

def is_bone_animated(armature, bone_name):
    anim_data = armature.animation_data
    if not anim_data or not anim_data.action:
        return False
    for fcurve in anim_data.action.fcurves:
        if f'pose.bones["{bone_name}"]' in fcurve.data_path:
            return True
    return False

def reset_bone(b):
    head_pos = (b.id_data.matrix_world@b.head)
    tail_pos = (b.id_data.matrix_world@b.tail)

    b.jiggle.rest_pose_position0 = b.jiggle.position0 = b.jiggle.position_last0 = head_pos + (head_pos-tail_pos)
    b.jiggle.rest_pose_position1 = b.jiggle.position1 = b.jiggle.position_last1 = head_pos
    b.jiggle.rest_pose_position2 = b.jiggle.position2 = b.jiggle.position_last2 = tail_pos

def update_pose_bone_jiggle_prop(self,context,prop): 
    global _jiggle_globals
    if _jiggle_globals.propagating_props:
        return
    if context.mode != 'POSE':
        return
    if context.selected_pose_bones is None:
        return
    def keyframe(auto_key, b, prop):
        if auto_key and prop in ['jiggle_root_elasticity', 'jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction']:
            b.keyframe_insert(data_path=prop, index=-1)
    _jiggle_globals.propagating_props = True
    try:
        auto_key = bpy.context.scene.tool_settings.use_keyframe_insert_auto
        for b in context.selected_pose_bones:
            if b == self:
                keyframe(auto_key, b, prop)
                continue
            if getattr(b,prop) == getattr(self,prop):
                keyframe(auto_key, b, prop)
                continue
            setattr(b, prop, getattr(self,prop))
            keyframe(auto_key, b, prop)
    finally:
        _jiggle_globals.propagating_props = False

def mark_jiggle_tree(obj):
    if not obj or obj.type != 'ARMATURE':
        return
    global _jiggle_globals
    _jiggle_globals.clear_per_object_caches()
    def visit(bone, last_jiggle_parent=None, promotion_queue=[]):
        jiggle_enabled = getattr(bone.jiggle, 'enable', False)
        jiggle_count = 0
        if last_jiggle_parent is None and jiggle_enabled:
            last_jiggle_parent = bone
            bone.jiggle.mode = 'root'
        elif jiggle_enabled:
            last_bone = bone
            for promoted_bone in promotion_queue:
                if (promoted_bone.head-last_bone.head).length < MERGE_BONE_THRESHOLD:
                    promoted_bone.jiggle.mode = 'merge'
                elif promoted_bone.parent and (promoted_bone.head-promoted_bone.parent.head).length < MERGE_BONE_THRESHOLD:
                    promoted_bone.jiggle.mode = 'merge'
                else:
                    promoted_bone.jiggle.mode = 'normal'
                last_bone = promoted_bone
            promotion_queue.clear()
            bone.jiggle.mode = 'normal' if len(bone.children) > 0 else 'tip'
            last_jiggle_parent = bone
            jiggle_count += 1
        elif not jiggle_enabled and last_jiggle_parent is not None:
            promotion_queue.append(bone)

        if len(bone.children) == 0:
            promotion_queue.clear()

        for child in bone.children:
            jiggle_count += visit(child, last_jiggle_parent, promotion_queue.copy())
        if jiggle_count == 0:
            if bone.jiggle.mode == 'root':
                bone.jiggle.mode = 'solo'
            elif bone.jiggle.mode == 'normal':
                bone.jiggle.mode = 'tip'
        return jiggle_count

    bones = obj.pose.bones
    for bone in bones:
        bone.jiggle.mode = 'none'
    root_bones = [b for b in bones if b.parent is None]
    for bone in root_bones:
        visit(bone, None, [])

def update_nested_jiggle_prop(self,context,prop): 
    global _jiggle_globals
    if _jiggle_globals.propagating_props:
        return
    if prop == 'enable':
        self.id_data.jiggle.enable = True
        mark_jiggle_tree(self.id_data)
    if context.mode != 'POSE':
        return
    if context.selected_pose_bones is None:
        return
    _jiggle_globals.propagating_props = True
    try:
        for b in context.selected_pose_bones:
            if getattr(b.jiggle,prop) == getattr(self,prop):
                continue
            setattr(b.jiggle, prop, getattr(self,prop))
            if prop == 'enable':
                reset_bone(b)
    finally:
        _jiggle_globals.propagating_props = False

def get_jiggle_parent(b):
    p = b.parent
    if p and getattr(p.jiggle,'enable', False):
        return p
    return None

def billboard_circle(verts, center, radius, segments=8):
    rv3d = bpy.context.region_data
    view_matrix = rv3d.view_matrix
    inv_view = view_matrix.inverted()
    right = inv_view.col[0].xyz.normalized()
    up = inv_view.col[1].xyz.normalized()
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        dir_vec = math.cos(angle) * right + math.sin(angle) * up
        verts.append(center + dir_vec * radius)
        next_angle = 2 * math.pi * ((i + 1) % segments) / segments
        next_dir_vec = math.cos(next_angle) * right + math.sin(next_angle) * up
        verts.append(center + next_dir_vec * radius)

@persistent
def draw_callback():
    global _jiggle_globals
    jiggle = bpy.context.scene.jiggle
    if jiggle.debug: _jiggle_globals.profiler.enable()
    if not jiggle.enable:
        if jiggle.debug: _jiggle_globals.profiler.disable()
        return
    if not bpy.context.area or not any(space.type == 'VIEW_3D' and space.overlay.show_overlays for space in bpy.context.area.spaces):
        if jiggle.debug: _jiggle_globals.profiler.disable()
        return

    area_properties = _jiggle_globals.get_area_overlay_properties(bpy.context.area)
    do_pose = area_properties.overlay_pose
    do_simulation = area_properties.overlay_simulation

    if not do_pose and not do_simulation:
        if jiggle.debug: _jiggle_globals.profiler.disable()
        return

    virtual_particles = get_virtual_particles(bpy.context.scene)
    verts = []
    rest_pose_overlay_verts = []
    for particle in virtual_particles:
        if particle.parent:
            rest_pose_overlay_verts.append(particle.parent.rest_pose_position)
            rest_pose_overlay_verts.append(particle.rest_pose_position)
            verts.append(particle.parent.position)
            verts.append(particle.position)
            local_radius = particle.jiggle_settings.collision_radius
            bone_matrix_world = particle.bone.id_data.matrix_world @ particle.bone.matrix
            world_radius = sum(bone_matrix_world.to_scale()) / 3.0 * local_radius
            billboard_circle(verts, particle.position, world_radius)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()

    if do_pose:
        batch1 = batch_for_shader(shader, 'LINES', {"pos": rest_pose_overlay_verts})
        shader.uniform_float("color", (0.10196078431372549, 1, 0.10196078431372549, 1))
        batch1.draw(shader)

    if do_simulation:
        batch2 = batch_for_shader(shader, 'LINES', {"pos": verts})
        shader.uniform_float("color", (0.29411764705882354, 0, 0.5725490196078431, 1))
        batch2.draw(shader)
    if jiggle.debug: _jiggle_globals.profiler.disable()
        
def jiggle_simulate(scene, depsgraph, virtual_particles, framecount):
    dt = 1.0 / scene.render.fps
    dt2 = dt*dt
    for _ in range(framecount):
        for particle in virtual_particles:
            particle.verlet_integrate(dt2, scene.gravity)
        for particle in virtual_particles:
            particle.constrain(depsgraph)
        for particle in virtual_particles:
            particle.finish_step()
    for particle in virtual_particles:
        particle.apply_pose()
        particle.write()

@persistent
def jiggle_playback_start(scene):
    global _jiggle_globals
    _jiggle_globals.is_animation_playing = True

@persistent
def jiggle_playback_end(scene):
    global _jiggle_globals
    _jiggle_globals.is_animation_playing = False

@persistent                
def jiggle_post(scene,depsgraph):
    global _jiggle_globals
    _jiggle_globals.clear_per_frame_caches()
    physics_resetting = _jiggle_globals.physics_resetting
    jiggle_scene_virtual_point_cache = _jiggle_globals.jiggle_scene_virtual_point_cache
    profiler = _jiggle_globals.profiler
    is_rendering = _jiggle_globals.is_rendering

    if physics_resetting:
        return
    objects = scene.objects
    jiggle = scene.jiggle

    if not jiggle.enable or is_rendering or (not scene.jiggle.simulate_during_scrub and not _jiggle_globals.is_animation_playing and not _jiggle_globals.jiggle_baking):
        return

    if jiggle.debug: profiler.enable()

    lastframe = jiggle.lastframe

    if (lastframe == scene.frame_current):
        try:
            virtual_particles = get_virtual_particles(scene)
        except KeyError:
            jiggle_reset(bpy.context)
            if jiggle.debug: profiler.disable()
            return
        for particle in virtual_particles:
            particle.apply_pose()
        if jiggle.debug: profiler.disable()
        return

    frame_start, frame_end, frame_current = scene.frame_start, scene.frame_end, scene.frame_current
    frame_is_preroll = _jiggle_globals.is_preroll
    frame_loop = jiggle.loop

    if (frame_current == frame_start) and not frame_loop and not frame_is_preroll:
        jiggle_reset(bpy.context)
        if jiggle.debug: profiler.disable()
        return

    if frame_current >= lastframe:
        frames_elapsed = frame_current - lastframe
    else:
        e1 = (frame_end - lastframe) + (frame_current - frame_start) + 1
        e2 = lastframe - frame_current
        frames_elapsed = min(e1,e2)

    if frames_elapsed > 4 or frame_is_preroll:
        frames_elapsed = 1

    jiggle.lastframe = frame_current
    accumulatedFrames = frames_elapsed

    try:
        virtual_particles = get_virtual_particles(scene)
    except KeyError:
        jiggle_reset(bpy.context)
        if jiggle.debug: profiler.disable()
        return

    jiggle_simulate(scene, depsgraph, virtual_particles, accumulatedFrames)

    if jiggle.debug: profiler.disable()

def collider_poll(self, object):
    return object.type == 'MESH' or object.type == 'EMPTY'

@persistent        
def jiggle_render_pre(scene):
    global _jiggle_globals
    _jiggle_globals.is_rendering = True
    
@persistent
def jiggle_render_post(scene):
    global _jiggle_globals
    _jiggle_globals.is_rendering = False
    
@persistent
def jiggle_render_cancel(scene):
    global _jiggle_globals
    _jiggle_globals.is_rendering = False
            
class ARMATURE_OT_JiggleCopy(Operator):
    """Copy active jiggle settings to selected bones"""
    bl_idname = "armature.jiggle_copy"
    bl_label = "Copy Settings to Selected"
    bl_options = {'UNDO'}
    
    @classmethod
    def poll(cls,context):
        return context.mode in ['POSE'] and context.active_pose_bone and (len(context.selected_pose_bones)>1) and getattr(context.active_pose_bone.jiggle, "enable", False)
    
    def execute(self,context):
        bone = context.active_pose_bone
        for other_bone in context.selected_pose_bones:
            if other_bone == bone: continue
            other_bone.jiggle.enable = bone.jiggle.enable
            other_bone.jiggle.collider_type = bone.jiggle.collider_type
            other_bone.jiggle.collider = bone.jiggle.collider
            other_bone.jiggle.collider_collection = bone.jiggle.collider_collection
            other_bone.jiggle_root_elasticity = bone.jiggle_root_elasticity
            other_bone.jiggle_angle_elasticity = bone.jiggle_angle_elasticity
            other_bone.jiggle_length_elasticity = bone.jiggle_length_elasticity
            other_bone.jiggle_elasticity_soften = bone.jiggle_elasticity_soften
            other_bone.jiggle_gravity = bone.jiggle_gravity
            other_bone.jiggle_blend = bone.jiggle_blend
            other_bone.jiggle_air_drag = bone.jiggle_air_drag
            other_bone.jiggle_friction = bone.jiggle_friction
        return {'FINISHED'}

def jiggle_reset(context):
    global _jiggle_globals
    _jiggle_globals.clear_per_object_caches()
    _jiggle_globals.clear_per_frame_caches()
    jiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle.enable and not obj.jiggle.mute]
    for ob in jiggle_objs:
        mark_jiggle_tree(ob)
        for bone in ob.pose.bones:
            reset_bone(bone)
    context.scene.jiggle.lastframe = context.scene.frame_current

class SCENE_OT_JiggleToggleProfiler(Operator):
    """Toggle the jiggle profiler"""
    bl_idname = "scene.jiggle_toggle_profiler"
    bl_label = "Toggle Jiggle Profiler"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable
    
    def execute(self,context):
        context.scene.jiggle.debug = not context.scene.jiggle.debug
        context.area.tag_redraw()
        return {'FINISHED'}

class VIEW3D_OT_JiggleTogglePoseOverlay(Operator):
    """Toggle the detected rest pose overlay"""
    bl_idname = "view3d.jiggle_toggle_pose_overlay"
    bl_label = "Toggle Jiggle Rest Pose Overlay"
    
    @classmethod
    def poll(cls,context):
        return context.area.type == 'VIEW_3D' and len(context.area.spaces)>0
    
    def execute(self,context):
        global _jiggle_globals
        current = _jiggle_globals.get_area_overlay_properties(context.area)
        current.overlay_pose = not current.overlay_pose
        _jiggle_globals.update_overlay_draw_handler()
        context.area.tag_redraw()
        return {'FINISHED'}

class VIEW3D_OT_JiggleToggleSimulationOverlay(Operator):
    """Toggle the jiggle simulation overlay"""
    bl_idname = "view3d.jiggle_toggle_simulation_overlay"
    bl_label = "Toggle Jiggle Simulation Overlay"
    
    @classmethod
    def poll(cls,context):
        return context.area.type == 'VIEW_3D' and len(context.area.spaces)>0
    
    def execute(self,context):
        global _jiggle_globals
        current = _jiggle_globals.get_area_overlay_properties(context.area)
        current.overlay_simulation = not current.overlay_simulation
        _jiggle_globals.update_overlay_draw_handler()
        context.area.tag_redraw()
        return {'FINISHED'}

class SCENE_OT_JiggleReset(Operator):
    """Reset jiggle physics of scene, bone, or object depending on context"""
    bl_idname = "scene.jiggle_reset"
    bl_label = "Reset Physics"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.mode in ['OBJECT', 'POSE']
    
    def execute(self,context):
        frame = context.scene.frame_current
        global _jiggle_globals
        _jiggle_globals.physics_resetting = True
        try:
            context.scene.frame_set(frame-1)
            jiggle_reset(context)
            context.scene.frame_set(frame)
        finally:
            _jiggle_globals.physics_resetting = False
        return {'FINISHED'}

class ANIM_OT_JiggleClearKeyframes(Operator):
    """Reset keyframes on jiggle parameters"""
    bl_idname = "anim.jiggle_clear_keyframes"
    bl_label = "Clear Parameter Keyframes"
    bl_description = "Remove keyframes from jiggle parameters on selected bones. This will not remove the jiggle settings themselves, just the keyframes that control them."
    bl_options = {'UNDO'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.mode in ['POSE'] and context.object and context.object.animation_data and context.object.animation_data.action
    
    def execute(self,context):
        action = context.object.animation_data.action
        for bone in context.selected_pose_bones:
            for prop in ['jiggle_root_elasticity', 'jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction', 'jiggle_collision_radius']:
                data_path = f'pose.bones["{bone.name}"].{prop}'
                fcurves_to_remove = [fc for fc in action.fcurves if fc.data_path == data_path]
                for fc in fcurves_to_remove:
                    action.fcurves.remove(fc)
        return {'FINISHED'}

class SCENE_OT_JiggleProfile(Operator):
    """Prints the execution time of the top 20 functions to the System Console"""
    bl_idname = "scene.jiggle_profile"
    bl_label = "Print Profiling Information to Console"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.scene.jiggle.debug
    
    def execute(self,context):
        global _jiggle_globals
        pstats.Stats(_jiggle_globals.profiler).sort_stats('cumulative').print_stats(20)
        _jiggle_globals.profiler.clear()
        return {'FINISHED'}

def jiggle_select(context):
    jiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle.enable and not obj.jiggle.mute]
    for ob in jiggle_objs:
        jiggle_bones = [bone for bone in ob.pose.bones if getattr(bone.jiggle, 'enable', False)]
        for bone in jiggle_bones:
            bone.bone.select = True
    
class ARMATURE_OT_JiggleSelect(Operator):
    """Select jiggle bones on selected objects in pose mode"""
    bl_idname = "armature.jiggle_select"
    bl_label = "Select Enabled"
    bl_options = {'UNDO'}
    
    @classmethod
    def poll(cls,context):
        return context.mode in ['POSE']
    
    def execute(self,context):
        jiggle_select(context)
        return {'FINISHED'}
    
class ARMATURE_OT_JiggleBake(Operator):
    """Bake this object's visible jiggle bones to keyframes"""
    bl_idname = "armature.jiggle_bake"
    bl_label = "Bake Jiggle"
    bl_options = {'UNDO'}
    
    @classmethod
    def poll(cls,context):
        return context.object and context.mode == 'POSE' and context.object.type == 'ARMATURE' and context.object.jiggle.enable and not context.object.jiggle.mute
    
    def execute(self,context):
        global _jiggle_globals
        _jiggle_globals.jiggle_baking = True

        bone_collections = context.object.data.collections
        collection_visibility = {col.name: col.is_visible for col in bone_collections}
        collection_solo = {col.name: col.is_solo for col in bone_collections}

        try:
            def push_nla():
                if context.scene.jiggle.bake_overwrite: return
                if not context.scene.jiggle.bake_nla: return
                if not context.object.animation_data: return
                if not context.object.animation_data.action: return
                action = context.object.animation_data.action
                track = context.object.animation_data.nla_tracks.new()
                track.name = action.name
                track.strips.new(action.name, int(action.frame_range[0]), action)
                
            push_nla()
            
            #preroll
            duration = context.scene.frame_end - context.scene.frame_start
            _jiggle_globals.is_preroll = False

            for col in bone_collections:
                col.is_solo = False
                col.is_visible = True

            bpy.ops.pose.select_all(action='DESELECT')
            jiggle_select(context)
            jiggle_reset(context)
            if context.scene.jiggle.loop:
                for preroll in reversed(range(context.scene.jiggle.preroll)):
                    frame = context.scene.frame_end - (preroll%duration)
                    context.scene.frame_set(frame)
                    _jiggle_globals.is_preroll = True
            else:
                context.scene.frame_set(context.scene.frame_start)
                _jiggle_globals.is_preroll = True
                virtual_particles = get_virtual_particles(context.scene)
                jiggle_simulate(context.scene, context.evaluated_depsgraph_get(), virtual_particles, context.scene.jiggle.preroll)
            #bake
            if context.scene.use_preview_range:
                frame_start = context.scene.frame_preview_start
                frame_end = context.scene.frame_preview_end
            else:
                frame_start = context.scene.frame_start
                frame_end = context.scene.frame_end
            bpy.ops.nla.bake(frame_start = frame_start,
                            frame_end = frame_end,
                            only_selected = True,
                            visual_keying = True,
                            use_current_action = context.scene.jiggle.bake_overwrite,
                            bake_types={'POSE'},
                            channel_types={'LOCATION','ROTATION','SCALE'})
            _jiggle_globals.is_preroll = False
            context.object.jiggle.freeze = True
            if not context.scene.jiggle.bake_overwrite:
                context.object.animation_data.action.name = 'JiggleAction'
        finally:
            for col in bone_collections:
                col.is_solo = collection_solo[col.name]
                col.is_visible = collection_visibility[col.name]
            _jiggle_globals.jiggle_baking = False
        return {'FINISHED'}  

class JigglePanel:
    bl_category = 'Animation'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    
    @classmethod
    def poll(cls,context):
        return context.object

def draw_jiggle_overlay_menu(self, context):
    global _jiggle_globals
    area_properties = _jiggle_globals.get_area_overlay_properties(context.area)
    self.layout.label(text="Jiggle Physics")
    row = self.layout.row(align=True)
    icon = 'CHECKBOX_HLT' if area_properties.overlay_pose else 'CHECKBOX_DEHLT'
    row.operator(VIEW3D_OT_JiggleTogglePoseOverlay.bl_idname, text="Show Rest Pose", icon=icon, emboss=False)
    icon = 'CHECKBOX_HLT' if area_properties.overlay_simulation else 'CHECKBOX_DEHLT'
    row.operator(VIEW3D_OT_JiggleToggleSimulationOverlay.bl_idname, text="Show Simulation", icon=icon, emboss=False)

class JIGGLE_PT_Settings(JigglePanel, Panel):
    bl_label = "Jiggle Physics"

    def draw_header(self,context):
        JIGGLE_PT_JiggleBonePresets.draw_panel_header(self.layout)
        
    def draw(self,context):
        row = self.layout.row()

        icon = 'HIDE_ON' if not context.scene.jiggle.enable else 'SCENE_DATA'
        row.prop(context.scene.jiggle, "enable", icon=icon, text="",emboss=False)
        if not context.scene.jiggle.enable:
            row.label(text='Scene muted.')
            return
        if not context.object.type == 'ARMATURE':
            row.label(text = ' Select armature.')
            return
        if context.object.jiggle.freeze:
            row.prop(context.object.jiggle,'freeze',icon='FREEZE',icon_only=True,emboss=False)
            row.label(text = 'Jiggle Frozen after Bake.')
            return
        icon = 'HIDE_ON' if context.object.jiggle.mute else 'ARMATURE_DATA'
        row.prop(context.object.jiggle,'mute',icon=icon,icon_only=True,invert_checkbox=True,emboss=False)
        if context.object.jiggle.mute:
            row.label(text='Armature muted.')
            return
        if not context.active_pose_bone:
            row.label(text = ' Select pose bone.')
            return


class JIGGLE_OT_bone_connected_disable(Operator):
    bl_idname = "armature.jiggle_bone_connected_disable"
    bl_label = "Disconnect Selected Bones"
    bl_description = "Connected bones ignore length elasticity, preventing them from stretching. Click this button to automatically fix"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls,context):
        return context.object and context.active_pose_bone

    def execute(self, context):
        obj = context.object
        previous_mode = obj.mode
        bpy.ops.object.mode_set(mode='EDIT')
        for pose_bone in obj.pose.bones:
            if not pose_bone.bone.select:
                continue
            edit_bone = obj.data.edit_bones.get(pose_bone.name)
            if edit_bone is None:
                continue
            if edit_bone.parent:
                edit_bone.use_connect = False

        bpy.ops.object.mode_set(mode=previous_mode)
        return {'FINISHED'}

class JIGGLE_OT_bone_constraints_disable(Operator):
    bl_idname = "armature.jiggle_bone_constraints_disable"
    bl_label = "Disable Constraints"
    bl_description = "Constraints are applied after jiggle, which can cause strange behavior. Click this button to automatically disable constraints on selected bones"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls,context):
        return context.object and context.active_pose_bone

    def execute(self, context):
        obj = context.object
        for pose_bone in obj.pose.bones:
            if not pose_bone.bone.select:
                continue
            for constraint in pose_bone.constraints:
                constraint.enabled = False
        return {'FINISHED'}

class JIGGLE_PT_NoKeyframesWarning(JigglePanel,Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.object and not context.object.jiggle.mute and context.active_pose_bone and context.active_pose_bone.jiggle.enable and not is_bone_animated(context.active_pose_bone.id_data, context.active_pose_bone.name)
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.label(text='No Keyframes Detected', icon='ERROR')
    
    def draw(self,context):
        box = self.layout.box()
        box.label(text=f'Position and rotation keyframes are used for the rest pose.')
        box.label(text=f'You can safely ignore this if you are using actions in the NLA.')

class JIGGLE_PT_BoneConstraintsWarning(JigglePanel,Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.object and not context.object.jiggle.mute and context.active_pose_bone and context.active_pose_bone.jiggle.enable and any(c.enabled for c in context.active_pose_bone.constraints)
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.label(text='Bone Constraints Detected', icon='ERROR')
    
    def draw(self,context):
        box = self.layout.box()
        box.label(text=f'Bone constraints are applied after jiggle, which can cause strange behavior.')
        box.label(text=f'Be weary that using constraints in place of parenting can also cause the jiggle pose to not work as intended.')
        box.label(text=f'Click the button below to automatically disable constraints on selected bones.')
        box.label(text=f'You can safely ignore this if you are intending to constrain the jiggle pose.')
        self.layout.operator(JIGGLE_OT_bone_constraints_disable.bl_idname, text='Disable Constraints on Selected Bones')

class JIGGLE_PT_ConnectedBonesWarning(JigglePanel,Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.object and not context.object.jiggle.mute and context.active_pose_bone and context.active_pose_bone.jiggle.enable and any(bone.bone.use_connect for bone in context.selected_pose_bones)
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.label(text='Connected Bones Detected', icon='ERROR')
    
    def draw(self,context):
        box = self.layout.box()
        box.label(text=f'Bones that are connected cannot be translated, preventing them from stretching.')
        box.label(text=f'This makes them ignore length elasticity.')
        box.label(text=f'You can safely ignore this if stretchiness is not desired.')
        self.layout.operator(JIGGLE_OT_bone_connected_disable.bl_idname, text='Disconnect Selected Bones')


class JIGGLE_PT_NoKeyframesWarning(JigglePanel,Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.object and not context.object.jiggle.mute and context.active_pose_bone and context.active_pose_bone.jiggle.enable and not is_bone_animated(context.active_pose_bone.id_data, context.active_pose_bone.name)
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.label(text='No Keyframes Detected', icon='ERROR')
    
    def draw(self,context):
        box = self.layout.box()
        box.label(text=f'Position and rotation keyframes are used for the rest pose.')
        box.label(text=f'You can safely ignore this if you are using actions in the NLA.')

class JIGGLE_PT_FrameSkippingEnabledWarning(JigglePanel,Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}

    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.scene.sync_mode == 'FRAME_DROP'
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.label(text='Frame Dropping Detected', icon='ERROR')
    
    def draw(self,context):
        box = self.layout.box()
        box.label(text=f'Playback set to Frame Dropping can cause the preview to be inaccurate.')
        box.label(text=f'Bakes will look very different.')

class JIGGLE_PT_MeshCollisionWarning(JigglePanel,Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        if not context.scene.jiggle.enable or not context.object or context.object.jiggle.mute or not context.active_pose_bone or not context.active_pose_bone.jiggle.enable:
            return False
        if context.active_pose_bone.jiggle.collider_type == 'Object':
            collider = context.active_pose_bone.jiggle.collider
            if collider and collider.type == 'MESH':
                return True
        elif context.active_pose_bone.jiggle.collider_type == 'Collection':
            collider_collection = context.active_pose_bone.jiggle.collider_collection
            if collider_collection:
                for collider in collider_collection.objects:
                    if collider.type == 'MESH':
                        return True
        return False
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.label(text='Mesh Collision Detected', icon='ERROR')
    
    def draw(self,context):
        box = self.layout.box()
        box.label(text=f'Meshes are not convex, making them inoptimal for collisions.')
        box.label(text=f'Please use scaled Empty spheres instead (Add -> Empty -> Sphere)')

class JIGGLE_PT_Bone(JigglePanel,Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.object and not context.object.jiggle.mute and context.active_pose_bone
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.prop(context.active_pose_bone.jiggle, 'enable')
    
    def draw(self,context):
        b = context.active_pose_bone
        if not b.jiggle.enable: return
    
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        def drawprops(layout,b,props):
            for p in props:
                layout.prop(b, p)
        
        col = layout.column(align=True)
        if b.jiggle.mode == 'root' or b.jiggle.mode == 'solo':
            drawprops(col,b,['jiggle_root_elasticity', 'jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction'])
        else:
            drawprops(col,b,['jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction'])
        col.separator()
        collision = False
        col.prop(b.jiggle, 'collider_type', text='Collisions')
        if b.jiggle.collider_type == 'Object':
            row = col.row(align=True)
            row.prop_search(b.jiggle, 'collider', context.scene, 'objects',text=' ')
            if b.jiggle.collider:
                if b.jiggle.collider:
                    collision = True
                else:
                    row.label(text='',icon='UNLINKED')
        else:
            row = col.row(align=True)
            row.prop_search(b.jiggle, 'collider_collection', bpy.data, 'collections', text=' ')
            if b.jiggle.collider_collection:
                if b.jiggle.collider_collection in context.scene.collection.children_recursive:
                    collision = True
                else:
                    row.label(text='',icon='UNLINKED')
            
        col = layout.column(align=True)
        drawprops(col,b,['jiggle_collision_radius'])
        layout.operator(ANIM_OT_JiggleClearKeyframes.bl_idname)

class JIGGLE_PT_Utilities(JigglePanel,Panel):
    bl_label = 'Global Jiggle Utilities'
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable
    
    def draw(self,context):
        layout = self.layout
        layout.use_property_split=True
        layout.use_property_decorate=False
        col = layout.column(align=True)
        if context.object.jiggle.enable and context.mode == 'POSE':
            col.operator(ARMATURE_OT_JiggleCopy.bl_idname)
            col.operator(ARMATURE_OT_JiggleSelect.bl_idname)
        col.operator(SCENE_OT_JiggleReset.bl_idname)
        if context.scene.jiggle.debug: col.operator('scene.jiggle_profile')
        layout.prop(context.scene.jiggle, 'loop')
        layout.prop(context.scene.jiggle, 'simulate_during_scrub')
        
class JIGGLE_PT_Bake(JigglePanel,Panel):
    bl_label = 'Bake Jiggle'
    bl_parent_id = 'JIGGLE_PT_Utilities'
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.object.jiggle.enable and context.mode == 'POSE'
    
    def draw(self,context):
        layout = self.layout
        layout.use_property_split=True
        layout.use_property_decorate=False
        jiggle = context.scene.jiggle
        layout.prop(jiggle, 'preroll')
        layout.prop(jiggle, 'bake_overwrite')
        row = layout.row()
        row.enabled = not jiggle.bake_overwrite
        row.prop(jiggle, 'bake_nla')
        layout.operator('armature.jiggle_bake')

class JIGGLE_PT_JiggleBonePresets(PresetPanel, Panel):
    bl_label = "Jiggle Bone Presets"
    preset_subdir = "jigglebones"
    preset_operator = "script.execute_preset"
    preset_add_operator = "armature.add_jigglebone_preset"

class JIGGLE_MT_JiggleBonePresets(Menu):
    bl_label = 'Jiggle Bone Presets'
    preset_subdir = 'jigglebones'
    preset_operator = 'script.execute_preset'
    draw = Menu.draw_preset

class JIGGLE_OT_AddJiggleBonePreset(AddPresetBase, Operator):
    """Saves or removes the current jiggle bone settings on the active bone as a preset."""
    bl_idname = 'armature.add_jigglebone_preset'
    bl_label = 'Add/Remove Jiggle Bone preset'
    preset_menu = 'JIGGLE_MT_JiggleBonePresets'
    bl_options = {'UNDO', 'REGISTER'}

    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.object and context.object.jiggle.enable and context.mode == 'POSE' and context.active_pose_bone and context.active_pose_bone.jiggle.enable

    # Common variable used for all preset values
    preset_defines = [
        'b = bpy.context.active_pose_bone',
    ]

    preset_values = [
        'b.jiggle_root_elasticity',
        'b.jiggle_angle_elasticity',
        'b.jiggle_length_elasticity',
        'b.jiggle_elasticity_soften',
        'b.jiggle_gravity',
        'b.jiggle_blend',
        'b.jiggle_air_drag',
        'b.jiggle_friction',
    ]

    # Directory to store the presets
    preset_subdir = 'jigglebones'

class JiggleBone(PropertyGroup):
    position0: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last0: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    rest_pose_position0: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    position1: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last1: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    rest_pose_position1: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    position2: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last2: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    rest_pose_position2: FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    enable: BoolProperty(
        name = 'Enable Bone Jiggle',
        description = "Enable jiggle on this bone", default = False,
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'enable')
    )
    mode: EnumProperty(
        name='Jiggle Mode',
        items=[('none','None','Not jiggled, and contains no jiggle bone children'),
               ('root','Root','A root jiggle bone'),
               ('solo','Solo','A bone without jiggled children or parents'),
               ('merge','Merge','Ignore this bone, but within its children is a jiggle bone'),
               ('normal','Normal','User-driven jiggle'),
               ('tip', 'Tip', 'A tip jiggle bone, needing a forward projected particle, contains no jiggle bone children')],
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
    )
    collider_type: EnumProperty(
        name='Collider Type',
        items=[('Object','Object','Collide with a selected mesh'),('Collection','Collection','Collide with all meshes in selected collection')],
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'collider_type')
    )
    collider: PointerProperty(
        name='Collider Object', 
        description='Mesh object to collide with', 
        type=Object, 
        poll = collider_poll, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'collider')
    )
    collider_collection: PointerProperty(
        name = 'Collider Collection', 
        description='Collection to collide with', 
        type=Collection, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'collider_collection')
    )
    
class JiggleScene(PropertyGroup):
    lastframe: IntProperty()
    loop: BoolProperty(name='Loop Physics', description='Physics continues as timeline loops', default=False)
    preroll: IntProperty(name = 'Preroll', description='Frames to run simulation before bake', min=0, default=30)
    bake_overwrite: BoolProperty(name='Overwrite Current Action', description='Bake jiggle into current action, instead of creating a new one', default = False)
    bake_nla: BoolProperty(name='Current Action to NLA', description='Move existing animation on the armature into an NLA strip', default = False) 
    enable: BoolProperty(
        name = 'Enable Scene',
        description = 'Enable jiggle on this scene',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
    )
    debug: BoolProperty(
        name = 'Enable debug',
        description = 'Enable profiling and debug features',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
    )
    simulate_during_scrub: BoolProperty(
        name = 'Simulate During Scrub',
        description = 'Simulate jiggle physics while scrubbing the timeline',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
    )

class JiggleObject(PropertyGroup):
    enable: BoolProperty(
        name = 'Enable Armature',
        description = 'Enable jiggle on this armature',
        default = False,
        options={'HIDDEN'},
        override={'LIBRARY_OVERRIDABLE'}
    )
    mute: BoolProperty(
        name = 'Mute Armature',
        description = 'Mute jiggle on this armature',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
    )
    freeze: BoolProperty(
        name = 'Freeze Jiggle',
        description = 'Jiggle Calculation frozen after baking',
        default = False,
        override={'LIBRARY_OVERRIDABLE'}
    )

def install_presets():
    try:
        # Path to bundled presets
        src_dir = os.path.join(os.path.dirname(__file__), "presets", "jigglebones")
        # Blender's user preset directory
        dst_dir = bpy.utils.user_resource('SCRIPTS', path="presets/jigglebones", create=True)

        # Copy each preset if it doesn't exist already
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            if not os.path.exists(dst_file):
                shutil.copyfile(src_file, dst_file)
    except Exception as e:
        print(f"Error installing default Jiggle Physics jigglebone presets: {e}")

def register():
    install_presets()
    # These properties are strictly animatable properties, as nested properties cannot be animated on pose bones.
    PoseBone.jiggle_angle_elasticity = FloatProperty(
        name = 'Angle Elasticity',
        description = 'Spring angle stiffness, higher means more rigid. Also has a small effect on the bone length',
        min = 0,
        default = 0.6,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_angle_elasticity')
    )
    PoseBone.jiggle_length_elasticity = FloatProperty(
        name = 'Length Elasticity',
        description = 'Spring length stiffness, higher means more rigid to tension',
        min = 0,
        default = 0.8,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_length_elasticity')
    )
    PoseBone.jiggle_root_elasticity = FloatProperty(
        name = 'Root Elasticity',
        description = 'Elasticity of the root bone, higher means more rigid to tension',
        min = 0,
        default = 0.8,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_root_elasticity')
    )
    PoseBone.jiggle_elasticity_soften = FloatProperty(
        name = 'Elasticity Soften',
        description = 'Weakens the elasticity of the bone when its closer to the target pose. Higher means more like a free-rolling-ball-socket',
        min = 0,
        default = 0,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_elasticity_soften')
    )
    PoseBone.jiggle_gravity = FloatProperty(
        name = 'Gravity',
        description = 'Multiplier for scene gravity',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_gravity')
    )
    PoseBone.jiggle_blend = FloatProperty(
        name = 'Blend',
        description = 'jiggle blend, 0 means no jiggle, 1 means full jiggle',
        min = 0,
        default = 1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_blend')
    )
    PoseBone.jiggle_air_drag = FloatProperty(
        name = 'Air Drag',
        description = 'How much the bone is slowed down by air, higher means more drag',
        min = 0,
        default = 0,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_air_drag')
    )
    PoseBone.jiggle_friction = FloatProperty(
        name = 'Friction',
        description = 'Internal friction, higher means return to rest quicker',
        min = 0,
        default = 0.1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_friction')
    )
    PoseBone.jiggle_collision_radius = FloatProperty(
        name = 'Collision Radius',
        description = 'Collision radius for use in collision detection and depenetration.',
        min = 0,
        default = 0.1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_collision_radius')
    )
    
    
    #internal variables
    bpy.utils.register_class(JiggleBone)
    PoseBone.jiggle = PointerProperty(type=JiggleBone, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(JiggleObject)
    Object.jiggle = PointerProperty(type=JiggleObject, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(JiggleScene)
    Scene.jiggle = PointerProperty(type=JiggleScene, override={'LIBRARY_OVERRIDABLE'})

    bpy.utils.register_class(SCENE_OT_JiggleReset)
    bpy.utils.register_class(ANIM_OT_JiggleClearKeyframes)
    bpy.utils.register_class(SCENE_OT_JiggleProfile)
    bpy.utils.register_class(ARMATURE_OT_JiggleCopy)
    bpy.utils.register_class(ARMATURE_OT_JiggleSelect)
    bpy.utils.register_class(ARMATURE_OT_JiggleBake)
    bpy.utils.register_class(JIGGLE_PT_Settings)
    bpy.utils.register_class(JIGGLE_PT_Bone)
    bpy.utils.register_class(JIGGLE_PT_NoKeyframesWarning)
    bpy.utils.register_class(JIGGLE_PT_ConnectedBonesWarning)
    bpy.utils.register_class(JIGGLE_PT_BoneConstraintsWarning)
    bpy.utils.register_class(JIGGLE_PT_MeshCollisionWarning)
    bpy.utils.register_class(JIGGLE_PT_FrameSkippingEnabledWarning)
    bpy.utils.register_class(JIGGLE_PT_Utilities)
    bpy.utils.register_class(JIGGLE_PT_Bake)
    bpy.utils.register_class(JIGGLE_OT_bone_connected_disable)
    bpy.utils.register_class(JIGGLE_OT_bone_constraints_disable)
    bpy.utils.register_class(VIEW3D_OT_JiggleTogglePoseOverlay)
    bpy.utils.register_class(VIEW3D_OT_JiggleToggleSimulationOverlay)
    bpy.utils.register_class(SCENE_OT_JiggleToggleProfiler)
    bpy.utils.register_class(JIGGLE_PT_JiggleBonePresets)
    bpy.utils.register_class(JIGGLE_MT_JiggleBonePresets)
    bpy.utils.register_class(JIGGLE_OT_AddJiggleBonePreset)


    bpy.types.VIEW3D_PT_overlay.append(draw_jiggle_overlay_menu)
    
    bpy.app.handlers.frame_change_post.append(jiggle_post)
    bpy.app.handlers.render_pre.append(jiggle_render_pre)
    bpy.app.handlers.render_post.append(jiggle_render_post)
    bpy.app.handlers.render_cancel.append(jiggle_render_cancel)
    bpy.app.handlers.animation_playback_pre.append(jiggle_playback_start)
    bpy.app.handlers.animation_playback_post.append(jiggle_playback_end)

def unregister():
    bpy.utils.unregister_class(JiggleBone)
    bpy.utils.unregister_class(JiggleObject)
    bpy.utils.unregister_class(JiggleScene)
    bpy.utils.unregister_class(SCENE_OT_JiggleReset)
    bpy.utils.unregister_class(ANIM_OT_JiggleClearKeyframes)
    bpy.utils.unregister_class(SCENE_OT_JiggleProfile)
    bpy.utils.unregister_class(ARMATURE_OT_JiggleCopy)
    bpy.utils.unregister_class(ARMATURE_OT_JiggleSelect)
    bpy.utils.unregister_class(ARMATURE_OT_JiggleBake)
    bpy.utils.unregister_class(JIGGLE_PT_Settings)
    bpy.utils.unregister_class(JIGGLE_PT_Bone)
    bpy.utils.unregister_class(JIGGLE_PT_NoKeyframesWarning)
    bpy.utils.unregister_class(JIGGLE_PT_ConnectedBonesWarning)
    bpy.utils.unregister_class(JIGGLE_PT_BoneConstraintsWarning)
    bpy.utils.unregister_class(JIGGLE_PT_MeshCollisionWarning)
    bpy.utils.unregister_class(JIGGLE_PT_FrameSkippingEnabledWarning)
    bpy.utils.unregister_class(JIGGLE_PT_Utilities)
    bpy.utils.unregister_class(JIGGLE_PT_Bake)
    bpy.utils.unregister_class(JIGGLE_OT_bone_connected_disable)
    bpy.utils.unregister_class(JIGGLE_OT_bone_constraints_disable)
    bpy.utils.unregister_class(VIEW3D_OT_JiggleTogglePoseOverlay)
    bpy.utils.unregister_class(VIEW3D_OT_JiggleToggleSimulationOverlay)
    bpy.utils.unregister_class(SCENE_OT_JiggleToggleProfiler)
    bpy.utils.unregister_class(JIGGLE_PT_JiggleBonePresets)
    bpy.utils.unregister_class(JIGGLE_MT_JiggleBonePresets)
    bpy.utils.unregister_class(JIGGLE_OT_AddJiggleBonePreset)

    bpy.types.VIEW3D_PT_overlay.remove(draw_jiggle_overlay_menu)
    
    bpy.app.handlers.frame_change_post.remove(jiggle_post)
    bpy.app.handlers.render_pre.remove(jiggle_render_pre)
    bpy.app.handlers.render_post.remove(jiggle_render_post)
    bpy.app.handlers.render_cancel.remove(jiggle_render_cancel)
    bpy.app.handlers.animation_playback_pre.remove(jiggle_playback_start)
    bpy.app.handlers.animation_playback_post.remove(jiggle_playback_end)

    global _jiggle_globals
    _jiggle_globals.on_unregister()
    
if __name__ == "__main__":
    register()
