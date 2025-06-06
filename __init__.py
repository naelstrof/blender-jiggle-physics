import bpy, math, cProfile, pstats, gpu
from mathutils import Vector, Matrix, Euler, Quaternion, geometry
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

ZERO_VEC = Vector((0,0,0))
IDENTITY_MAT = Matrix.Identity(4)
IDENTITY_QUAT = Quaternion()
# We merge bones that are closer than this as bones perfectly on top of each other don't work well with jiggle physics.
MERGE_BONE_THRESHOLD = 0.01

_profiler = cProfile.Profile()
jiggle_overlay_handler = None
area_pose_overlay = {}
area_simulation_overlay = {}
jiggle_physics_resetting = False
jiggle_object_virtual_point_cache = {}
jiggle_scene_virtual_point_cache = None

class JiggleSettings:
    def __init__(self, angle_elasticity, length_elasticity, elasticity_soften, gravity, blend, air_drag, friction, collision_radius):
        self.angle_elasticity = angle_elasticity
        self.length_elasticity = length_elasticity
        self.elasticity_soften = elasticity_soften
        self.gravity = gravity
        self.blend = blend
        self.air_drag = air_drag
        self.friction = friction
        self.collision_radius = collision_radius
    @classmethod
    def from_bone(cls, bone):
        return cls(bone.jiggle_angle_elasticity, bone.jiggle_length_elasticity, bone.jiggle_elasticity_soften, bone.jiggle_gravity, bone.jiggle_blend, bone.jiggle_air_drag, bone.jiggle_friction, bone.jiggle_collision_radius)

STATIC_JIGGLE_SETTINGS = JiggleSettings(1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.1)

class VirtualParticle:
    def read(self):
        self.obj_world_matrix = self.obj.matrix_world
        match self.particleType:
            case 'normal':
                self.position = self.bone.jiggle.position1.copy()
                self.position_last = self.bone.jiggle.position_last1
                self.rest_pose_position = self.bone.jiggle.rest_pose_position1
                self.pose = (self.obj_world_matrix@self.bone.head)
                self.working_position = self.position.copy()
                self.jiggle_settings = JiggleSettings.from_bone(self.bone) if not self.static else STATIC_JIGGLE_SETTINGS
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
                self.working_position = self.pose.copy()
                self.parent_pose = diff+self.pose
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
                self.working_position = self.pose.copy()
                self.jiggle_settings = JiggleSettings.from_bone(self.bone) if not self.static else STATIC_JIGGLE_SETTINGS
        if self.parent:
            self.parent_pose = self.parent.pose
        self.desired_length_to_parent = max((self.pose - self.parent_pose).length, MERGE_BONE_THRESHOLD)

    def __init__(self, obj, bone, particleType, static=False):
        self.obj = obj
        self.obj_world_matrix = obj.matrix_world
        self.bone = bone
        self.particleType = particleType
        self.static = static
        self.parent = None
        self.pose = ZERO_VEC
        self.parent_pose = ZERO_VEC
        self.rolling_error = IDENTITY_QUAT
        self.desired_length_to_parent = 1
        self.children = []
        self.jiggle_settings = None
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
        self.working_position = self.position + velocity * (1.0-self.parent.jiggle_settings.air_drag) + local_space_velocity * (1.0-self.parent.jiggle_settings.friction) + gravity * self.parent.jiggle_settings.gravity * dt2

    def mesh_collide(self, collider, depsgraph):
        collider_matrix = collider.matrix_world
        local_working_position = collider_matrix.inverted() @ self.working_position
        result, local_location, local_normal, _ = collider.closest_point_on_mesh(local_working_position, depsgraph=depsgraph)
        if not result:
            return
        location = collider_matrix @ local_location
        normal = collider_matrix.to_quaternion() @ local_normal
        diff = self.working_position-location

        local_radius = self.parent.jiggle_settings.collision_radius
        bone_matrix_world = (self.bone.id_data.matrix_world @ self.bone.matrix)
        world_radius = sum(bone_matrix_world.to_scale()) / 3.0 * local_radius

        if (diff).length > world_radius:
            return
        self.working_position = location + diff.normalized() * world_radius

    def empty_collide(self, collider):
        collider_matrix = collider.matrix_world

        local_radius = self.parent.jiggle_settings.collision_radius
        bone_matrix_world = (self.bone.id_data.matrix_world @ self.bone.matrix)
        world_radius = sum(bone_matrix_world.to_scale()) / 3.0 * local_radius

        world_vec = (self.working_position-collider_matrix.translation).normalized()*world_radius;
        local_vec = collider_matrix.inverted().to_3x3() @ world_vec

        local_working_position = collider_matrix.inverted() @ self.working_position
        local_radius = local_vec.length

        diff = local_working_position
        empty_radius = 1.0
        if diff.length-local_radius > empty_radius:
            return
        local_working_position = diff.normalized() * (empty_radius+local_radius)
        self.working_position = collider_matrix @ local_working_position

    def solve_collisions(self, depsgraph):
        if not self.bone.jiggle.collider_type:
            return

        if self.bone.jiggle.collider_type == 'Object':
            collider = self.bone.jiggle.collider
            if not collider:
                return
            if collider.type == 'MESH':
                self.mesh_collide(collider, depsgraph)
            if collider.type == 'EMPTY':
                self.empty_collide(collider)
        else:
            collider_collection = self.bone.jiggle.collider_collection
            if not collider_collection:
                return
            for collider in collider_collection.objects:
                if collider.type == 'MESH':
                    self.mesh_collide(collider, depsgraph)
                if collider.type == 'EMPTY':
                    self.empty_collide(collider)

    def constrain(self, depsgraph):
        if not self.parent:
            return

        # constrain angle
        parent_aim_pose = (self.parent_pose - self.parent.parent_pose).normalized()
        if not self.parent.parent:
            parent_aim = (self.parent.working_position - self.parent.parent_pose).normalized()
        else:
            parent_aim = (self.parent.working_position - self.parent.parent.working_position).normalized()

        current_length = (self.working_position - self.parent.working_position).length
        from_to_rot = parent_aim_pose.rotation_difference(parent_aim)
        current_pose_dir = (self.pose - self.parent_pose).normalized()
        constraintTarget = from_to_rot @ (current_pose_dir * current_length)

        error = (self.working_position - (self.parent.working_position + constraintTarget)).length
        error /= self.desired_length_to_parent
        error = min(error, 1.0)
        error = pow(error, self.parent.jiggle_settings.elasticity_soften*self.parent.jiggle_settings.elasticity_soften)
        self.working_position = self.working_position.lerp(self.parent.working_position + constraintTarget, self.parent.jiggle_settings.angle_elasticity * self.parent.jiggle_settings.angle_elasticity * error)

        # collisions
        self.solve_collisions(depsgraph)

        # constrain length
        length_elasticity = self.parent.jiggle_settings.length_elasticity * self.parent.jiggle_settings.length_elasticity
        if self.bone.bone.use_connect:
            length_elasticity = 1
        diff = self.working_position - self.parent.working_position
        dir = diff.normalized()
        self.working_position = self.working_position.lerp(self.parent.working_position + dir * self.desired_length_to_parent, length_elasticity)

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
    global jiggle_object_virtual_point_cache
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
        if bone.jiggle.mode == 'root':
            visit(bone)
    jiggle_object_virtual_point_cache[obj] = virtual_particles_cache
    return virtual_particles_cache

def get_virtual_particles(scene):
    global jiggle_scene_virtual_point_cache
    if jiggle_scene_virtual_point_cache is not None:
        return jiggle_scene_virtual_point_cache 
    jiggle_scene_virtual_point_cache = []
    jiggle_objs = [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.jiggle.enable and not obj.jiggle.mute and not obj.jiggle.freeze]
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

def flatten(mat):
    return [mat[j][i] for i in range(4) for j in range(4)]

def reset_bone(b):
    head_pos = (b.id_data.matrix_world@b.head)
    tail_pos = (b.id_data.matrix_world@b.tail)

    b.jiggle.rest_pose_position0 = b.jiggle.position0 = b.jiggle.position_last0 = head_pos + (head_pos-tail_pos)
    b.jiggle.rest_pose_position1 = b.jiggle.position1 = b.jiggle.position_last1 = head_pos
    b.jiggle.rest_pose_position2 = b.jiggle.position2 = b.jiggle.position_last2 = tail_pos

# TODO: This is kinda nasty, bones recursively propagate-- to prevent infinite recursion we use a simple global flag.
jiggle_propagating = False
def update_pose_bone_jiggle_prop(self,context,prop): 
    global jiggle_propagating
    if jiggle_propagating:
        return
    jiggle_propagating = True
    auto_key = bpy.context.scene.tool_settings.use_keyframe_insert_auto
    for b in context.selected_pose_bones:
        if b == self:
            continue
        if getattr(b,prop) == getattr(self,prop):
            continue
        setattr(b, prop, getattr(self,prop))
        if auto_key and prop in ['jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction']:
            b.keyframe_insert(data_path=prop, index=-1)
    jiggle_propagating = False

def mark_jiggle_tree(obj):
    if not obj or obj.type != 'ARMATURE':
        return
    jiggle_clear_cache()
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
    global jiggle_propagating
    if jiggle_propagating:
        return
    jiggle_propagating = True
    for b in context.selected_pose_bones:
        if getattr(b.jiggle,prop) == getattr(self,prop):
            continue
        setattr(b.jiggle, prop, getattr(self,prop))
        if prop == 'enable':
            reset_bone(b)
    if prop == 'enable':
        self.id_data.jiggle.enable = True
        mark_jiggle_tree(self.id_data)
    jiggle_propagating = False

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
    if bpy.context.scene.jiggle.debug: _profiler.enable()
    if not bpy.context.scene.jiggle.enable:
        if bpy.context.scene.jiggle.debug: _profiler.disable()
        return
    if not bpy.context.area or not any(space.type == 'VIEW_3D' and space.overlay.show_overlays for space in bpy.context.area.spaces):
        if bpy.context.scene.jiggle.debug: _profiler.disable()
        return
    do_pose = area_pose_overlay.get(bpy.context.area.as_pointer(), False)
    do_simulation = area_simulation_overlay.get(bpy.context.area.as_pointer(), False)

    if not do_pose and not do_simulation:
        if bpy.context.scene.jiggle.debug: _profiler.disable()
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
    if bpy.context.scene.jiggle.debug: _profiler.disable()
        
@persistent                
def jiggle_post(scene,depsgraph):
    global jiggle_physics_resetting
    jiggle_clear_cache() # can't cache between frames or blender will crash due to PoseBones being destroyed :sob:
    if jiggle_physics_resetting:
        return
    if scene.jiggle.debug: _profiler.enable()
    jiggle = scene.jiggle
    objects = scene.objects

    if not scene.jiggle.enable or jiggle.is_rendering:
        if scene.jiggle.debug: _profiler.disable()
        return

    if (jiggle.lastframe == scene.frame_current):
        virtual_particles = get_virtual_particles(scene)
        for particle in virtual_particles:
            particle.apply_pose()
        if scene.jiggle.debug: _profiler.disable()
        return

    lastframe = jiggle.lastframe
    frame_start, frame_end, frame_current = scene.frame_start, scene.frame_end, scene.frame_current
    frame_is_preroll = jiggle.is_preroll
    frame_loop = jiggle.loop

    if (frame_current == frame_start) and not frame_loop and not frame_is_preroll:
        jiggle_reset(bpy.context)
        if scene.jiggle.debug: _profiler.disable()
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
    dt = 1.0 / scene.render.fps
    dt2 = dt*dt
    accumulatedFrames = frames_elapsed

    virtual_particles = get_virtual_particles(scene)
    for _ in range(accumulatedFrames):
        for particle in virtual_particles:
            particle.verlet_integrate(dt2, scene.gravity)
        for particle in virtual_particles:
            particle.constrain(depsgraph)
        for particle in virtual_particles:
            particle.finish_step()
    for particle in virtual_particles:
        particle.apply_pose()
        particle.write()

    if scene.jiggle.debug: _profiler.disable()

def collider_poll(self, object):
    return object.type == 'MESH' or object.type == 'EMPTY'

@persistent        
def jiggle_render_pre(scene):
    scene.jiggle.is_rendering = True
    
@persistent
def jiggle_render_post(scene):
    scene.jiggle.is_rendering = False
    
@persistent
def jiggle_render_cancel(scene):
    scene.jiggle.is_rendering = False
    
@persistent
def jiggle_load(scene):
    s = bpy.context.scene
    s.jiggle.is_rendering = False
            
class ARMATURE_OT_JiggleCopy(bpy.types.Operator):
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
            other_bone.jiggle_angle_elasticity = bone.jiggle_angle_elasticity
            other_bone.jiggle_length_elasticity = bone.jiggle_length_elasticity
            other_bone.jiggle_elasticity_soften = bone.jiggle_elasticity_soften
            other_bone.jiggle_gravity = bone.jiggle_gravity
            other_bone.jiggle_blend = bone.jiggle_blend
            other_bone.jiggle_air_drag = bone.jiggle_air_drag
            other_bone.jiggle_friction = bone.jiggle_friction
        return {'FINISHED'}

def jiggle_clear_cache():
    global jiggle_object_virtual_point_cache, jiggle_scene_virtual_point_cache
    jiggle_scene_virtual_point_cache = None
    jiggle_object_virtual_point_cache.clear()

def jiggle_reset(context):
    jiggle_clear_cache()
    jiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle.enable and not obj.jiggle.mute]
    for ob in jiggle_objs:
        mark_jiggle_tree(ob)
        for bone in ob.pose.bones:
            reset_bone(bone)
    context.scene.jiggle.lastframe = context.scene.frame_current

class SCENE_OT_JiggleToggleProfiler(bpy.types.Operator):
    """Toggle the jiggle profiler"""
    bl_idname = "scene.jiggle_toggle_profiler"
    bl_label = "Toggle Jiggle Profiler"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable
    
    def execute(self,context):
        context.scene.jiggle.debug = not context.scene.jiggle.debug
        return {'FINISHED'}

class VIEW3D_OT_JiggleTogglePoseOverlay(bpy.types.Operator):
    """Toggle the detected rest pose overlay"""
    bl_idname = "view3d.jiggle_toggle_pose_overlay"
    bl_label = "Toggle Jiggle Rest Pose Overlay"
    
    @classmethod
    def poll(cls,context):
        return context.area.type == 'VIEW_3D' and len(context.area.spaces)>0
    
    def execute(self,context):
        current = area_pose_overlay.get(context.area.as_pointer(), False)
        if not current:
            area_pose_overlay[context.area.as_pointer()] = True
        else:
            area_pose_overlay[context.area.as_pointer()] = False

        global jiggle_overlay_handler
        if jiggle_overlay_handler:
            bpy.types.SpaceView3D.draw_handler_remove(jiggle_overlay_handler, 'WINDOW')

        # FIXME: This doesn't handle areas being destroyed
        for area in area_pose_overlay:
            if area:
                jiggle_overlay_handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_VIEW')
                break
        if jiggle_overlay_handler is None:
            for area in area_simulation_overlay:
                if area:
                    jiggle_overlay_handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_VIEW')
                    break
        context.area.tag_redraw()
        return {'FINISHED'}

class VIEW3D_OT_JiggleToggleSimulationOverlay(bpy.types.Operator):
    """Toggle the jiggle simulation overlay"""
    bl_idname = "view3d.jiggle_toggle_simulation_overlay"
    bl_label = "Toggle Jiggle Simulation Overlay"
    
    @classmethod
    def poll(cls,context):
        return context.area.type == 'VIEW_3D' and len(context.area.spaces)>0
    
    def execute(self,context):
        current = area_simulation_overlay.get(context.area.as_pointer(), False)
        if not current:
            area_simulation_overlay[context.area.as_pointer()] = True
        else:
            area_simulation_overlay[context.area.as_pointer()] = False

        global jiggle_overlay_handler
        if jiggle_overlay_handler:
            bpy.types.SpaceView3D.draw_handler_remove(jiggle_overlay_handler, 'WINDOW')

        jiggle_overlay_handler = None
        # FIXME: This doesn't handle areas being destroyed
        for area in area_pose_overlay:
            if area:
                jiggle_overlay_handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_VIEW')
                break
        if jiggle_overlay_handler is None:
            for area in area_simulation_overlay:
                if area:
                    jiggle_overlay_handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_VIEW')
                    break

        context.area.tag_redraw()
        return {'FINISHED'}

class SCENE_OT_JiggleReset(bpy.types.Operator):
    """Reset jiggle physics of scene, bone, or object depending on context"""
    bl_idname = "scene.jiggle_reset"
    bl_label = "Reset Physics"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.mode in ['OBJECT', 'POSE']
    
    def execute(self,context):
        frame = context.scene.frame_current
        global jiggle_physics_resetting
        jiggle_physics_resetting = True
        try:
            context.scene.frame_set(frame-1)
            jiggle_reset(context)
            context.scene.frame_set(frame)
        finally:
            jiggle_physics_resetting = False
        return {'FINISHED'}

class ANIM_OT_JiggleClearKeyframes(bpy.types.Operator):
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
            for prop in ['jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction']:
                data_path = f'pose.bones["{bone.name}"].{prop}'
                fcurves_to_remove = [fc for fc in action.fcurves if fc.data_path == data_path]
                for fc in fcurves_to_remove:
                    action.fcurves.remove(fc)
        return {'FINISHED'}

class SCENE_OT_JiggleProfile(bpy.types.Operator):
    bl_idname = "scene.jiggle_profile"
    bl_label = "Print Profiling Information to Console"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle.enable and context.scene.jiggle.debug
    
    def execute(self,context):
        pstats.Stats(_profiler).sort_stats('cumulative').print_stats(20)
        _profiler.clear()
        return {'FINISHED'}

def jiggle_select(context):
    jiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle.enable and not obj.jiggle.mute]
    for ob in jiggle_objs:
        jiggle_bones = [bone for bone in ob.pose.bones if getattr(bone.jiggle, 'enable', False)]
        for bone in jiggle_bones:
            bone.bone.select = True
    
class ARMATURE_OT_JiggleSelect(bpy.types.Operator):
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
    
class ARMATURE_OT_JiggleBake(bpy.types.Operator):
    """Bake this object's visible jiggle bones to keyframes"""
    bl_idname = "armature.jiggle_bake"
    bl_label = "Bake Jiggle"
    bl_options = {'UNDO'}
    
    @classmethod
    def poll(cls,context):
        return context.object and context.mode == 'POSE'
    
    def execute(self,context):
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
        preroll = context.scene.jiggle.preroll
        context.scene.jiggle.is_preroll = False
        bpy.ops.pose.select_all(action='DESELECT')
        jiggle_select(context)
        jiggle_reset(context)
        while preroll >= 0:
            if context.scene.jiggle.loop:
                frame = context.scene.frame_end - (preroll%duration)
                context.scene.frame_set(frame)
            else:
                context.scene.frame_set(context.scene.frame_start-preroll)
            context.scene.jiggle.is_preroll = True
            preroll -= 1
        #bake
        if bpy.app.version[0] >= 4 and bpy.app.version[1] > 0:
            bpy.ops.nla.bake(frame_start = context.scene.frame_start,
                            frame_end = context.scene.frame_end,
                            only_selected = True,
                            visual_keying = True,
                            use_current_action = context.scene.jiggle.bake_overwrite,
                            bake_types={'POSE'},
                            channel_types={'LOCATION','ROTATION','SCALE'})
        else:
            bpy.ops.nla.bake(frame_start = context.scene.frame_start,
                            frame_end = context.scene.frame_end,
                            only_selected = True,
                            visual_keying = True,
                            use_current_action = context.scene.jiggle.bake_overwrite,
                            bake_types={'POSE'})
        context.scene.jiggle.is_preroll = False
        context.object.jiggle.freeze = True
        if not context.scene.jiggle.bake_overwrite:
            context.object.animation_data.action.name = 'JiggleAction'
        return {'FINISHED'}  

class JigglePanel:
    bl_category = 'Animation'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    
    @classmethod
    def poll(cls,context):
        return context.object

def draw_jiggle_overlay_menu(self, context):
    self.layout.label(text="Jiggle Physics")
    row = self.layout.row(align=True)
    area = area_pose_overlay.get(context.area.as_pointer(), False)
    icon = 'CHECKBOX_HLT' if area else 'CHECKBOX_DEHLT'
    row.operator(VIEW3D_OT_JiggleTogglePoseOverlay.bl_idname, text="Show Rest Pose", icon=icon, emboss=False)
    area = area_simulation_overlay.get(context.area.as_pointer(), False)
    icon = 'CHECKBOX_HLT' if area else 'CHECKBOX_DEHLT'
    row.operator(VIEW3D_OT_JiggleToggleSimulationOverlay.bl_idname, text="Show Simulation", icon=icon, emboss=False)

class JIGGLE_PT_Settings(JigglePanel, bpy.types.Panel):
    bl_label = "Jiggle Physics"
        
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


class JIGGLE_OT_bone_connected_disable(bpy.types.Operator):
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

class JIGGLE_OT_bone_constraints_disable(bpy.types.Operator):
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

class JIGGLE_PT_NoKeyframesWarning(JigglePanel,bpy.types.Panel):
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

class JIGGLE_PT_BoneConstraintsWarning(JigglePanel,bpy.types.Panel):
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

class JIGGLE_PT_ConnectedBonesWarning(JigglePanel,bpy.types.Panel):
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


class JIGGLE_PT_NoKeyframesWarning(JigglePanel,bpy.types.Panel):
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

class JIGGLE_PT_MeshCollisionWarning(JigglePanel,bpy.types.Panel):
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

class JIGGLE_PT_Bone(JigglePanel,bpy.types.Panel):
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

class JIGGLE_PT_Utilities(JigglePanel,bpy.types.Panel):
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
        
class JIGGLE_PT_Bake(JigglePanel,bpy.types.Panel):
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
        layout.prop(context.scene.jiggle, 'preroll')
        layout.prop(context.scene.jiggle, 'bake_overwrite')
        row = layout.row()
        row.enabled = not context.scene.jiggle.bake_overwrite
        row.prop(context.scene.jiggle, 'bake_nla')
        layout.operator('armature.jiggle_bake')

class JiggleBone(bpy.types.PropertyGroup):
    position0: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last0: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    rest_pose_position0: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    position1: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last1: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    rest_pose_position1: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    position2: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last2: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    rest_pose_position2: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    enable: bpy.props.BoolProperty(
        name = 'Enable Bone Jiggle',
        description = "Enable jiggle on this bone", default = False,
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'enable')
    )
    mode: bpy.props.EnumProperty(
        name='Jiggle Mode',
        items=[('none','None','Not jiggled, and contains no jiggle bone children'),
               ('root','Root','A root jiggle bone'),
               ('solo','Solo','A bone without jiggled children or parents'),
               ('merge','Merge','Ignore this bone, but within it is a jiggle bone'),
               ('normal','Normal','User-driven jiggle'),
               ('tip', 'Tip', 'A tip jiggle bone, needing a forward projected particle, contains no jiggle bone children')],
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
    )
    collider_type: bpy.props.EnumProperty(
        name='Collider Type',
        items=[('Object','Object','Collide with a selected mesh'),('Collection','Collection','Collide with all meshes in selected collection')],
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'collider_type')
    )
    collider: bpy.props.PointerProperty(
        name='Collider Object', 
        description='Mesh object to collide with', 
        type=bpy.types.Object, 
        poll = collider_poll, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'collider')
    )
    collider_collection: bpy.props.PointerProperty(
        name = 'Collider Collection', 
        description='Collection to collide with', 
        type=bpy.types.Collection, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_nested_jiggle_prop(s, c, 'collider_collection')
    )
    
class JiggleScene(bpy.types.PropertyGroup):
    lastframe: bpy.props.IntProperty()
    loop: bpy.props.BoolProperty(name='Loop Physics', description='Physics continues as timeline loops', default=True)
    preroll: bpy.props.IntProperty(name = 'Preroll', description='Frames to run simulation before bake', min=0, default=30)
    is_preroll: bpy.props.BoolProperty(default=False)
    bake_overwrite: bpy.props.BoolProperty(name='Overwrite Current Action', description='Bake jiggle into current action, instead of creating a new one', default = False)
    bake_nla: bpy.props.BoolProperty(name='Current Action to NLA', description='Move existing animation on the armature into an NLA strip', default = False) 
    is_rendering: bpy.props.BoolProperty(default=False)
    show_no_keyframes_warning: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    enable: bpy.props.BoolProperty(
        name = 'Enable Scene',
        description = 'Enable jiggle on this scene',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
    )
    debug: bpy.props.BoolProperty(
        name = 'Enable debug',
        description = 'Enable profiling and debug features',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
    )

class JiggleObject(bpy.types.PropertyGroup):
    enable: bpy.props.BoolProperty(
        name = 'Enable Armature',
        description = 'Enable jiggle on this armature',
        default = False,
        options={'HIDDEN'},
        override={'LIBRARY_OVERRIDABLE'}
    )
    mute: bpy.props.BoolProperty(
        name = 'Mute Armature',
        description = 'Mute jiggle on this armature',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
    )
    freeze: bpy.props.BoolProperty(
        name = 'Freeze Jiggle',
        description = 'Jiggle Calculation frozen after baking',
        default = False,
        override={'LIBRARY_OVERRIDABLE'}
    )

def register():
    # These properties are strictly animatable properties, as nested properties cannot be animated on pose bones.
    bpy.types.PoseBone.jiggle_angle_elasticity = bpy.props.FloatProperty(
        name = 'Angle Elasticity',
        description = 'Spring angle stiffness, higher means more rigid. Also has a small effect on the bone length',
        min = 0,
        default = 0.6,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_angle_elasticity')
    )
    bpy.types.PoseBone.jiggle_length_elasticity = bpy.props.FloatProperty(
        name = 'Length Elasticity',
        description = 'Spring length stiffness, higher means more rigid to tension',
        min = 0,
        default = 0.8,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_length_elasticity')
    )
    bpy.types.PoseBone.jiggle_elasticity_soften = bpy.props.FloatProperty(
        name = 'Elasticity Soften',
        description = 'Weakens the elasticity of the bone when its closer to the target pose. Higher means more like a free-rolling-ball-socket',
        min = 0,
        default = 0,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_elasticity_soften')
    )
    bpy.types.PoseBone.jiggle_gravity = bpy.props.FloatProperty(
        name = 'Gravity',
        description = 'Multiplier for scene gravity',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_gravity')
    )
    bpy.types.PoseBone.jiggle_blend = bpy.props.FloatProperty(
        name = 'Blend',
        description = 'jiggle blend, 0 means no jiggle, 1 means full jiggle',
        min = 0,
        default = 1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_blend')
    )
    bpy.types.PoseBone.jiggle_air_drag = bpy.props.FloatProperty(
        name = 'Air Drag',
        description = 'How much the bone is slowed down by air, higher means more drag',
        min = 0,
        default = 0,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_air_drag')
    )
    bpy.types.PoseBone.jiggle_friction = bpy.props.FloatProperty(
        name = 'Friction',
        description = 'Internal friction, higher means return to rest quicker',
        min = 0,
        default = 0.1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_friction')
    )
    bpy.types.PoseBone.jiggle_collision_radius = bpy.props.FloatProperty(
        name = 'Collision Radius',
        description = 'Collision radius for use in collision detection and depenetration.',
        min = 0,
        default = 0.1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_pose_bone_jiggle_prop(s, c, 'jiggle_collision_radius')
    )
    
    
    #internal variables
    bpy.utils.register_class(JiggleBone)
    bpy.types.PoseBone.jiggle = bpy.props.PointerProperty(type=JiggleBone, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(JiggleObject)
    bpy.types.Object.jiggle = bpy.props.PointerProperty(type=JiggleObject, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(JiggleScene)
    bpy.types.Scene.jiggle = bpy.props.PointerProperty(type=JiggleScene, override={'LIBRARY_OVERRIDABLE'})

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
    bpy.utils.register_class(JIGGLE_PT_Utilities)
    bpy.utils.register_class(JIGGLE_PT_Bake)
    bpy.utils.register_class(JIGGLE_OT_bone_connected_disable)
    bpy.utils.register_class(JIGGLE_OT_bone_constraints_disable)
    bpy.utils.register_class(VIEW3D_OT_JiggleTogglePoseOverlay)
    bpy.utils.register_class(VIEW3D_OT_JiggleToggleSimulationOverlay)
    bpy.utils.register_class(SCENE_OT_JiggleToggleProfiler)

    bpy.types.VIEW3D_PT_overlay.append(draw_jiggle_overlay_menu)
    
    bpy.app.handlers.frame_change_post.append(jiggle_post)
    bpy.app.handlers.render_pre.append(jiggle_render_pre)
    bpy.app.handlers.render_post.append(jiggle_render_post)
    bpy.app.handlers.render_cancel.append(jiggle_render_cancel)
    bpy.app.handlers.load_post.append(jiggle_load)

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
    bpy.utils.unregister_class(JIGGLE_PT_Utilities)
    bpy.utils.unregister_class(JIGGLE_PT_Bake)
    bpy.utils.unregister_class(JIGGLE_OT_bone_connected_disable)
    bpy.utils.unregister_class(JIGGLE_OT_bone_constraints_disable)
    bpy.utils.unregister_class(VIEW3D_OT_JiggleTogglePoseOverlay)
    bpy.utils.unregister_class(VIEW3D_OT_JiggleToggleSimulationOverlay)
    bpy.utils.unregister_class(SCENE_OT_JiggleToggleProfiler)

    bpy.types.VIEW3D_PT_overlay.remove(draw_jiggle_overlay_menu)
    
    bpy.app.handlers.frame_change_post.remove(jiggle_post)
    bpy.app.handlers.render_pre.remove(jiggle_render_pre)
    bpy.app.handlers.render_post.remove(jiggle_render_post)
    bpy.app.handlers.render_cancel.remove(jiggle_render_cancel)
    bpy.app.handlers.load_post.remove(jiggle_load)

    global jiggle_overlay_handler
    if jiggle_overlay_handler:
        bpy.types.SpaceView3D.draw_handler_remove(jiggle_overlay_handler, 'WINDOW')
    
if __name__ == "__main__":
    register()
