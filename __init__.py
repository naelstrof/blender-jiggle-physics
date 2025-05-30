import bpy, math, cProfile, pstats, gpu
from mathutils import Vector, Matrix, Euler, Quaternion, geometry
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

ZERO_VEC = Vector((0,0,0))
ONE_VEC = Vector((1,1,1))
IDENTITY_MAT = Matrix.Identity(4)
IDENTITY_QUAT = Quaternion()

_profiler = cProfile.Profile()

class VirtualParticle:
    def __init__(self, obj, bone, particleType):
        self.obj_world_matrix = obj.matrix_world
        self.bone = bone
        self.particleType = particleType
        self.parent = None
        self.child = None
        self.pose = ZERO_VEC
        self.parent_pose = ZERO_VEC
        self.rolling_error = IDENTITY_QUAT
        self.jiggle = bone.jiggle
        self.desired_length_to_parent = 1
        self.bone_length_change = 0
        match particleType:
            case 'backProject':
                self.position = bone.jiggle.position0.copy()
                self.position_last = bone.jiggle.position_last0.copy()
                self.working_position = bone.jiggle.working_position0.copy()
                self.debug = bone.jiggle.debug0.copy()
            case 'normal':
                self.position = bone.jiggle.position1.copy()
                self.position_last = bone.jiggle.position_last1.copy()
                self.working_position = bone.jiggle.working_position1.copy()
                self.debug = bone.jiggle.debug1.copy()
                self.pose = (self.obj_world_matrix@bone.head)
            case 'forwardProject':
                self.position = bone.jiggle.position2.copy()
                self.position_last = bone.jiggle.position_last2.copy()
                self.working_position = bone.jiggle.working_position2.copy()
                self.debug = bone.jiggle.debug2.copy()
                self.pose = (self.obj_world_matrix@bone.tail)

    def set_parent(self, parent):
        self.parent = parent
        parent.set_child(self)
        self.parent_pose = parent.pose
        self.desired_length_to_parent = (self.pose - self.parent_pose).length

    def set_child(self, child):
        if self.particleType == 'backProject':
            self.pose = (child.pose-(self.obj_world_matrix @ child.bone.tail))+child.pose
            self.working_position = self.pose
            self.parent_pose = (self.pose-child.pose) + self.pose
        else:
            self.child = child

    def write(self):
        match self.particleType:
            case 'backProject':
                self.bone.jiggle.position0 = self.position
                self.bone.jiggle.position_last0 = self.position_last
                self.bone.jiggle.working_position0 = self.working_position
                self.bone.jiggle.debug0 = self.debug
            case 'normal':
                self.bone.jiggle.position1 = self.position
                self.bone.jiggle.position_last1 = self.position_last
                self.bone.jiggle.working_position1 = self.working_position
                self.bone.jiggle.debug1 = self.debug
            case 'forwardProject':
                self.bone.jiggle.position2 = self.position
                self.bone.jiggle.position_last2 = self.position_last
                self.bone.jiggle.working_position2 = self.working_position
                self.bone.jiggle.debug2 = self.debug

    def verlet_integrate(self, dt2, gravity):
        if not self.parent:
            return
        delta = self.position - self.position_last
        local_space_velocity = delta - (self.parent.position - self.parent.position_last)
        velocity = delta - local_space_velocity
        self.working_position = self.position + velocity * (1.0-self.bone.jiggle_air_drag) + local_space_velocity * (1.0-self.bone.jiggle_friction) + gravity * self.bone.jiggle_gravity * dt2

    def constrain(self):
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
        error = pow(error, self.bone.jiggle_elasticity_soften*self.bone.jiggle_elasticity_soften)
        self.working_position = self.working_position.lerp(self.parent.working_position + constraintTarget, self.bone.jiggle_angle_elasticity * self.bone.jiggle_angle_elasticity * error)

        # todo collisions here

        # constrain length
        length_elasticity = self.bone.jiggle_length_elasticity * self.bone.jiggle_length_elasticity
        if self.bone.bone.use_connect:
            length_elasticity = 1
        diff = self.working_position - self.parent.working_position
        dir = diff.normalized()
        self.working_position = self.working_position.lerp(self.parent.working_position + dir * self.desired_length_to_parent, length_elasticity)

    def finish_step(self):
        self.position_last = self.position
        self.position = self.working_position

    def apply_pose(self):
        if not self.child:
            if bpy.context.scene.jiggle_debug:
                self.debug = self.pose
            return

        inverted_obj_matrix = self.obj_world_matrix.inverted()

        local_pose = (inverted_obj_matrix@self.pose)
        local_child_pose = (inverted_obj_matrix@self.child.pose)
        local_child_working_position = (inverted_obj_matrix@self.child.working_position)
        local_working_position = (inverted_obj_matrix@self.working_position)

        if bpy.context.scene.jiggle_debug:
            self.debug = self.pose
            self.child.debug = self.child.pose

        self.bone_length_change = (local_child_working_position - local_working_position).length - (local_child_pose - local_pose).length

        if not self.parent:
            return

        cachedAnimatedVector = (local_child_pose - local_pose).normalized()
        simulatedVector = (local_child_working_position - local_working_position).normalized()
        animPoseToPhysicsPose = cachedAnimatedVector.rotation_difference(simulatedVector).slerp(IDENTITY_QUAT, 1-self.bone.jiggle_blend).normalized()

        if self.parent.parent:
            loc, rot, scale = self.bone.matrix.decompose()
            if self.bone.bone.use_inherit_rotation:
                prot = self.parent.rolling_error.inverted()
            else:
                prot = IDENTITY_QUAT
            dir = (loc - self.parent.bone.head).normalized()
            loc = loc + dir * lerp(0,self.parent.bone_length_change, self.bone.jiggle_blend)
            new_matrix = Matrix.Translation(loc) @ prot.to_matrix().to_4x4() @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
            self.bone.matrix = new_matrix
            self.rolling_error = animPoseToPhysicsPose
        else:
            diff = local_working_position-local_pose
            diff = diff.lerp(ZERO_VEC, 1-self.bone.jiggle_blend)
            loc, rot, scale = self.bone.matrix.decompose()
            new_matrix = Matrix.Translation(loc+diff) @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
            self.bone.matrix = new_matrix 
            self.rolling_error = animPoseToPhysicsPose

def get_virtual_particles(scene):
    virtual_particles = []
    jiggle_objs = [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable and not obj.jiggle_mute and not obj.jiggle_freeze and obj.visible_get()]
    for ob in jiggle_objs:
        pose_bones = ob.pose.bones
        bones = [bone for bone in pose_bones if getattr(bone, 'jiggle_enable', False)]
        last_particle = None
        for bone in bones:
            if last_particle is None or get_parent(bone) != last_particle.bone:
                last_particle = VirtualParticle(ob, bone, 'backProject')
                virtual_particles.append(last_particle)
            new_particle = VirtualParticle(ob, bone, 'normal')
            new_particle.set_parent(last_particle)
            virtual_particles.append(new_particle)
            last_particle = new_particle
            child = get_child(bone)
            if not child:
                child_particle = VirtualParticle(ob, bone, 'forwardProject')
                child_particle.set_parent(last_particle)
                virtual_particles.append(child_particle)
                last_particle = None
    return virtual_particles

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

def reset_scene():
    jiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable]
    for ob in jiggle_objs:
        reset_ob(ob)
                              
def reset_ob(ob):
    jiggle_bones = [bone for bone in ob.pose.bones if getattr(bone, 'jiggle_enable', False)]
    for bone in jiggle_bones:
        reset_bone(bone)

def reset_bone(b):
    b.jiggle.working_position0 = b.jiggle.position0 = b.jiggle.position_last0 = (b.id_data.matrix_world@b.head)
    b.jiggle.working_position1 = b.jiggle.position1 = b.jiggle.position_last1 = (b.id_data.matrix_world@b.head)
    b.jiggle.working_position2 = b.jiggle.position2 = b.jiggle.position_last2 = (b.id_data.matrix_world@b.head)
        
def update_prop(self,context,prop): 
    if type(self) == bpy.types.PoseBone: 
        auto_key = bpy.context.scene.tool_settings.use_keyframe_insert_auto
        for b in context.selected_pose_bones:
            b[prop] = self[prop]
            if auto_key:
                if prop not in ['jiggle_enable', 'jiggle_mute', 'jiggle_freeze']:
                    b.keyframe_insert(data_path=prop, index=-1)
        if prop in ['jiggle_enable']:
            self.id_data.jiggle_enable = self[prop]
            for b in context.selected_pose_bones:
                reset_bone(b)
    context.scene.jiggle.is_rendering = False
        
def get_parent(b):
    return b.parent

def get_jiggle_parent(b):
    p = b.parent
    if p and getattr(p,'jiggle_enable', False):
        return p
    return None

def get_child(b):
    for child in b.children:
        if (child.jiggle_enable):
            return child
    return None

def draw_apply_pose(bone,child,coords):
    if not child:
        return
    coords.append(bone.jiggle.debug)
    coords.append(child.jiggle.debug)

@persistent
def draw_callback():
    if not bpy.context.scene.jiggle_enable or not bpy.context.scene.jiggle_debug:
        return
    virtual_particles = get_virtual_particles(bpy.context.scene)
    coords = []
    for particle in virtual_particles:
        if particle.parent:
            coords.append(particle.parent.working_position)
            coords.append(particle.working_position)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", (1, 0, 0, 1))
    batch.draw(shader)

@persistent
def draw_callback_pose():
    if not bpy.context.scene.jiggle_enable or not bpy.context.scene.jiggle_debug:
        return
    virtual_particles = get_virtual_particles(bpy.context.scene)
    coords = []
    for particle in virtual_particles:
        if particle.parent:
            coords.append(particle.parent.debug)
            coords.append(particle.debug)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", (0, 1, 0, 1))
    batch.draw(shader)
        
@persistent
def jiggle_pre(scene):
    if scene.jiggle_debug: _profiler.enable()
    if (scene.jiggle.lastframe == scene.frame_current) or scene.jiggle.is_rendering:
        if scene.jiggle_debug: _profiler.disable()
        return

    if not scene.jiggle_enable:
        reset_scene()
        if scene.jiggle_debug: _profiler.disable()
        return

    if scene.jiggle_debug: _profiler.disable()


@persistent                
def jiggle_post(scene,dg):
    if scene.jiggle_debug: _profiler.enable()
    jiggle = scene.jiggle
    objects = scene.objects

    if not scene.jiggle_enable or jiggle.is_rendering:
        if scene.jiggle_debug: _profiler.disable()
        return

    if (jiggle.lastframe == scene.frame_current):
        virtual_particles = get_virtual_particles(scene)
        for particle in virtual_particles:
            particle.apply_pose()
        if scene.jiggle_debug: _profiler.disable()
        return

    lastframe = jiggle.lastframe
    frame_start, frame_end, frame_current = scene.frame_start, scene.frame_end, scene.frame_current
    frame_is_preroll = jiggle.is_preroll
    frame_loop = jiggle.loop

    if (frame_current == frame_start) and not frame_loop and not frame_is_preroll:
        bpy.ops.jiggle.reset()
        if scene.jiggle_debug: _profiler.disable()
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
            particle.constrain()
        for particle in virtual_particles:
            particle.finish_step()
    for particle in virtual_particles:
        particle.apply_pose()
        particle.write()
    if scene.jiggle_debug: _profiler.disable()

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
            
class JiggleCopy(bpy.types.Operator):
    """Copy active jiggle settings to selected bones"""
    bl_idname = "jiggle.copy"
    bl_label = "Copy Settings to Selected"
    
    @classmethod
    def poll(cls,context):
        return context.mode in ['POSE'] and context.active_pose_bone and (len(context.selected_pose_bones)>1) and getattr(context.active_pose_bone, "jiggle_enable", False)
    
    def execute(self,context):
        bone = context.active_pose_bone
        for other_bone in context.selected_pose_bones:
            if other_bone == bone: continue
            other_bone.jiggle_enable = bone.jiggle_enable
            other_bone.jiggle_angle_elasticity = bone.jiggle_angle_elasticity
            other_bone.jiggle_length_elasticity = bone.jiggle_length_elasticity
            other_bone.jiggle_elasticity_soften = bone.jiggle_elasticity_soften
            other_bone.jiggle_gravity = bone.jiggle_gravity
            other_bone.jiggle_blend = bone.jiggle_blend
            other_bone.jiggle_air_drag = bone.jiggle_air_drag
            other_bone.jiggle_friction = bone.jiggle_friction
        return {'FINISHED'}

class JiggleReset(bpy.types.Operator):
    """Reset scene jiggle physics to rest state"""
    bl_idname = "jiggle.reset"
    bl_label = "Reset Physics"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle_enable and context.mode in ['OBJECT', 'POSE']
    
    def execute(self,context):
        jiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable and not obj.jiggle_mute]
        for ob in jiggle_objs:
            jiggle_bones = [bone for bone in ob.pose.bones if getattr(bone, 'jiggle_enable', False)]
            for bone in jiggle_bones:
                reset_bone(bone)
        context.scene.jiggle.lastframe = context.scene.frame_current
        return {'FINISHED'}

class JiggleClearKeyframes(bpy.types.Operator):
    """Reset keyframes on jiggle parameters"""
    bl_idname = "jiggle.clear_keyframes"
    bl_label = "Clear Parameter Keyframes"
    bl_description = "Remove keyframes from jiggle parameters on selected bones. This will not remove the jiggle settings themselves, just the keyframes that control them."
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle_enable and context.mode in ['POSE'] and context.object and context.object.animation_data and context.object.animation_data.action
    
    def execute(self,context):
        action = context.object.animation_data.action
        for bone in context.selected_pose_bones:
            for prop in ['jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction']:
                data_path = f'pose.bones["{bone.name}"].{prop}'
                fcurves_to_remove = [fc for fc in action.fcurves if fc.data_path == data_path]
                for fc in fcurves_to_remove:
                    action.fcurves.remove(fc)
        return {'FINISHED'}

class JiggleProfile(bpy.types.Operator):
    bl_idname = "jiggle.profile"
    bl_label = "Print Profiling Information to Console"
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle_enable and context.scene.jiggle_debug
    
    def execute(self,context):
        pstats.Stats(_profiler).sort_stats('cumulative').print_stats(20)
        _profiler.clear()
        return {'FINISHED'}
    
class JiggleSelect(bpy.types.Operator):
    """Select jiggle bones on selected objects in pose mode"""
    bl_idname = "jiggle.select"
    bl_label = "Select Enabled"
    
    @classmethod
    def poll(cls,context):
        return context.mode in ['POSE']
    
    def execute(self,context):
        bpy.ops.pose.select_all(action='DESELECT')
        jiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable and not obj.jiggle_mute]
        for ob in jiggle_objs:
            jiggle_bones = [bone for bone in ob.pose.bones if getattr(bone, 'jiggle_enable', False)]
            for bone in jiggle_bones:
                bone.bone.select = True
        return {'FINISHED'}
    
class JiggleBake(bpy.types.Operator):
    """Bake this object's visible jiggle bones to keyframes"""
    bl_idname = "jiggle.bake"
    bl_label = "Bake Jiggle"
    
    @classmethod
    def poll(cls,context):
        return context.object
    
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
        
        bpy.ops.jiggle.reset()
            
        #preroll
        duration = context.scene.frame_end - context.scene.frame_start
        preroll = context.scene.jiggle.preroll
        context.scene.jiggle.is_preroll = False
        bpy.ops.jiggle.select()
        bpy.ops.jiggle.reset()
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
        context.object.jiggle_freeze = True
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

class JIGGLE_PT_Settings(JigglePanel, bpy.types.Panel):
    bl_label = "Jiggle Physics"
        
    def draw(self,context):
        row = self.layout.row()
        icon = 'RADIOBUT_OFF' if not context.scene.jiggle_debug else 'RADIOBUT_ON'
        row.prop(context.scene, "jiggle_debug", icon=icon, text="",emboss=False)

        icon = 'HIDE_ON' if not context.scene.jiggle_enable else 'SCENE_DATA'
        row.prop(context.scene, "jiggle_enable", icon=icon, text="",emboss=False)
        if not context.scene.jiggle_enable:
            row.label(text='Scene muted.')
            return
        if not context.object.type == 'ARMATURE':
            row.label(text = ' Select armature.')
            return
#        row.label(icon='TRIA_RIGHT')
        if context.object.jiggle_freeze:
            row.prop(context.object,'jiggle_freeze',icon='FREEZE',icon_only=True,emboss=False)
            row.label(text = 'Jiggle Frozen after Bake.')
            return
        icon = 'HIDE_ON' if context.object.jiggle_mute else 'ARMATURE_DATA'
        row.prop(context.object,'jiggle_mute',icon=icon,icon_only=True,invert_checkbox=True,emboss=False)
        if context.object.jiggle_mute:
            row.label(text='Armature muted.')
            return
        if not context.active_pose_bone:
            row.label(text = ' Select pose bone.')
            return


class JIGGLE_OT_no_keyframe_tooltip_operator(bpy.types.Operator):
    bl_idname = "jiggle.no_keyframes_tooltip"
    bl_label = "No Rest Pose Detected"
    bl_description = "Keyframes are used to define the jiggle's rest pose. Please add keyframes to the bones in their rest state. You may safely ignore this warning if you are using actions in the NLA"

    def execute(self, context):
        return {'FINISHED'}

class JIGGLE_OT_connected_tooltip_operator(bpy.types.Operator):
    bl_idname = "jiggle.connected_tooltip"
    bl_label = "Connected Bone Detected"
    bl_description = "Connected bones ignore length elasticity, preventing them from stretching. Click this button to automatically fix"

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

class JIGGLE_OT_constraints_tooltip_operator(bpy.types.Operator):
    bl_idname = "jiggle.constraints_tooltip"
    bl_label = "Constraints Detected"
    bl_description = "Constraints are applied after jiggle, which can cause strange behavior. Click this button to automatically disable constraints on selected bones"

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

class JIGGLE_PT_Bone(JigglePanel,bpy.types.Panel):
    bl_label = ''
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle_enable and context.object and not context.object.jiggle_mute and context.active_pose_bone
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.prop(context.active_pose_bone, 'jiggle_enable')
    
    def draw(self,context):
        b = context.active_pose_bone
        if not b.jiggle_enable: return
    
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        def drawprops(layout,b,props):
            for p in props:
                layout.prop(b, p)
        
        col = layout.column(align=True)
        layout.operator(JiggleClearKeyframes.bl_idname)
        if not is_bone_animated(b.id_data, b.name):
            layout.operator(JIGGLE_OT_no_keyframe_tooltip_operator.bl_idname, icon='ERROR')
        for c in b.constraints:
            if c.enabled:
                layout.operator(JIGGLE_OT_constraints_tooltip_operator.bl_idname, icon='ERROR')
                break
        if b.bone.use_connect:
            layout.operator(JIGGLE_OT_connected_tooltip_operator.bl_idname, icon='ERROR')
        drawprops(col,b,['jiggle_angle_elasticity', 'jiggle_length_elasticity', 'jiggle_elasticity_soften', 'jiggle_gravity', 'jiggle_blend', 'jiggle_air_drag', 'jiggle_friction'])
            

class JIGGLE_PT_Utilities(JigglePanel,bpy.types.Panel):
    bl_label = 'Global Jiggle Utilities'
    bl_parent_id = 'JIGGLE_PT_Settings'
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle_enable
    
    def draw(self,context):
        layout = self.layout
        layout.use_property_split=True
        layout.use_property_decorate=False
        col = layout.column(align=True)
        if context.object.jiggle_enable and context.mode == 'POSE':
            col.operator('jiggle.copy')
            col.operator('jiggle.select')
        col.operator('jiggle.reset')
        if context.scene.jiggle_debug: col.operator('jiggle.profile')
        layout.prop(context.scene.jiggle, 'loop')
        
class JIGGLE_PT_Bake(JigglePanel,bpy.types.Panel):
    bl_label = 'Bake Jiggle'
    bl_parent_id = 'JIGGLE_PT_Utilities'
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(cls,context):
        return context.scene.jiggle_enable and context.object.jiggle_enable and context.mode == 'POSE'
    
    def draw(self,context):
        layout = self.layout
        layout.use_property_split=True
        layout.use_property_decorate=False
        layout.prop(context.scene.jiggle, 'preroll')
        layout.prop(context.scene.jiggle, 'bake_overwrite')
        row = layout.row()
        row.enabled = not context.scene.jiggle.bake_overwrite
        row.prop(context.scene.jiggle, 'bake_nla')
        layout.operator('jiggle.bake')

class JiggleBone(bpy.types.PropertyGroup):
    working_position0: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position0: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last0: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    debug0: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    working_position1: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position1: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last1: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    debug1: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})

    working_position2: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position2: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last2: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    debug2: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    
class JiggleScene(bpy.types.PropertyGroup):
    lastframe: bpy.props.IntProperty()
    loop: bpy.props.BoolProperty(name='Loop Physics', description='Physics continues as timeline loops', default=True)
    preroll: bpy.props.IntProperty(name = 'Preroll', description='Frames to run simulation before bake', min=0, default=0)
    is_preroll: bpy.props.BoolProperty(default=False)
    bake_overwrite: bpy.props.BoolProperty(name='Overwrite Current Action', description='Bake jiggle into current action, instead of creating a new one', default = False)
    bake_nla: bpy.props.BoolProperty(name='Current Action to NLA', description='Move existing animation on the armature into an NLA strip', default = False) 
    is_rendering: bpy.props.BoolProperty(default=False)

def register():
    global debug_handler
    global debug_handler2
    debug_handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_VIEW')
    debug_handler2 = bpy.types.SpaceView3D.draw_handler_add(draw_callback_pose, (), 'WINDOW', 'POST_VIEW')
    
    #JIGGLE TOGGLES
    
    bpy.types.Scene.jiggle_enable = bpy.props.BoolProperty(
        name = 'Enable Scene',
        description = 'Enable jiggle on this scene',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_enable')
    )
    bpy.types.Scene.jiggle_debug = bpy.props.BoolProperty(
        name = 'Enable Debug',
        description = 'Enable drawing of jiggle debug lines. Green is the detected rest pose, red is the simulated physics pose. This is slow so disable when not needed',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_debug')
    )
    bpy.types.Object.jiggle_enable = bpy.props.BoolProperty(
        name = 'Enable Armature',
        description = 'Enable jiggle on this armature',
        default = False,
        options={'HIDDEN'},
        override={'LIBRARY_OVERRIDABLE'}
    )
    bpy.types.Object.jiggle_mute = bpy.props.BoolProperty(
        name = 'Mute Armature',
        description = 'Mute jiggle on this armature',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_mute')
    )
    bpy.types.Object.jiggle_freeze = bpy.props.BoolProperty(
        name = 'Freeze Jiggle',
        description = 'Jiggle Calculation frozen after baking',
        default = False,
        override={'LIBRARY_OVERRIDABLE'}
    )
    bpy.types.PoseBone.jiggle_enable = bpy.props.BoolProperty(
        name = 'Enable Bone Jiggle',
        description = "Enable jiggle on this bone",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
        update=lambda s, c: update_prop(s, c, 'jiggle_enable')
    )
    bpy.types.PoseBone.jiggle_angle_elasticity = bpy.props.FloatProperty(
        name = 'Angle Elasticity',
        description = 'Spring angle stiffness, higher means more rigid. Also has a small effect on the bone length',
        min = 0,
        default = 0.6,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_angle_elasticity')
    )
    bpy.types.PoseBone.jiggle_length_elasticity = bpy.props.FloatProperty(
        name = 'Length Elasticity',
        description = 'Spring length stiffness, higher means more rigid to tension',
        min = 0,
        default = 0.8,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_length_elasticity')
    )
    bpy.types.PoseBone.jiggle_elasticity_soften = bpy.props.FloatProperty(
        name = 'Elasticity Soften',
        description = 'Weakens the elasticity of the bone when its closer to the target pose. Higher means more like a free-rolling-ball-socket',
        min = 0,
        default = 0,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_elasticity_soften')
    )
    bpy.types.PoseBone.jiggle_gravity = bpy.props.FloatProperty(
        name = 'Gravity',
        description = 'Multiplier for scene gravity',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_gravity')
    )
    bpy.types.PoseBone.jiggle_blend = bpy.props.FloatProperty(
        name = 'Blend',
        description = 'jiggle blend, 0 means no jiggle, 1 means full jiggle',
        min = 0,
        default = 1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_blend')
    )
    bpy.types.PoseBone.jiggle_air_drag = bpy.props.FloatProperty(
        name = 'Air Drag',
        description = 'How much the bone is slowed down by air, higher means more drag',
        min = 0,
        default = 0,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_air_drag')
    )
    bpy.types.PoseBone.jiggle_friction = bpy.props.FloatProperty(
        name = 'Friction',
        description = 'Internal friction, higher means return to rest quicker',
        min = 0,
        default = 0.1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'jiggle_friction')
    )
    
    #internal variables
    bpy.utils.register_class(JiggleBone)
    bpy.types.PoseBone.jiggle = bpy.props.PointerProperty(type=JiggleBone, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(JiggleScene)
    bpy.types.Scene.jiggle = bpy.props.PointerProperty(type=JiggleScene, override={'LIBRARY_OVERRIDABLE'})
    
    bpy.utils.register_class(JiggleReset)
    bpy.utils.register_class(JiggleClearKeyframes)
    bpy.utils.register_class(JiggleProfile)
    bpy.utils.register_class(JiggleCopy)
    bpy.utils.register_class(JiggleSelect)
    bpy.utils.register_class(JiggleBake)
    bpy.utils.register_class(JIGGLE_PT_Settings)
    bpy.utils.register_class(JIGGLE_PT_Bone)
    bpy.utils.register_class(JIGGLE_PT_Utilities)
    bpy.utils.register_class(JIGGLE_PT_Bake)
    bpy.utils.register_class(JIGGLE_OT_no_keyframe_tooltip_operator)
    bpy.utils.register_class(JIGGLE_OT_connected_tooltip_operator)
    bpy.utils.register_class(JIGGLE_OT_constraints_tooltip_operator)
    
    bpy.app.handlers.frame_change_pre.append(jiggle_pre)
    bpy.app.handlers.frame_change_post.append(jiggle_post)
    bpy.app.handlers.render_pre.append(jiggle_render_pre)
    bpy.app.handlers.render_post.append(jiggle_render_post)
    bpy.app.handlers.render_cancel.append(jiggle_render_cancel)
    bpy.app.handlers.load_post.append(jiggle_load)

def unregister():
    global debug_handler
    global debug_handler2
    bpy.utils.unregister_class(JiggleBone)
    bpy.utils.unregister_class(JiggleScene)
    bpy.utils.unregister_class(JiggleReset)
    bpy.utils.unregister_class(JiggleClearKeyframes)
    bpy.utils.unregister_class(JiggleProfile)
    bpy.utils.unregister_class(JiggleCopy)
    bpy.utils.unregister_class(JiggleSelect)
    bpy.utils.unregister_class(JiggleBake)
    bpy.utils.unregister_class(JIGGLE_PT_Settings)
    bpy.utils.unregister_class(JIGGLE_PT_Bone)
    bpy.utils.unregister_class(JIGGLE_PT_Utilities)
    bpy.utils.unregister_class(JIGGLE_PT_Bake)
    bpy.utils.unregister_class(JIGGLE_OT_no_keyframe_tooltip_operator)
    bpy.utils.unregister_class(JIGGLE_OT_connected_tooltip_operator)
    bpy.utils.unregister_class(JIGGLE_OT_constraints_tooltip_operator)
    
    bpy.app.handlers.frame_change_pre.remove(jiggle_pre)
    bpy.app.handlers.frame_change_post.remove(jiggle_post)
    bpy.app.handlers.render_pre.remove(jiggle_render_pre)
    bpy.app.handlers.render_post.remove(jiggle_render_post)
    bpy.app.handlers.render_cancel.remove(jiggle_render_cancel)
    bpy.app.handlers.load_post.remove(jiggle_load)
    bpy.types.SpaceView3D.draw_handler_remove(debug_handler, 'WINDOW')
    bpy.types.SpaceView3D.draw_handler_remove(debug_handler2, 'WINDOW')
    
if __name__ == "__main__":
    register()
