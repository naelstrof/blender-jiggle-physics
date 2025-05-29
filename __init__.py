import bpy, math, cProfile, pstats, gpu
from mathutils import Vector, Matrix, Euler, Quaternion, geometry
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

ZERO_VEC = Vector((0,0,0))
ONE_VEC = Vector((1,1,1))
IDENTITY_MAT = Matrix.Identity(4)
IDENTITY_QUAT = Quaternion()

_profiler = cProfile.Profile()

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
    b.jiggle.position = b.jiggle.position_last = b.jiggle.virtual_position = b.jiggle.virtual_position_last = (b.id_data.matrix_world @ b.matrix).translation
        
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

def verlet_integrate(b, position, position_last, parent_position, parent_position_last, dt, dt2, gravity):
    delta = position - position_last
    local_space_velocity = delta - (parent_position - parent_position_last)
    velocity = delta - local_space_velocity
    return position + velocity * (1.0-b.jiggle_air_drag) + local_space_velocity * (1.0-b.jiggle_friction) + gravity * b.jiggle_gravity * dt2

def constrain_length(b, working_position, parent_working_position, desired_length):
    length_elasticity = b.jiggle_length_elasticity * b.jiggle_length_elasticity
    if b.bone.use_connect:
        length_elasticity = 1
    diff = working_position - parent_working_position
    dir = diff.normalized()
    return working_position.lerp(parent_working_position + dir * desired_length, length_elasticity)

def constrain(bone_pose_world, b, p, working_position):
    pw = p.jiggle

    parent_pose = b.jiggle.parent_pose

    length_to_parent = (bone_pose_world - parent_pose).length

    # constrain angle
    parent_aim_pose = (b.jiggle.parent_pose - pw.parent_pose).normalized()
    pp = get_jiggle_parent(p)
    if not pp:
        parent_aim = (pw.working_position - pw.parent_position).normalized()
    else:
        parent_aim = (pw.working_position - pp.jiggle.working_position).normalized()
    from_to_rot = parent_aim_pose.rotation_difference(parent_aim)
    current_pose = bone_pose_world - parent_pose
    constraintTarget = from_to_rot @ current_pose

    error = (working_position - (pw.working_position + constraintTarget)).length
    error /= length_to_parent
    error = min(error, 1.0)
    error = pow(error, b.jiggle_elasticity_soften*b.jiggle_elasticity_soften)
    working_position = working_position.lerp(pw.working_position + constraintTarget, b.jiggle_angle_elasticity * b.jiggle_angle_elasticity * error)

    # todo collisions here

    return constrain_length(b, working_position, pw.working_position, length_to_parent)

@persistent
def draw_callback():
    if not bpy.context.scene.jiggle_enable or not bpy.context.scene.jiggle_debug:
        return
    jiggle_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable and not obj.jiggle_mute and not obj.jiggle_freeze and obj.visible_get()]
    for ob in jiggle_objs:
        bones = [bone for bone in ob.pose.bones if getattr(bone, 'jiggle_enable', False)]
        coords = []
        for b in bones:
            p = get_jiggle_parent(b)
            if not p:
                coords.append(b.jiggle.root_position)
                coords.append(b.jiggle.working_position)
                c = get_child(b)
                if not c:
                    coords.append(b.jiggle.working_position)
                    coords.append(b.jiggle.virtual_working_position)
                continue
            coords.append(p.jiggle.working_position)
            coords.append(b.jiggle.working_position)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        shader.bind()
        shader.uniform_float("color", (1, 0, 0, 1))
        batch.draw(shader)

@persistent
def draw_callback_pose():
    if not bpy.context.scene.jiggle_enable or not bpy.context.scene.jiggle_debug:
        return
    jiggle_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable and not obj.jiggle_mute and not obj.jiggle_freeze and obj.visible_get()]
    for ob in jiggle_objs:
        bones = [bone for bone in ob.pose.bones if getattr(bone, 'jiggle_enable', False)]
        coords = []
        for b in bones:
            draw_apply_pose(b, get_child(b), coords)
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


def apply_pose(ob, bones, virtualbones):
    for bone in bones:
        child = get_child(bone)
        p = get_jiggle_parent(bone)
        if child:
            bone_pose = bone.matrix.translation
            child_pose = child.matrix.translation

            bw = bone.jiggle
            cw = child.jiggle
            child_working_position = cw.working_position
            if bpy.context.scene.jiggle_debug:
                bw.debug = ob.matrix_world@bone_pose
                cw.debug = ob.matrix_world@child_pose
        else:
            bone_pose = bone.matrix.translation
            child_pose = bone.tail

            bw = bone.jiggle
            child_working_position = bw.virtual_working_position
            if bpy.context.scene.jiggle_debug:
                bw.debug = ob.matrix_world@bone_pose


        cachedAnimatedVector = (child_pose - bone_pose).normalized()
        simulatedVector = ((ob.matrix_world.inverted()@child_working_position) - (ob.matrix_world.inverted()@bw.working_position)).normalized()
        animPoseToPhysicsPose = cachedAnimatedVector.rotation_difference(simulatedVector).slerp(IDENTITY_QUAT, 1-bone.jiggle_blend).normalized()

        bone.jiggle.bone_length_change = (child_working_position - bw.working_position).length - (child_pose - bone_pose).length

        if p:
            loc, rot, scale = bone.matrix.decompose()
            if bone.bone.use_inherit_rotation:
                prot = p.jiggle.rolling_error.inverted()
            else:
                prot = IDENTITY_QUAT
            dir = (loc - p.matrix.translation).normalized()
            loc = loc + dir * lerp(0,p.jiggle.bone_length_change, bone.jiggle_blend)
            new_matrix = Matrix.Translation(loc) @ prot.to_matrix().to_4x4() @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
            bone.matrix = new_matrix
            bw.rolling_error = animPoseToPhysicsPose
        else:
            diff = (ob.matrix_world.inverted()@bw.working_position)-(ob.matrix_world.inverted()@bw.root_position)
            diff = diff.lerp(ZERO_VEC, 1-bone.jiggle_blend)
            loc, rot, scale = bone.matrix.decompose()
            new_matrix = Matrix.Translation(loc+diff) @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
            bone.matrix = new_matrix 
            bw.rolling_error = animPoseToPhysicsPose
    for bone in virtualbones: # apply final pose for tips
        bone_pose = bone.matrix.translation
        child_pose = bone.tail
        bw = bone.jiggle
        bw.bone_length_change = (bw.virtual_working_position - bw.working_position).length - (child_pose - bone_pose).length
        p = get_jiggle_parent(bone)
        if not p:
            continue
        cachedAnimatedVector = (child_pose - bone_pose).normalized()
        simulatedVector = ((ob.matrix_world.inverted()@bone.jiggle.virtual_working_position) - (ob.matrix_world.inverted()@bone.jiggle.working_position)).normalized()
        animPoseToPhysicsPose = cachedAnimatedVector.rotation_difference(simulatedVector).slerp(Quaternion(), 1-bone.jiggle_blend).normalized()

        loc, rot, scale = bone.matrix.decompose()
        if bone.bone.use_inherit_rotation:
            prot = p.jiggle.rolling_error.inverted()
        else:
            prot = IDENTITY_QUAT
        dir = (loc - p.matrix.translation).normalized()
        loc = loc + dir * lerp(0,p.jiggle.bone_length_change, bone.jiggle_blend)
        new_matrix = Matrix.Translation(loc) @ prot.to_matrix().to_4x4() @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
        bone.matrix = new_matrix
        bw.rolling_error = animPoseToPhysicsPose

@persistent                
def jiggle_post(scene,dg):
    if scene.jiggle_debug: _profiler.enable()
    jiggle = scene.jiggle
    objects = scene.objects

    if not scene.jiggle_enable or jiggle.is_rendering:
        if scene.jiggle_debug: _profiler.disable()
        return

    if (jiggle.lastframe == scene.frame_current):
        jiggle_objs = [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable and not obj.jiggle_mute and not obj.jiggle_freeze and obj.visible_get()]
        for ob in jiggle_objs:
            bones = [bone for bone in ob.pose.bones if getattr(bone, 'jiggle_enable', False)]
            virtualbones = [bone for bone in bones if get_child(bone) is None]
            apply_pose(ob, bones, virtualbones)
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

    jiggle_objs = [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.jiggle_enable and not obj.jiggle_mute and not obj.jiggle_freeze and obj.visible_get()]
    for ob in jiggle_objs:
        pose_bones = ob.pose.bones
        bones = [bone for bone in pose_bones if getattr(bone, 'jiggle_enable', False)]
        virtualbones = [bone for bone in bones if get_child(bone) is None]
        for _ in range(accumulatedFrames):
            for b in bones: # Do some caching
                p = get_jiggle_parent(b)
                if not p: # root bone caching
                    fixed_anim_position = (ob.matrix_world @ b.matrix).translation
                    # kinda hacky, I store the desired position of the root bone in the virtual working position.
                    b.jiggle.root_position = fixed_anim_position
                    c = get_child(b)
                    if not c:
                        b.jiggle.parent_pose = 2 * fixed_anim_position - (ob.matrix_world @ b.tail)
                        b.jiggle.parent_position = b.jiggle.parent_pose
                    else:
                        b.jiggle.parent_pose = 2 * fixed_anim_position - (ob.matrix_world @ c.head)
                        b.jiggle.parent_position = b.jiggle.parent_pose
                else:
                    b.jiggle.parent_pose = (ob.matrix_world @ p.matrix).translation
                    b.jiggle.parent_position = p.jiggle.working_position

            for b in bones:
                p = get_jiggle_parent(b)
                if p:
                    b.jiggle.working_position = verlet_integrate(b, b.jiggle.position, b.jiggle.position_last, p.jiggle.position, p.jiggle.position_last, dt, dt2, scene.gravity)
                else: # root bone verlet, we treat the virtual position as the desired position motion
                    b.jiggle.working_position = verlet_integrate(b, b.jiggle.position, b.jiggle.position_last, b.jiggle.root_position, b.jiggle.root_position_last, dt, dt2, scene.gravity)
            for b in virtualbones: # virtual bones are just tips, we use the bone itself as the parent.
                b.jiggle.virtual_working_position = verlet_integrate(b, b.jiggle.virtual_position, b.jiggle.virtual_position_last, b.jiggle.position, b.jiggle.position_last, dt, dt2, scene.gravity)

            for b in bones:
                pw = get_jiggle_parent(b)
                if pw:
                    b.jiggle.working_position = constrain((ob.matrix_world @ b.matrix).translation, b, pw, b.jiggle.working_position)
                else:
                    b.jiggle.working_position = constrain_length(b, b.jiggle.working_position, b.jiggle.root_position, 0)
            for b in virtualbones:
                pw = get_jiggle_parent(b)
                if pw:
                    b.jiggle.virtual_working_position = constrain(ob.matrix_world @ b.tail, b, pw, b.jiggle.virtual_working_position)
                else:
                    pose_diff = (ob.matrix_world@b.tail)-(ob.matrix_world@b.head)
                    pose_length = pose_diff.length
                    b.jiggle.virtual_working_position = b.jiggle.virtual_working_position.lerp(pose_diff+b.jiggle.working_position, b.jiggle_angle_elasticity * b.jiggle_angle_elasticity)
                    real_diff = b.jiggle.virtual_working_position - b.jiggle.working_position
                    b.jiggle.virtual_working_position = b.jiggle.working_position + real_diff.normalized() * pose_length
            for b in bones:
                b.jiggle.root_position_last = b.jiggle.root_position
                b.jiggle.virtual_position_last = b.jiggle.virtual_position
                b.jiggle.virtual_position = b.jiggle.virtual_working_position
                b.jiggle.position_last = b.jiggle.position
                b.jiggle.position = b.jiggle.working_position
        apply_pose(ob, bones, virtualbones)
    if scene.jiggle_debug: _profiler.disable()

@persistent        
def jiggle_render_pre(scene):
    print("jiggle render pre")
    scene.jiggle.is_rendering = True
    
@persistent
def jiggle_render_post(scene):
    print("jiggle render post")
    scene.jiggle.is_rendering = False
    
@persistent
def jiggle_render_cancel(scene):
    print("jiggle render cancel")
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
    rolling_error: bpy.props.FloatVectorProperty(size=4, subtype='QUATERNION', override={'LIBRARY_OVERRIDABLE'})
    position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    working_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    parent_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    parent_pose: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    bone_length_change: bpy.props.FloatProperty(override={'LIBRARY_OVERRIDABLE'})
    root_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    root_position_last: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    virtual_working_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    virtual_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    virtual_position_last: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    debug: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    
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
