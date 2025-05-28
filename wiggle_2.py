bl_info = {
    "name": "Wiggle 2",
    "author": "Steve Miller",
    "version": (3, 0, 0),
    "blender": (3, 00, 0),
    "location": "3d Viewport > Animation Panel",
    "description": "Simulate spring-like physics on Bone transforms",
    "warning": "",
    "wiki_url": "https://github.com/shteeve3d/blender-wiggle-2",
    "category": "Animation",
}

import bpy, math
from mathutils import Vector, Matrix, Euler, Quaternion, geometry
from bpy.app.handlers import persistent
import cProfile
import pstats
import gpu
from gpu_extras.batch import batch_for_shader

ZERO_VEC = Vector((0,0,0))
ONE_VEC = Vector((1,1,1))
IDENTITY_MAT = Matrix.Identity(4)

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
    wiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.wiggle_enable]
    for ob in wiggle_objs:
        reset_ob(ob)
                              
def reset_ob(ob):
    wiggle_bones = [bone for bone in ob.pose.bones if getattr(bone, 'wiggle_enabled', False)]
    for bone in wiggle_bones:
        reset_bone(bone)

def reset_bone(b):
    b.wiggle.position = b.wiggle.position_last = b.wiggle.virtual_position = b.wiggle.virtual_position_last = (b.id_data.matrix_world @ b.matrix).translation
        
def update_prop(self,context,prop): 
    if type(self) == bpy.types.PoseBone: 
        for b in context.selected_pose_bones:
            b[prop] = self[prop]
        if prop in ['wiggle_enabled']:
            for b in context.selected_pose_bones:
                reset_bone(b)
    context.scene.wiggle.is_rendering = False
        
def get_parent(b):
    return b.parent

def get_wiggle_parent(b):
    p = b.parent
    if p and getattr(p,'wiggle_enabled', False):
        return p
    return None

def get_child(b):
    for child in b.children:
        if (child.wiggle_enabled):
            return child
    return None

def draw_apply_pose(bone,child,coords):
    if not child:
        return
    coords.append(bone.wiggle.debug)
    coords.append(child.wiggle.debug)

def verlet_integrate(b, position, position_last, parent_position, parent_position_last, dt, dt2, gravity):
    delta = position - position_last
    local_space_velocity = delta - (parent_position - parent_position_last)
    velocity = delta - local_space_velocity
    return position + velocity * (1.0-b.wiggle_air_drag) + local_space_velocity * (1.0-b.wiggle_friction) + gravity * b.wiggle_gravity * dt2

def constrain_length(b, working_position, parent_working_position, desired_length):
    length_elasticity = b.wiggle_length_elasticity * b.wiggle_length_elasticity
    if b.bone.use_connect:
        length_elasticity = 1
    diff = working_position - parent_working_position
    dir = diff.normalized()
    return working_position.lerp(parent_working_position + dir * desired_length, length_elasticity)

def constrain(bone_pose_world, b, p, working_position):
    pw = p.wiggle

    parent_pose = b.wiggle.parent_pose

    length_to_parent = (bone_pose_world - parent_pose).length

    # constrain angle
    parent_aim_pose = (b.wiggle.parent_pose - pw.parent_pose).normalized()
    pp = get_wiggle_parent(p)
    if not pp:
        parent_aim = (pw.working_position - pw.parent_position).normalized()
    else:
        parent_aim = (pw.working_position - pp.wiggle.working_position).normalized()
    from_to_rot = parent_aim_pose.rotation_difference(parent_aim)
    current_pose = bone_pose_world - parent_pose
    constraintTarget = from_to_rot @ current_pose

    error = (working_position - (pw.working_position + constraintTarget)).length
    error /= length_to_parent
    error = min(error, 1.0)
    error = pow(error, b.wiggle_elasticity_soften*b.wiggle_elasticity_soften)
    working_position = working_position.lerp(pw.working_position + constraintTarget, b.wiggle_angle_elasticity * b.wiggle_angle_elasticity * error)

    # todo collisions here

    return constrain_length(b, working_position, pw.working_position, length_to_parent)

@persistent
def draw_callback():
    if not bpy.context.scene.wiggle_enable or not bpy.context.scene.wiggle_debug:
        return
    wiggle_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE' and obj.wiggle_enable and not obj.wiggle_mute and not obj.wiggle_freeze]
    for ob in wiggle_objs:
        bones = [bone for bone in ob.pose.bones if getattr(bone, 'wiggle_enabled', False)]
        coords = []
        for b in bones:
            p = get_wiggle_parent(b)
            if not p:
                continue
            coords.append(p.wiggle.working_position)
            coords.append(b.wiggle.working_position)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        shader.bind()
        shader.uniform_float("color", (1, 0, 0, 1))
        batch.draw(shader)

@persistent
def draw_callback_pose():
    if not bpy.context.scene.wiggle_enable or not bpy.context.scene.wiggle_debug:
        return
    wiggle_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE' and obj.wiggle_enable and not obj.wiggle_mute and not obj.wiggle_freeze]
    for ob in wiggle_objs:
        bones = [bone for bone in ob.pose.bones if getattr(bone, 'wiggle_enabled', False)]
        coords = []
        for b in bones:
            draw_apply_pose(b, get_child(b), coords)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        shader.bind()
        shader.uniform_float("color", (0, 1, 0, 1))
        batch.draw(shader)
        
@persistent
def wiggle_pre(scene):
    if scene.wiggle_debug: _profiler.enable()
    if (scene.wiggle.lastframe == scene.frame_current and not scene.wiggle.reset) or scene.wiggle.is_rendering:
        if scene.wiggle_debug: _profiler.disable()
        return

    if not scene.wiggle_enable:
        reset_scene()
        if scene.wiggle_debug: _profiler.disable()
        return

    if scene.wiggle_debug: _profiler.disable()


def apply_pose(ob, bones, virtualbones):
    for bone in bones: # apply final pose
        bone.wiggle.virtual_position_last = bone.wiggle.virtual_position
        bone.wiggle.virtual_position = bone.wiggle.virtual_working_position
        bone.wiggle.position_last = bone.wiggle.position
        bone.wiggle.position = bone.wiggle.working_position

        child = get_child(bone)
        if not child:
            continue

        bone_pose = bone.matrix.translation
        child_pose = child.matrix.translation

        bone.wiggle.debug = ob.matrix_world@bone_pose
        child.wiggle.debug = ob.matrix_world@child_pose

        cachedAnimatedVector = (child_pose - bone_pose).normalized()
        simulatedVector = (ob.matrix_world.inverted()@(child.wiggle.working_position - bone.wiggle.working_position)).normalized()
        animPoseToPhysicsPose = cachedAnimatedVector.rotation_difference(simulatedVector).slerp(Quaternion(), 1-bone.wiggle_blend).normalized()
        

        bone.wiggle.bone_length_change = (child.wiggle.working_position - bone.wiggle.working_position).length - (child_pose - bone_pose).length

        p = get_wiggle_parent(bone)
        if p:
            loc, rot, scale = bone.matrix.decompose()
            if bone.bone.use_inherit_rotation:
                prot = p.wiggle.rolling_error.inverted()
            else:
                prot = Quaternion()
            dir = (loc - p.matrix.translation).normalized()
            loc = loc + dir * lerp(0,p.wiggle.bone_length_change, bone.wiggle_blend)
            new_matrix = Matrix.Translation(loc) @ prot.to_matrix().to_4x4() @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
            bone.matrix = new_matrix
            bone.wiggle.rolling_error = animPoseToPhysicsPose
        else:
            diff = ob.matrix_world.inverted()@(bone.wiggle.working_position-bone.wiggle.virtual_working_position)
            loc, rot, scale = bone.matrix.decompose()
            new_matrix = Matrix.Translation(loc+diff) @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
            bone.matrix = new_matrix 
            bone.wiggle.rolling_error = animPoseToPhysicsPose
    for bone in virtualbones: # apply final pose for tips
        bone_pose = bone.matrix.translation
        child_pose = bone.tail

        cachedAnimatedVector = (child_pose - bone_pose).normalized()
        simulatedVector = (ob.matrix_world.inverted()@(bone.wiggle.virtual_working_position - bone.wiggle.working_position)).normalized()
        animPoseToPhysicsPose = cachedAnimatedVector.rotation_difference(simulatedVector).slerp(Quaternion(), 1-bone.wiggle_blend).normalized()

        bone.wiggle.bone_length_change = (bone.wiggle.virtual_working_position - bone.wiggle.working_position).length - (child_pose - bone_pose).length

        p = get_wiggle_parent(bone)
        if not p:
            continue
        loc, rot, scale = bone.matrix.decompose()
        if bone.bone.use_inherit_rotation:
            prot = p.wiggle.rolling_error.inverted()
        else:
            prot = Quaternion()
        dir = (loc - p.matrix.translation).normalized()
        loc = loc + dir * lerp(0,p.wiggle.bone_length_change, bone.wiggle_blend)
        new_matrix = Matrix.Translation(loc) @ prot.to_matrix().to_4x4() @ animPoseToPhysicsPose.to_matrix().to_4x4() @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
        bone.matrix = new_matrix
        bone.wiggle.rolling_error = animPoseToPhysicsPose

@persistent                
def wiggle_post(scene,dg):
    if scene.wiggle_debug: _profiler.enable()
    wiggle = scene.wiggle
    objects = scene.objects

    if not scene.wiggle_enable or wiggle.is_rendering:
        if scene.wiggle_debug: _profiler.disable()
        return

    if (wiggle.lastframe == scene.frame_current) and not wiggle.reset:
        wiggle_objs = [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.wiggle_enable and not obj.wiggle_mute and not obj.wiggle_freeze]
        for ob in wiggle_objs:
            bones = [bone for bone in ob.pose.bones if getattr(bone, 'wiggle_enabled', False)]
            virtualbones = [bone for bone in bones if get_child(bone) is None]
            apply_pose(ob, bones, virtualbones)
        if scene.wiggle_debug: _profiler.disable()
        return

    lastframe = wiggle.lastframe
    frame_start, frame_end, frame_current = scene.frame_start, scene.frame_end, scene.frame_current
    frame_is_preroll = wiggle.is_preroll
    frame_loop = wiggle.loop

    if (frame_current == frame_start) and not frame_loop and not frame_is_preroll:
        bpy.ops.wiggle.reset()
        if scene.wiggle_debug: _profiler.disable()
        return

    if frame_current >= lastframe:
        frames_elapsed = frame_current - lastframe
    else:
        e1 = (frame_end - lastframe) + (frame_current - frame_start) + 1
        e2 = lastframe - frame_current
        frames_elapsed = min(e1,e2)

    if frames_elapsed > 4 or frame_is_preroll:
        frames_elapsed = 1

    wiggle.lastframe = frame_current
    dt = 1.0 / scene.render.fps
    dt2 = dt*dt
    accumulatedFrames = frames_elapsed

    wiggle_objs = [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.wiggle_enable and not obj.wiggle_mute and not obj.wiggle_freeze]
    for ob in wiggle_objs:
        bones = [bone for bone in ob.pose.bones if getattr(bone, 'wiggle_enabled', False)]
        virtualbones = [bone for bone in bones if get_child(bone) is None]
        for _ in range(accumulatedFrames):
            for b in bones: # Do some caching
                p = get_wiggle_parent(b)
                if not p: # root bone caching
                    fixed_anim_position = (ob.matrix_world @ b.matrix).translation
                    # kinda hacky, I store the desired position of the root bone in the virtual working position.
                    b.wiggle.virtual_working_position = fixed_anim_position
                    c = get_child(b)
                    if not c:
                        continue
                    b.wiggle.parent_pose = 2 * fixed_anim_position - (ob.matrix_world @ c.matrix).translation
                    b.wiggle.parent_position = b.wiggle.parent_pose
                else:
                    b.wiggle.parent_pose = (ob.matrix_world @ p.matrix).translation
                    b.wiggle.parent_position = p.wiggle.working_position

            for b in bones:
                p = get_wiggle_parent(b)
                if p:
                    b.wiggle.working_position = verlet_integrate(b, b.wiggle.position, b.wiggle.position_last, p.wiggle.position, p.wiggle.position_last, dt, dt2, scene.gravity)
                else: # root bone verlet, we treat the virtual position as the desired position motion
                    b.wiggle.working_position = verlet_integrate(b, b.wiggle.position, b.wiggle.position_last, b.wiggle.virtual_position, b.wiggle.virtual_position_last, dt, dt2, scene.gravity)
            for b in virtualbones: # virtual bones are just tips, we use the bone itself as the parent.
                b.wiggle.virtual_working_position = verlet_integrate(b, b.wiggle.virtual_position, b.wiggle.virtual_position_last, b.wiggle.position, b.wiggle.position_last, dt, dt2, scene.gravity)

            for b in bones:
                pw = get_wiggle_parent(b)
                if pw:
                    b.wiggle.working_position = constrain((ob.matrix_world @ b.matrix).translation, b, pw, b.wiggle.working_position)
                else:
                    b.wiggle.working_position = constrain_length(b, b.wiggle.working_position, b.wiggle.virtual_working_position, 0)
            for b in virtualbones:
                pw = get_wiggle_parent(b)
                if pw:
                    b.wiggle.virtual_working_position = constrain((ob.matrix_world @ Matrix.Translation(b.tail)).translation, b, pw, b.wiggle.virtual_working_position)
        apply_pose(ob, bones, virtualbones)
    if scene.wiggle_debug: _profiler.disable()

@persistent        
def wiggle_render_pre(scene):
    print("wiggle render pre")
    scene.wiggle.is_rendering = True
    
@persistent
def wiggle_render_post(scene):
    print("wiggle render post")
    scene.wiggle.is_rendering = False
    
@persistent
def wiggle_render_cancel(scene):
    print("wiggle render cancel")
    scene.wiggle.is_rendering = False
    
@persistent
def wiggle_load(scene):
    s = bpy.context.scene
    s.wiggle.is_rendering = False
            
class WiggleCopy(bpy.types.Operator):
    """Copy active wiggle settings to selected bones"""
    bl_idname = "wiggle.copy"
    bl_label = "Copy Settings to Selected"
    
    @classmethod
    def poll(cls,context):
        return context.mode in ['POSE'] and context.active_pose_bone and (len(context.selected_pose_bones)>1) and getattr(context.active_pose_bone, "wiggle_enabled", False)
    
    def execute(self,context):
        bone = context.active_pose_bone
        for other_bone in context.selected_pose_bones:
            if other_bone == bone: continue
            other_bone.wiggle_enabled = bone.wiggle_enabled
            other_bone.wiggle_angle_elasticity = bone.wiggle_angle_elasticity
            other_bone.wiggle_length_elasticity = bone.wiggle_length_elasticity
            other_bone.wiggle_elasticity_soften = bone.wiggle_elasticity_soften
            other_bone.wiggle_gravity = bone.wiggle_gravity
            other_bone.wiggle_blend = bone.wiggle_blend
            other_bone.wiggle_air_drag = bone.wiggle_air_drag
            other_bone.wiggle_friction = bone.wiggle_friction
        return {'FINISHED'}

class WiggleReset(bpy.types.Operator):
    """Reset scene wiggle physics to rest state"""
    bl_idname = "wiggle.reset"
    bl_label = "Reset Physics"
    
    @classmethod
    def poll(cls,context):
        return context.scene.wiggle_enable and context.mode in ['OBJECT', 'POSE']
    
    def execute(self,context):
        context.scene.wiggle.reset = True
        context.scene.frame_set(context.scene.frame_current)
        context.scene.wiggle.reset = False
        wiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.wiggle_enable and not obj.wiggle_mute]
        for ob in wiggle_objs:
            wiggle_bones = [bone for bone in ob.pose.bones if getattr(bone, 'wiggle_enabled', False)]
            for bone in wiggle_bones:
                reset_bone(bone)
        context.scene.wiggle.lastframe = context.scene.frame_current
        return {'FINISHED'}

class WiggleProfile(bpy.types.Operator):
    bl_idname = "wiggle.profile"
    bl_label = "Print Profiling Information to Console"
    
    @classmethod
    def poll(cls,context):
        return context.scene.wiggle_enable and context.scene.wiggle_debug
    
    def execute(self,context):
        pstats.Stats(_profiler).sort_stats('cumulative').print_stats(20)
        _profiler.clear()
        return {'FINISHED'}
    
class WiggleSelect(bpy.types.Operator):
    """Select wiggle bones on selected objects in pose mode"""
    bl_idname = "wiggle.select"
    bl_label = "Select Enabled"
    
    @classmethod
    def poll(cls,context):
        return context.mode in ['POSE']
    
    def execute(self,context):
        bpy.ops.pose.select_all(action='DESELECT')
        wiggle_objs = [obj for obj in context.scene.objects if obj.type == 'ARMATURE' and obj.wiggle_enable and not obj.wiggle_mute]
        for ob in wiggle_objs:
            wiggle_bones = [bone for bone in ob.pose.bones if getattr(bone, 'wiggle_enabled', False)]
            for bone in wiggle_bones:
                bone.bone.select = True
        return {'FINISHED'}
    
class WiggleBake(bpy.types.Operator):
    """Bake this object's visible wiggle bones to keyframes"""
    bl_idname = "wiggle.bake"
    bl_label = "Bake Wiggle"
    
    @classmethod
    def poll(cls,context):
        return context.object
    
    def execute(self,context):
        def push_nla():
            if context.scene.wiggle.bake_overwrite: return
            if not context.scene.wiggle.bake_nla: return
            if not context.object.animation_data: return
            if not context.object.animation_data.action: return
            action = context.object.animation_data.action
            track = context.object.animation_data.nla_tracks.new()
            track.name = action.name
            track.strips.new(action.name, int(action.frame_range[0]), action)
            
        push_nla()
        
        bpy.ops.wiggle.reset()
            
        #preroll
        duration = context.scene.frame_end - context.scene.frame_start
        preroll = context.scene.wiggle.preroll
        context.scene.wiggle.is_preroll = False
        bpy.ops.wiggle.select()
        bpy.ops.wiggle.reset()
        while preroll >= 0:
            if context.scene.wiggle.loop:
                frame = context.scene.frame_end - (preroll%duration)
                context.scene.frame_set(frame)
            else:
                context.scene.frame_set(context.scene.frame_start-preroll)
            context.scene.wiggle.is_preroll = True
            preroll -= 1
        #bake
        if bpy.app.version[0] >= 4 and bpy.app.version[1] > 0:
            bpy.ops.nla.bake(frame_start = context.scene.frame_start,
                            frame_end = context.scene.frame_end,
                            only_selected = True,
                            visual_keying = True,
                            use_current_action = context.scene.wiggle.bake_overwrite,
                            bake_types={'POSE'},
                            channel_types={'LOCATION','ROTATION','SCALE'})
        else:
            bpy.ops.nla.bake(frame_start = context.scene.frame_start,
                            frame_end = context.scene.frame_end,
                            only_selected = True,
                            visual_keying = True,
                            use_current_action = context.scene.wiggle.bake_overwrite,
                            bake_types={'POSE'})
        context.scene.wiggle.is_preroll = False
        context.object.wiggle_freeze = True
        if not context.scene.wiggle.bake_overwrite:
            context.object.animation_data.action.name = 'WiggleAction'
        return {'FINISHED'}  

class WigglePanel:
    bl_category = 'Animation'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    
    @classmethod
    def poll(cls,context):
        return context.object  

class WIGGLE_PT_Settings(WigglePanel, bpy.types.Panel):
    bl_label = 'Wiggle 2'
        
    def draw(self,context):
        row = self.layout.row()
        icon = 'RADIOBUT_OFF' if not context.scene.wiggle_debug else 'RADIOBUT_ON'
        row.prop(context.scene, "wiggle_debug", icon=icon, text="",emboss=False)

        icon = 'HIDE_ON' if not context.scene.wiggle_enable else 'SCENE_DATA'
        row.prop(context.scene, "wiggle_enable", icon=icon, text="",emboss=False)
        if not context.scene.wiggle_enable:
            row.label(text='Scene muted.')
            return
        if not context.object.type == 'ARMATURE':
            row.label(text = ' Select armature.')
            return
#        row.label(icon='TRIA_RIGHT')
        if context.object.wiggle_freeze:
            row.prop(context.object,'wiggle_freeze',icon='FREEZE',icon_only=True,emboss=False)
            row.label(text = 'Wiggle Frozen after Bake.')
            return
        icon = 'HIDE_ON' if context.object.wiggle_mute else 'ARMATURE_DATA'
        row.prop(context.object,'wiggle_mute',icon=icon,icon_only=True,invert_checkbox=True,emboss=False)
        if context.object.wiggle_mute:
            row.label(text='Armature muted.')
            return
        if not context.active_pose_bone:
            row.label(text = ' Select pose bone.')
            return


class WIGGLE_OT_no_keyframe_tooltip_operator(bpy.types.Operator):
    bl_idname = "wiggle.no_keyframes_tooltip"
    bl_label = "No Rest Pose Detected"
    bl_description = "Keyframes are used to define the wiggle's rest pose. Please add keyframes to the bones in their rest state. You may safely ignore this warning if you are using actions in the NLA"

    def execute(self, context):
        return {'FINISHED'}

class WIGGLE_OT_connected_tooltip_operator(bpy.types.Operator):
    bl_idname = "wiggle.connected_tooltip"
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

class WIGGLE_OT_constraints_tooltip_operator(bpy.types.Operator):
    bl_idname = "wiggle.constraints_tooltip"
    bl_label = "Constraints Detected"
    bl_description = "Constraints are applied after wiggle, which can cause strange behavior. Click this button to automatically disable constraints on selected bones"

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

class WIGGLE_PT_Bone(WigglePanel,bpy.types.Panel):
    bl_label = ''
    bl_parent_id = 'WIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.wiggle_enable and context.object and not context.object.wiggle_mute and context.active_pose_bone
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.prop(context.active_pose_bone, 'wiggle_enabled')
    
    def draw(self,context):
        b = context.active_pose_bone
        if not b.wiggle_enabled: return
    
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        def drawprops(layout,b,props):
            for p in props:
                layout.prop(b, p)
        
        col = layout.column(align=True)
        if not is_bone_animated(b.id_data, b.name):
            layout.operator(WIGGLE_OT_no_keyframe_tooltip_operator.bl_idname, icon='ERROR')
        for c in b.constraints:
            if c.enabled:
                layout.operator(WIGGLE_OT_constraints_tooltip_operator.bl_idname, icon='ERROR')
                break
        if b.bone.use_connect:
            layout.operator(WIGGLE_OT_connected_tooltip_operator.bl_idname, icon='ERROR')
        drawprops(col,b,['wiggle_angle_elasticity', 'wiggle_length_elasticity', 'wiggle_elasticity_soften', 'wiggle_gravity', 'wiggle_blend', 'wiggle_air_drag', 'wiggle_friction'])
            

class WIGGLE_PT_Utilities(WigglePanel,bpy.types.Panel):
    bl_label = 'Global Wiggle Utilities'
    bl_parent_id = 'WIGGLE_PT_Settings'
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(cls,context):
        return context.scene.wiggle_enable
    
    def draw(self,context):
        layout = self.layout
        layout.use_property_split=True
        layout.use_property_decorate=False
        col = layout.column(align=True)
        if context.object.wiggle_enable and context.mode == 'POSE':
            col.operator('wiggle.copy')
            col.operator('wiggle.select')
        col.operator('wiggle.reset')
        if context.scene.wiggle_debug: col.operator('wiggle.profile')
        layout.prop(context.scene.wiggle, 'loop')
        
class WIGGLE_PT_Bake(WigglePanel,bpy.types.Panel):
    bl_label = 'Bake Wiggle'
    bl_parent_id = 'WIGGLE_PT_Utilities'
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(cls,context):
        return context.scene.wiggle_enable and context.object.wiggle_enable and context.mode == 'POSE'
    
    def draw(self,context):
        layout = self.layout
        layout.use_property_split=True
        layout.use_property_decorate=False
        layout.prop(context.scene.wiggle, 'preroll')
        layout.prop(context.scene.wiggle, 'bake_overwrite')
        row = layout.row()
        row.enabled = not context.scene.wiggle.bake_overwrite
        row.prop(context.scene.wiggle, 'bake_nla')
        layout.operator('wiggle.bake')

class WiggleBone(bpy.types.PropertyGroup):
    rolling_error: bpy.props.FloatVectorProperty(size=4, subtype='QUATERNION', override={'LIBRARY_OVERRIDABLE'})
    position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    working_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    parent_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    parent_pose: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    bone_length_change: bpy.props.FloatProperty(override={'LIBRARY_OVERRIDABLE'})
    virtual_working_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    virtual_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    virtual_position_last: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    debug: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    
class WiggleScene(bpy.types.PropertyGroup):
    lastframe: bpy.props.IntProperty()
    loop: bpy.props.BoolProperty(name='Loop Physics', description='Physics continues as timeline loops', default=True)
    preroll: bpy.props.IntProperty(name = 'Preroll', description='Frames to run simulation before bake', min=0, default=0)
    is_preroll: bpy.props.BoolProperty(default=False)
    bake_overwrite: bpy.props.BoolProperty(name='Overwrite Current Action', description='Bake wiggle into current action, instead of creating a new one', default = False)
    bake_nla: bpy.props.BoolProperty(name='Current Action to NLA', description='Move existing animation on the armature into an NLA strip', default = False) 
    is_rendering: bpy.props.BoolProperty(default=False)
    reset: bpy.props.BoolProperty(default=False)

def register():
    global debug_handler
    global debug_handler2
    debug_handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_VIEW')
    debug_handler2 = bpy.types.SpaceView3D.draw_handler_add(draw_callback_pose, (), 'WINDOW', 'POST_VIEW')
    
    #WIGGLE TOGGLES
    
    bpy.types.Scene.wiggle_enable = bpy.props.BoolProperty(
        name = 'Enable Scene',
        description = 'Enable wiggle on this scene',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_enable')
    )
    bpy.types.Scene.wiggle_debug = bpy.props.BoolProperty(
        name = 'Enable Debug',
        description = 'Enable drawing of wiggle debug lines. Green is the detected rest pose, red is the simulated physics pose. This is slow so disable when not needed.',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_enable')
    )
    bpy.types.Object.wiggle_enable = bpy.props.BoolProperty(
        name = 'Enable Armature',
        description = 'Enable wiggle on this armature',
        default = False,
        options={'HIDDEN'},
        override={'LIBRARY_OVERRIDABLE'}
    )
    bpy.types.Object.wiggle_mute = bpy.props.BoolProperty(
        name = 'Mute Armature',
        description = 'Mute wiggle on this armature.',
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_mute')
    )
    bpy.types.Object.wiggle_freeze = bpy.props.BoolProperty(
        name = 'Freeze Wiggle',
        description = 'Wiggle Calculation frozen after baking',
        default = False,
        override={'LIBRARY_OVERRIDABLE'}
    )
    bpy.types.PoseBone.wiggle_enabled = bpy.props.BoolProperty(
        name = 'Enable Bone Wiggle',
        description = "Enable wiggle on this bone",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
        update=lambda s, c: update_prop(s, c, 'wiggle_enabled')
    )
    bpy.types.PoseBone.wiggle_angle_elasticity = bpy.props.FloatProperty(
        name = 'Angle Elasticity',
        description = 'Spring angle stiffness, higher means more rigid',
        min = 0,
        default = 0.6,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_angle_elasticity')
    )
    bpy.types.PoseBone.wiggle_length_elasticity = bpy.props.FloatProperty(
        name = 'Length Elasticity',
        description = 'Spring length stiffness, higher means more rigid.',
        min = 0,
        default = 0.8,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_length_elasticity')
    )
    bpy.types.PoseBone.wiggle_elasticity_soften = bpy.props.FloatProperty(
        name = 'Elasticity Soften',
        description = 'Weakens the elasticity of the bone when its closer to the target pose. Higher means more like a free-rolling-ball-socket',
        min = 0,
        default = 0,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_elasticity_soften')
    )
    bpy.types.PoseBone.wiggle_gravity = bpy.props.FloatProperty(
        name = 'Gravity',
        description = 'Multiplier for scene gravity',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_gravity')
    )
    bpy.types.PoseBone.wiggle_blend = bpy.props.FloatProperty(
        name = 'Blend',
        description = 'wiggle blend, 0 means no wiggle, 1 means full wiggle',
        min = 0,
        default = 1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_blend')
    )
    bpy.types.PoseBone.wiggle_air_drag = bpy.props.FloatProperty(
        name = 'Air Drag',
        description = 'Multiplier for scene gravity',
        min = 0,
        default = 0,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_air_drag')
    )
    bpy.types.PoseBone.wiggle_friction = bpy.props.FloatProperty(
        name = 'Friction',
        description = 'Internal friction, higher means return to rest quicker',
        min = 0,
        default = 0.1,
        max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_friction')
    )
    
    #internal variables
    bpy.utils.register_class(WiggleBone)
    bpy.types.PoseBone.wiggle = bpy.props.PointerProperty(type=WiggleBone, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(WiggleScene)
    bpy.types.Scene.wiggle = bpy.props.PointerProperty(type=WiggleScene, override={'LIBRARY_OVERRIDABLE'})
    
    bpy.utils.register_class(WiggleReset)
    bpy.utils.register_class(WiggleProfile)
    bpy.utils.register_class(WiggleCopy)
    bpy.utils.register_class(WiggleSelect)
    bpy.utils.register_class(WiggleBake)
    bpy.utils.register_class(WIGGLE_PT_Settings)
    bpy.utils.register_class(WIGGLE_PT_Bone)
    bpy.utils.register_class(WIGGLE_PT_Utilities)
    bpy.utils.register_class(WIGGLE_PT_Bake)
    bpy.utils.register_class(WIGGLE_OT_no_keyframe_tooltip_operator)
    bpy.utils.register_class(WIGGLE_OT_connected_tooltip_operator)
    bpy.utils.register_class(WIGGLE_OT_constraints_tooltip_operator)
    
    bpy.app.handlers.frame_change_pre.append(wiggle_pre)
    bpy.app.handlers.frame_change_post.append(wiggle_post)
    bpy.app.handlers.render_pre.append(wiggle_render_pre)
    bpy.app.handlers.render_post.append(wiggle_render_post)
    bpy.app.handlers.render_cancel.append(wiggle_render_cancel)
    bpy.app.handlers.load_post.append(wiggle_load)

def unregister():
    global debug_handler
    global debug_handler2
    bpy.utils.unregister_class(WiggleBone)
    bpy.utils.unregister_class(WiggleScene)
    bpy.utils.unregister_class(WiggleReset)
    bpy.utils.unregister_class(WiggleProfile)
    bpy.utils.unregister_class(WiggleCopy)
    bpy.utils.unregister_class(WiggleSelect)
    bpy.utils.unregister_class(WiggleBake)
    bpy.utils.unregister_class(WIGGLE_PT_Settings)
    bpy.utils.unregister_class(WIGGLE_PT_Bone)
    bpy.utils.unregister_class(WIGGLE_PT_Utilities)
    bpy.utils.unregister_class(WIGGLE_PT_Bake)
    bpy.utils.unregister_class(WIGGLE_OT_no_keyframe_tooltip_operator)
    bpy.utils.unregister_class(WIGGLE_OT_connected_tooltip_operator)
    bpy.utils.unregister_class(WIGGLE_OT_constraints_tooltip_operator)
    
    bpy.app.handlers.frame_change_pre.remove(wiggle_pre)
    bpy.app.handlers.frame_change_post.remove(wiggle_post)
    bpy.app.handlers.render_pre.remove(wiggle_render_pre)
    bpy.app.handlers.render_post.remove(wiggle_render_post)
    bpy.app.handlers.render_cancel.remove(wiggle_render_cancel)
    bpy.app.handlers.load_post.remove(wiggle_load)
    bpy.types.SpaceView3D.draw_handler_remove(debug_handler, 'WINDOW')
    bpy.types.SpaceView3D.draw_handler_remove(debug_handler2, 'WINDOW')
    
if __name__ == "__main__":
    register()
