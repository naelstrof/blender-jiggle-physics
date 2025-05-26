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

### TO DO #####

# Basic object wiggle?
# handle inherit rotation?

# bugs:
# weird glitch when starting playback?

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
SCALE_MAT = Matrix.Identity(4)

_profiler = cProfile.Profile()

#return m2 in m1 space
def relative_matrix(m1,m2):
    return (m2.inverted() @ m1).inverted()

def flatten(mat):
    return [mat[j][i] for i in range(4) for j in range(4)]

def reset_scene():
    for wo in bpy.context.scene.wiggle.list:
        reset_ob(bpy.data.objects.get(wo.name))
                              
def reset_ob(ob):
    if not ob or not hasattr(ob, 'pose') or not hasattr(ob.pose, 'bones'):
        return
    wo = bpy.context.scene.wiggle.list.get(ob.name)
    if not wo:
        return
    for wb in wo.list:
        bone = ob.pose.bones.get(wb.name)
        if bone:
            reset_bone(bone.pose.bones.get(wb.name))

def reset_bone(b):
    b.wiggle.working_position = b.wiggle.position = b.wiggle.position_last = (b.id_data.matrix_world @ Matrix.Translation(b.head)).translation

def sort_bones(bones):
    # bones: list of pose bone objects, each with .name and .parent (or None)
    bone_map = {bone.name: bone for bone in bones}
    sorted_bones = []
    visited = set()

    def visit(bone):
        if bone.name in visited:
            return
        if bone.parent is not None and bone.parent.name in bone_map:
            visit(bone.parent)
        visited.add(bone.name)
        sorted_bones.append(bone)

    for bone in bones:
        visit(bone)

    return sorted_bones
                      
def build_list():
    bpy.context.scene.wiggle.list.clear()
    for ob in bpy.context.scene.objects:
        if ob.type != 'ARMATURE': continue
        wigglebones = []
        for b in ob.pose.bones:
            wigglebones.append(b)
                
        if not wigglebones:
            ob.wiggle_enable = False
            continue
        #wigglebones = sort_bones(wigglebones)
        
        ob.wiggle_enable = True
        wo = bpy.context.scene.wiggle.list.add()
        wo.name = ob.name
        for b in wigglebones:
            wb = wo.list.add()
            wb.name = b.name

        
def update_prop(self,context,prop): 
    if prop in ['wiggle_enable']:
        build_list()
    if type(self) == bpy.types.PoseBone: 
        for b in context.selected_pose_bones:
            b[prop] = self[prop]
        if prop in ['wiggle_enabled']:
            build_list()
            for b in context.selected_pose_bones:
                reset_bone(b)
    #edge case where is_rendering gets stuck, the user fiddling with any setting should unstuck it!
    context.scene.wiggle.is_rendering = False
        
def get_parent(b):
    return b.parent

def get_wiggle_parent(b):
    p = b.parent
    if not p: return None
    par = p if (p.wiggle_enabled and (not p.wiggle_mute)) else get_wiggle_parent(p)
    return par

def get_child(b):
    for child in b.children:
        if (child.wiggle_enabled and (not child.wiggle_mute)):
            return child
    return None

def collider_poll(self, object):
    return object.type == 'MESH'

def wind_poll(self, object):
    return object.field and object.field.type =='WIND'

def draw_apply_pose(bone,child,coords):
    if not child:
        return
    coords.append(bone.wiggle.debug)
    coords.append(child.wiggle.debug)

def verlet_integrate(b,dt,dt2,gravity):
    bw = b.wiggle
    p=get_wiggle_parent(b)
    
    if not p:
        return
    else:
        pw = p.wiggle
        position = bw.position
        delta = position - bw.position_last
        local_space_velocity = delta - (pw.position - pw.position_last)
        velocity = delta - local_space_velocity
        bw.working_position = position + velocity * (1.0-b.wiggle_air_drag) + local_space_velocity * (1.0-b.wiggle_friction) + gravity * b.wiggle_gravity * dt2

def constrain(b):
    bw = b.wiggle

    p=get_wiggle_parent(b)
    if not p:
        return

    pw = p.wiggle

    bone_pose = (b.id_data.matrix_world @ b.matrix).translation
    parent_pose = (p.id_data.matrix_world @ p.matrix).translation

    lengthToParent = (bone_pose - parent_pose).length
    # constrain angle
    parent_aim_pose = (parent_pose - pw.parent_pose).normalized()
    parent_aim = (pw.working_position - pw.parent_position).normalized()
    from_to_rot = parent_aim_pose.rotation_difference(parent_aim)
    current_pose = bone_pose - pw.parent_pose
    constraintTarget = from_to_rot @ current_pose
    error = (bw.working_position - (pw.parent_position + constraintTarget)).length
    error = min(error, 1.0)
    error = pow(error, b.wiggle_elasticity_soften*b.wiggle_elasticity_soften)
    bw.working_position = bw.working_position.lerp(pw.parent_position + constraintTarget, b.wiggle_angle_elasticity * b.wiggle_angle_elasticity * error)

    # todo collisions here

    # constrain length
    diff = bw.working_position - pw.working_position
    dir = diff.normalized()
    bw.working_position = bw.working_position.lerp(pw.working_position + dir * lengthToParent, b.wiggle_length_elasticity * b.wiggle_length_elasticity)

@persistent
def draw_callback():
    wiggle = bpy.context.scene.wiggle
    wiggle_list = wiggle.list
    if not wiggle_list:
        build_list()
        return
    objects = bpy.context.scene.objects
    object_cache = {ob.name: ob for ob in objects}
    for wo in wiggle_list:
        ob = object_cache.get(wo.name)

        if not ob:
            continue

        if getattr(ob, 'wiggle_mute', False) or getattr(ob, 'wiggle_freeze', False):
            continue

        pose_bones = ob.pose.bones
        bones = [
            pose_bones[wb.name]
            for wb in wo.list
            if not getattr(pose_bones[wb.name], 'wiggle_mute', False)
            and (getattr(pose_bones[wb.name], 'wiggle_enabled', False))
        ]
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
        shader.uniform_float("color", (1, 0, 0, 1))  # RGBA Red
        batch.draw(shader)

@persistent
def draw_callback_pose():
    wiggle = bpy.context.scene.wiggle
    wiggle_list = wiggle.list
    if not wiggle_list:
        build_list()
        return
    objects = bpy.context.scene.objects
    object_cache = {ob.name: ob for ob in objects}
    for wo in wiggle_list:
        ob = object_cache.get(wo.name)

        if not ob:
            continue

        if getattr(ob, 'wiggle_mute', False) or getattr(ob, 'wiggle_freeze', False):
            continue

        pose_bones = ob.pose.bones
        bones = [
            pose_bones[wb.name]
            for wb in wo.list
            if not getattr(pose_bones[wb.name], 'wiggle_mute', False)
            and (getattr(pose_bones[wb.name], 'wiggle_enabled', False))
        ]
        coords = []
        for b in bones:
            draw_apply_pose(b, get_child(b), coords)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        shader.bind()
        shader.uniform_float("color", (0, 1, 0, 1))  # RGBA Red
        batch.draw(shader)
        
@persistent
def wiggle_pre(scene):
    _profiler.enable()
    if (scene.wiggle.lastframe == scene.frame_current and not scene.wiggle.reset) or scene.wiggle.is_rendering:
        _profiler.disable()
        return

    if not scene.wiggle_enable:
        reset_scene()
        _profiler.disable()
        return

    scene_objects = scene.objects
    scene_collection = scene.collection

    for wo in scene.wiggle.list:
        ob = scene.objects.get(wo.name)
        if not ob:
            build_list()
            _profiler.disable()
            return

        if getattr(ob, "wiggle_mute", False) or getattr(ob, "wiggle_freeze", False):
            reset_ob(ob)
            _profiler.disable()
            continue

        ob_pose_bones = getattr(ob.pose, "bones", {})

        for wb in wo.list:
            b = ob_pose_bones.get(wb.name)
            if not b:
                build_list()
                _profiler.disable()
                return
            if getattr(b, "wiggle_mute", False) or not (getattr(b, "wiggle_enabled", False)):
                reset_bone(b)
                continue
    _profiler.disable()

@persistent                
def wiggle_post(scene,dg):
    _profiler.enable()
    wiggle = scene.wiggle

    if (wiggle.lastframe == scene.frame_current) and not wiggle.reset:
        _profiler.disable()
        return
    if not scene.wiggle_enable or wiggle.is_rendering:
        _profiler.disable()
        return

    lastframe = wiggle.lastframe
    frame_start, frame_end, frame_current = scene.frame_start, scene.frame_end, scene.frame_current
    frame_is_preroll = wiggle.is_preroll
    frame_loop = wiggle.loop

    if (frame_current == frame_start) and not frame_loop and not frame_is_preroll:
        bpy.ops.wiggle.reset()
        _profiler.disable()
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

    wiggle_list = wiggle.list

    objects = scene.objects
    object_cache = {ob.name: ob for ob in objects}
    for _ in range(accumulatedFrames):
        for wo in wiggle_list:
            ob = object_cache.get(wo.name)

            if not ob:
                continue

            if getattr(ob, 'wiggle_mute', False) or getattr(ob, 'wiggle_freeze', False):
                continue

            pose_bones = ob.pose.bones
            bones = [
                pose_bones[wb.name]
                for wb in wo.list
                if not getattr(pose_bones[wb.name], 'wiggle_mute', False)
                and (getattr(pose_bones[wb.name], 'wiggle_enabled', False))
            ]
            for b in bones:
                # caching for later
                p = get_wiggle_parent(b)
                if not p:
                    fixed_anim_position = (b.id_data.matrix_world @ b.matrix).translation
                    b.wiggle.working_position = fixed_anim_position 
                    c = get_child(b)
                    b.wiggle.parent_pose = 2 * fixed_anim_position - (c.id_data.matrix_world @ c.matrix).translation
                    b.wiggle.parent_position = b.wiggle.parent_pose
                else:
                    b.wiggle.parent_pose = (p.id_data.matrix_world @ p.matrix).translation
                    b.wiggle.parent_position = p.wiggle.working_position
            for b in bones:
                verlet_integrate(b,dt,dt2, scene.gravity)
            for b in bones:
                constrain(b)
            for bone in bones:
                bone.wiggle.position_last = bone.wiggle.position
                bone.wiggle.position = bone.wiggle.working_position

                child = get_child(bone)
                if not child:
                    return

                bone_pose = (bone.id_data.matrix_world @ bone.matrix).translation
                child_pose = (child.id_data.matrix_world @ child.matrix).translation

                bone.wiggle.debug = (bone.id_data.matrix_world @ bone.matrix).translation
                child.wiggle.debug = (child.id_data.matrix_world @ child.matrix).translation

                cachedAnimatedVector = (child_pose - bone_pose).normalized()
                simulatedVector = (child.wiggle.working_position - bone.wiggle.working_position).normalized()
                animPoseToPhysicsPose = cachedAnimatedVector.rotation_difference(simulatedVector)
                mat = animPoseToPhysicsPose.to_matrix().to_4x4()
                #mat.translation = bone.wiggle.working_position-bone_pose

                p = get_wiggle_parent(bone)
                if p:
                    loc, rot, scale = bone.matrix.decompose()
                    new_matrix = Matrix.Translation(loc) @ p.wiggle.matrix.inverted() @ mat @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
                    bone.matrix = new_matrix
                else:
                    loc, rot, scale = bone.matrix.decompose()
                    new_matrix = Matrix.Translation(bone.wiggle.working_position) @ mat @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(scale).to_4x4()
                    bone.matrix = new_matrix 
                bone.wiggle.matrix = flatten(mat)
    _profiler.disable()

@persistent        
def wiggle_render_pre(scene):
    scene.wiggle.is_rendering = True
    
@persistent
def wiggle_render_post(scene):
    scene.wiggle.is_rendering = False
    
@persistent
def wiggle_render_cancel(scene):
    scene.wiggle.is_rendering = False
    
@persistent
def wiggle_load(scene):
    build_list()
    s = bpy.context.scene
    s.wiggle.is_rendering = False
            
class WiggleCopy(bpy.types.Operator):
    """Copy active wiggle settings to selected bones"""
    bl_idname = "wiggle.copy"
    bl_label = "Copy Settings to Selected"
    
    @classmethod
    def poll(cls,context):
        return context.mode in ['POSE'] and context.active_pose_bone and (len(context.selected_pose_bones)>1)
    
    def execute(self,context):
        b = context.active_pose_bone
        for ob in context.selected_pose_bones:
            if ob == b: continue
            ob.wiggle_mute = b.wiggle_mute
            ob.wiggle_enabled = b.wiggle_enabled
            
            ob.wiggle_mass = b.wiggle_mass
            ob.wiggle_stiff = b.wiggle_stiff
            ob.wiggle_stretch = b.wiggle_stretch
            ob.wiggle_damp = b.wiggle_damp
            ob.wiggle_gravity = b.wiggle_gravity
            ob.wiggle_wind_ob = b.wiggle_wind_ob
            ob.wiggle_wind = b.wiggle_wind
            ob.wiggle_collider_type = b.wiggle_collider_type
            ob.wiggle_collider = b.wiggle_collider
            ob.wiggle_collider_collection = b.wiggle_collider_collection
            ob.wiggle_radius = b.wiggle_radius
            ob.wiggle_friction = b.wiggle_friction
            ob.wiggle_bounce = b.wiggle_bounce
            ob.wiggle_sticky = b.wiggle_sticky
            ob.wiggle_chain = b.wiggle_chain
            
            ob.wiggle_mass_head = b.wiggle_mass_head
            ob.wiggle_stiff_head = b.wiggle_stiff_head
            ob.wiggle_stretch_head = b.wiggle_stretch_head
            ob.wiggle_damp_head = b.wiggle_damp_head
            ob.wiggle_gravity_head = b.wiggle_gravity_head
            ob.wiggle_wind_ob_head = b.wiggle_wind_ob_head
            ob.wiggle_wind_head = b.wiggle_wind_head
            ob.wiggle_collider_type_head = b.wiggle_collider_type_head
            ob.wiggle_collider_head = b.wiggle_collider_head
            ob.wiggle_collider_collection_head = b.wiggle_collider_collection_head
            ob.wiggle_radius_head = b.wiggle_radius_head
            ob.wiggle_friction_head = b.wiggle_friction_head
            ob.wiggle_bounce_head = b.wiggle_bounce_head
            ob.wiggle_sticky_head = b.wiggle_sticky_head
            ob.wiggle_chain_head = b.wiggle_chain_head
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
        rebuild = False
        for wo in context.scene.wiggle.list:
            ob = context.scene.objects.get(wo.name)
            if not ob:
                rebuild = True
                continue
            for wb in wo.list:
                b = ob.pose.bones.get(wb.name)
                if not b:
                    rebuild = True
                    continue
                reset_bone(b)
        context.scene.wiggle.lastframe = context.scene.frame_current
        if rebuild: build_list()
        return {'FINISHED'}

class WiggleProfile(bpy.types.Operator):
    """Reset scene wiggle physics to rest state"""
    bl_idname = "wiggle.profile"
    bl_label = "Print Profiling Information"
    
    @classmethod
    def poll(cls,context):
        return context.scene.wiggle_enable and context.mode in ['OBJECT', 'POSE']
    
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
        rebuild = False
        for wo in context.scene.wiggle.list:
            ob = context.scene.objects.get(wo.name)
            if not ob:
                rebuild = True
                continue
            for wb in wo.list:
                b = ob.pose.bones.get(wb.name)
                if not b:
                    rebuild = True
                    continue
                b.bone.select = True
        if rebuild: build_list()
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
#        row.label(icon='TRIA_RIGHT')
        icon = 'HIDE_ON' if context.active_pose_bone.wiggle_mute else 'BONE_DATA'
        row.prop(context.active_pose_bone,'wiggle_mute',icon=icon,icon_only=True,invert_checkbox=True,emboss=False)
        if context.active_pose_bone.wiggle_mute:
            row.label(text='Bone muted.')
            return

class WIGGLE_PT_Head(WigglePanel,bpy.types.Panel):
    bl_label = ''
    bl_parent_id = 'WIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
        return context.scene.wiggle_enable and context.object and not context.object.wiggle_mute and context.active_pose_bone and not context.active_pose_bone.wiggle_mute
    
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
        col.operator('wiggle.profile')
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
        
class WiggleBoneItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(override={'LIBRARY_OVERRIDABLE'})
    
class WiggleItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(override={'LIBRARY_OVERRIDABLE'})  
    list: bpy.props.CollectionProperty(type=WiggleBoneItem, override={'LIBRARY_OVERRIDABLE','USE_INSERTION'})    

#store properties for a bone. custom properties for user editable. property group for internal calculations
class WiggleBone(bpy.types.PropertyGroup):
    matrix: bpy.props.FloatVectorProperty(name = 'Matrix', size=16, subtype = 'MATRIX', override={'LIBRARY_OVERRIDABLE'})
    position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    working_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    parent_position: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    parent_pose: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    debug: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    
class WiggleObject(bpy.types.PropertyGroup):
    list: bpy.props.CollectionProperty(type=WiggleItem, override={'LIBRARY_OVERRIDABLE'})
    
class WiggleScene(bpy.types.PropertyGroup):
    lastframe: bpy.props.IntProperty()
    loop: bpy.props.BoolProperty(name='Loop Physics', description='Physics continues as timeline loops', default=True)
    list: bpy.props.CollectionProperty(type=WiggleItem, override={'LIBRARY_OVERRIDABLE','USE_INSERTION'})
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
    
    bpy.types.PoseBone.wiggle_mute = bpy.props.BoolProperty(
        name = 'Bone Mute',
        description = "Mute wiggle",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_mute')
    )
    
    #TAIL PROPS
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
        description = 'Spring length stiffness, higher means more rigid',
        min = 0,
        default = 0.6,
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
    bpy.utils.register_class(WiggleBoneItem)
    bpy.utils.register_class(WiggleItem)
    bpy.utils.register_class(WiggleBone)
    bpy.types.PoseBone.wiggle = bpy.props.PointerProperty(type=WiggleBone, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(WiggleObject)
    bpy.types.Object.wiggle = bpy.props.PointerProperty(type=WiggleObject, override={'LIBRARY_OVERRIDABLE'})
    bpy.utils.register_class(WiggleScene)
    bpy.types.Scene.wiggle = bpy.props.PointerProperty(type=WiggleScene, override={'LIBRARY_OVERRIDABLE'})
    
    bpy.utils.register_class(WiggleReset)
    bpy.utils.register_class(WiggleProfile)
    bpy.utils.register_class(WiggleCopy)
    bpy.utils.register_class(WiggleSelect)
    bpy.utils.register_class(WiggleBake)
    bpy.utils.register_class(WIGGLE_PT_Settings)
    bpy.utils.register_class(WIGGLE_PT_Head)
    bpy.utils.register_class(WIGGLE_PT_Utilities)
    bpy.utils.register_class(WIGGLE_PT_Bake)
    
    bpy.app.handlers.frame_change_pre.append(wiggle_pre)
    bpy.app.handlers.frame_change_post.append(wiggle_post)
    bpy.app.handlers.render_pre.append(wiggle_render_pre)
    bpy.app.handlers.render_post.append(wiggle_render_post)
    bpy.app.handlers.render_cancel.append(wiggle_render_cancel)
    bpy.app.handlers.load_post.append(wiggle_load)

def unregister():
    global debug_handler
    global debug_handler2
    bpy.utils.unregister_class(WiggleBoneItem)
    bpy.utils.unregister_class(WiggleItem)
    bpy.utils.unregister_class(WiggleBone)
    bpy.utils.unregister_class(WiggleObject)
    bpy.utils.unregister_class(WiggleScene)
    bpy.utils.unregister_class(WiggleReset)
    bpy.utils.unregister_class(WiggleProfile)
    bpy.utils.unregister_class(WiggleCopy)
    bpy.utils.unregister_class(WiggleSelect)
    bpy.utils.unregister_class(WiggleBake)
    bpy.utils.unregister_class(WIGGLE_PT_Settings)
    bpy.utils.unregister_class(WIGGLE_PT_Head)
    bpy.utils.unregister_class(WIGGLE_PT_Utilities)
    bpy.utils.unregister_class(WIGGLE_PT_Bake)
    
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
