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

ZERO_VEC = Vector((0,0,0))
ONE_VEC = Vector((1,1,1))

#return m2 in m1 space
def relative_matrix(m1,m2):
    return (m2.inverted() @ m1).inverted()

def flatten(mat):
    dim = len(mat)
    return [mat[j][i] for i in range(dim) 
                      for j in range(dim)]

def reset_scene():
    for wo in bpy.context.scene.wiggle.list:
        reset_ob(bpy.data.objects.get(wo.name))
                              
def reset_ob(ob):
    wo = bpy.context.scene.wiggle.list.get(ob.name)
    for wb in wo.list:
        reset_bone(bpy.data.objects.get(wo.name).pose.bones.get(wb.name))

def reset_bone(b):
    b.wiggle.position = b.wiggle.position_last = (b.id_data.matrix_world @ Matrix.Translation(b.tail)).translation
    b.wiggle.position_head = b.wiggle.position_last_head = (b.id_data.matrix_world @ b.matrix).translation
    b.wiggle.velocity = b.wiggle.velocity_head = b.wiggle.collision_normal = b.wiggle.collision_normal_head = ZERO_VEC
    b.wiggle.matrix = flatten(b.id_data.matrix_world @ b.matrix)
                      
def build_list():
    bpy.context.scene.wiggle.list.clear()
    for ob in bpy.context.scene.objects:
        if ob.type != 'ARMATURE': continue
        wigglebones = []
        for b in ob.pose.bones:
            if b.wiggle_tail or (b.wiggle_head and not b.bone.use_connect):
                wigglebones.append(b)
                b.wiggle_enable = True
            else:
                b.wiggle_enable = False
                
        if not wigglebones:
            ob.wiggle_enable = False
            continue
        
        ob.wiggle_enable = True
        wo = bpy.context.scene.wiggle.list.add()
        wo.name = ob.name
        for b in wigglebones:
            wb = wo.list.add()
            wb.name = b.name

        
def update_prop(self,context,prop): 
    if prop in ['wiggle_mute','wiggle_enable']:
        build_list()
    if type(self) == bpy.types.PoseBone: 
        for b in context.selected_pose_bones:
            b[prop] = self[prop]
        if prop in ['wiggle_head', 'wiggle_tail']:
            build_list()
            for b in context.selected_pose_bones:
                reset_bone(b)
    #edge case where is_rendering gets stuck, the user fiddling with any setting should unstuck it!
    context.scene.wiggle.is_rendering = False
        
def get_parent(b):
    p = b.parent
    if not p: return None
    par = p if (p.wiggle_enable and (not p.wiggle_mute) and ((p.wiggle_head and not p.bone.use_connect) or p.wiggle_tail)) else get_parent(p)
    return par

def length_world(b):
    return (b.id_data.matrix_world @ b.head - b.id_data.matrix_world @ b.tail).length

def collider_poll(self, object):
    return object.type == 'MESH'

def wind_poll(self, object):
    return object.field and object.field.type =='WIND'

def collide(b,dg,dt,head=False):
    if head:
        pos = b.wiggle.position_head
        vel = b.wiggle.velocity_head
        cp = b.wiggle.collision_point_head
        co = b.wiggle.collision_ob_head
        cn = b.wiggle.collision_normal_head
        
        collider_type = b.wiggle_collider_type_head
        wiggle_collider = b.wiggle_collider_head
        wiggle_collection = b.wiggle_collider_collection_head
        
        radius = b.wiggle_radius_head
        sticky = b.wiggle_sticky_head
        bounce = b.wiggle_bounce_head
        friction = b.wiggle_friction_head
    else:
        pos = b.wiggle.position
        vel = b.wiggle.velocity
        cp = b.wiggle.collision_point
        co = b.wiggle.collision_ob
        cn = b.wiggle.collision_normal
        
        collider_type = b.wiggle_collider_type
        wiggle_collider = b.wiggle_collider
        wiggle_collection = b.wiggle_collider_collection
        
        radius = b.wiggle_radius
        sticky = b.wiggle_sticky
        bounce = b.wiggle_bounce
        friction = b.wiggle_friction
        
    colliders = []
    if collider_type == 'Object' and wiggle_collider:
        if wiggle_collider.name in bpy.context.scene.objects:
            colliders = [wiggle_collider]
    if collider_type == 'Collection' and wiggle_collection:
        if wiggle_collection in bpy.context.scene.collection.children_recursive:
            colliders = [ob for ob in wiggle_collection.objects if ob.type == 'MESH']
    col = False
    for collider in colliders:
        cmw = collider.matrix_world
        p = collider.closest_point_on_mesh(cmw.inverted() @ pos, depsgraph=dg)
        n = (cmw.to_quaternion().to_matrix().to_4x4() @ p[2]).normalized()
        i = cmw @ p[1]
        v = i-pos
        
        if (n.dot(v.normalized()) > 0.01) or (v.length < radius) or (co and (v.length < (radius+sticky))):
            if n.dot(v.normalized()) > 0: #vec is below
                nv = v.normalized()
            else: #normal is opposite dir to vec
                nv = -v.normalized()
            pos = i + nv*radius
            
            if co:
                collision_point = co.matrix_world @ cp
                pos = pos.lerp(collision_point, friction)
            col = True
            co = collider
            cp = relative_matrix(cmw, Matrix.Translation(pos)).translation
            cn = nv
    if not col:
        co = None
    
    if head:
        b.wiggle.position_head = pos
        b.wiggle.collision_point_head = cp
        b.wiggle.collision_ob_head = co  
        b.wiggle.collision_normal_head = cn
    else:
        b.wiggle.position = pos
        b.wiggle.collision_point = cp
        b.wiggle.collision_ob = co  
        b.wiggle.collision_normal = cn

def update_matrix(b,last=False):
    id_world = b.id_data.matrix_world
    bone = b.bone
    bw = b.wiggle
    loc = Matrix.Identity(4)

    parent = get_parent(b)
    real_mat = None
    if parent:
        parent_wiggle_matrix = parent.wiggle.matrix
        real_mat = relative_matrix(parent.matrix, b.matrix)
        mat = parent_wiggle_matrix @ real_mat
        if bone.inherit_scale == 'FULL':
            m2 = mat
        else:
            diff = real_mat
            lo = Matrix.Translation((parent_wiggle_matrix @ diff).translation)
            parent_rot = parent_wiggle_matrix.to_quaternion().to_matrix().to_4x4()
            diff_rot = diff.to_quaternion().to_matrix().to_4x4()
            ro = parent_rot @ diff_rot
            scale_val = (id_world @ b.matrix).decompose()[2]
            sc = Matrix.LocRotScale(None,None,scale_val)
            m2 = lo @ ro @ sc
    else:
        mat = id_world @ b.matrix
        m2 = mat
            
    if b.wiggle_head and not bone.use_connect:
        m2 = Matrix.Translation(bw.position_head - m2.translation) @ m2
        loc = Matrix.Translation(relative_matrix(mat, Matrix.Translation(bw.position_head)).translation)
        mat = m2
    vec = relative_matrix(m2, Matrix.Translation(bw.position)).translation
    rxz = vec.to_track_quat('Y','Z')
    rot = rxz.to_matrix().to_4x4()
    
    l_world = length_world(b)
    if bone.inherit_scale == 'FULL':
        l0 = bone.length
        l1=relative_matrix(mat, Matrix.Translation(bw.position)).translation.length
        sy = l1 / l0 if l0 != 0 else 1.0
    else:
        par = b.parent
        if par:
            par_mat = par.matrix
            par_rel = relative_matrix(par_mat, b.matrix).translation
            if par:
                par_wiggle_matrix = par.wiggle.matrix
                sy = (par_wiggle_matrix @ par_rel - bw.position).length/l_world
            else:
                sy=(id_world @ par_mat @ par_rel - bw.position).length/l_world
        else:
            sy = (id_world @ b.matrix.translation - b.wiggle.position).length/l_world
    
    if b.wiggle_head and not b.bone.use_connect:
        sy = (b.wiggle.position_head - b.wiggle.position).length/l_world
            
    scale = Matrix.Identity(4)
    scale[1][1] = sy
    
    if last:
        b.matrix = b.matrix @ loc @ rot @ scale
    bw.matrix = flatten(m2 @ rot @ scale)

#can include gravity, wind, etc    
def move(b,dg,dt,dt2):
    if b.wiggle_tail:
        damp = max(min(1-b.wiggle_damp*dt, 1),0) 
        b.wiggle.velocity=b.wiggle.velocity*damp
        F = bpy.context.scene.gravity * b.wiggle_gravity
        if b.wiggle_wind_ob:
            dir = b.wiggle_wind_ob.matrix_world.to_quaternion().to_matrix().to_4x4() @ Vector((0,0,1))
            fac = 1 - b.wiggle_wind_ob.field.wind_factor * abs(dir.dot((b.wiggle.position - b.wiggle.matrix.translation).normalized()))
            F += dir * fac * b.wiggle_wind_ob.field.strength * b.wiggle_wind / b.wiggle_mass
        b.wiggle.position += b.wiggle.velocity + F*dt2
        collide(b,dg,dt)
    
    if b.wiggle_head and not b.bone.use_connect:
        damp = max(min(1-b.wiggle_damp_head*dt,1),0)
        b.wiggle.velocity_head = b.wiggle.velocity_head*damp
        F = bpy.context.scene.gravity * b.wiggle_gravity_head
        if b.wiggle_wind_ob_head:
            dir = b.wiggle_wind_ob_head.matrix_world.to_quaternion().to_matrix().to_4x4() @ Vector((0,0,1))
            F += dir * b.wiggle_wind_ob_head.field.strength * b.wiggle_wind_head / b.wiggle_mass_head
        b.wiggle.position_head += b.wiggle.velocity_head + F*dt2
        collide(b,dg,dt,True)
    update_matrix(b)

def bone_get_fac(mass1,mass2):
    return 0.5 if mass1 == mass2 else mass1/(mass1+mass2)

def bone_spring(target, position, stiff, dt2, iterations):
    s = target - position
    Fs = s * stiff / iterations
    if (Fs*dt2).length > s.length:
        return s
    return Fs*dt2

def bone_stretch(target, position, fac):
    return (target - position)*(1-fac)

def constrain(b,i,dg,dt,dt2,iterations):
    bw = b.wiggle
    p=get_parent(b)
    if p:
        mat = p.wiggle.matrix @ relative_matrix(p.matrix, b.matrix)
    else:
        mat = b.id_data.matrix_world @ b.matrix
    update_p = False  
    
    #spring
    if b.wiggle_head and not b.bone.use_connect:
        target = mat.translation
        s = bone_spring(target, bw.position_head, b.wiggle_stiff_head, dt2, iterations)
        if p and b.wiggle_chain_head:
            if p.wiggle_tail:
                fac = bone_get_fac(b.wiggle_mass_head, p.wiggle_mass) if i else p.wiggle_stretch
                p.wiggle.position -= s*fac
            else:
                fac = bone_get_fac(b.wiggle_mass_head, p.wiggle_mass_head)
                p.wiggle.position_head -= s*fac
            bw.position_head += s*(1-fac)
        else:
            bw.position_head += s

        mat = Matrix.LocRotScale(bw.position_head, mat.decompose()[1], b.matrix.decompose()[2])
        target = mat @ Vector((0,b.bone.length,0))
        if b.wiggle_tail:
            s = bone_spring(target, bw.position, b.wiggle_stiff, dt2, iterations)
            if b.wiggle_chain:
                fac = bone_get_fac(b.wiggle_mass, b.wiggle_mass_head)
                bw.position_head -= s*fac
                bw.position += s*(1-fac)
            else:
                bw.position += s
        else:
            bw.position = target
    else:
        mat = Matrix.LocRotScale(mat.decompose()[0], mat.decompose()[1],b.matrix.decompose()[2])
        target = mat @ Vector((0, b.bone.length,0))
        s = bone_spring(target, bw.position, b.wiggle_stiff, dt2, iterations)
        if p and b.wiggle_chain and p.wiggle_tail: # and b.bone.use_connect:
            fac = bone_get_fac(b.wiggle_mass, p.wiggle_mass)# if i else p.wiggle_stretch
            if i == 0: fac = p.wiggle_stretch
            if p == b.parent and b.bone.use_connect: #direct parent optimization
                p.wiggle.position -= s*fac
            else:
                tailpos = bw.matrix @ Vector((0,b.bone.length,0))
                midpos = (bw.matrix.translation + tailpos)/2
                v1 = midpos-p.wiggle.matrix.translation
                tailpos -= s*fac
                midpos = (bw.matrix.translation + tailpos)/2
                v2 = midpos-p.wiggle.matrix.translation
                sc = v2.length/v1.length
                q = v1.rotation_difference(v2)
                v3 = q @ (p.wiggle.position - p.wiggle.matrix.translation)
                p.wiggle.position = p.wiggle.matrix.translation + v3*sc
                
            bw.position += s*(1-fac)
            update_p = True
        else:
            bw.position += s
            
    #stretch
    if b.wiggle_head and not b.bone.use_connect:
        if p:
            if b.parent == p and p.wiggle_tail:
                target = p.wiggle.position + (bw.position_head - p.wiggle.position).normalized()*(b.id_data.matrix_world @ b.head - b.id_data.matrix_world @ p.tail).length
            else: #indirect
                targetpos = p.wiggle.matrix @ relative_matrix(p.matrix, b.parent.matrix) @ Vector((0,b.parent.length,0))
                target = targetpos + (bw.position_head - targetpos).normalized()*(b.id_data.matrix_world @ b.head - b.id_data.matrix_world @ b.parent.tail).length
        elif b.parent:
            ptail = b.id_data.matrix_world @ b.parent.tail
            target = ptail + (bw.position_head - ptail).normalized() * (b.id_data.matrix_world @ b.head - b.id_data.matrix_world @ b.parent.tail).length
        else:
            target = mat.translation
        s = bone_stretch(target, bw.position_head, b.wiggle_stretch_head)
        if p and b.wiggle_chain_head:
            if p.wiggle_tail:
                fac = bone_get_fac(b.wiggle_mass_head, p.wiggle_mass) if i else p.wiggle_stretch
                tailpos = p.wiggle.matrix @ relative_matrix(p.matrix, b.parent.matrix) @ Vector((0,b.parent.length,0))
                ratio = (p.wiggle.matrix.translation - p.wiggle.position).length/(p.wiggle.matrix.translation - tailpos).length
                tailpos -= s*fac
                p.wiggle.position -= s*ratio*fac
            else: #DOES THIS ASSUME ANYTHING? (No, head only translates, no bone stretching)
                fac = bone_get_fac(b.wiggle_mass_head, p.wiggle_mass_head) if i else p.wiggle_stretch_head
                p.wiggle.position_head -= s*fac
            bw.position_head += s*(1-fac)
        else:
            bw.position_head += s
            
        target = bw.position_head + (bw.position - bw.position_head).normalized()*length_world(b)
        if b.wiggle_tail: #tail stretch only relative to head
            s = bone_stretch(target, bw.position, b.wiggle_stretch)
            if b.wiggle_chain:
                fac = bone_get_fac(b.wiggle_mass, b.wiggle_mass_head) if i else b.wiggle_stretch_head
                bw.position_head -= s*fac
                bw.position += s*(1-fac)
            else:
                bw.position += s
        else: bw.position = target
    else: #tail stretch relative to parent or none
        target = mat.translation + (bw.position - mat.translation).normalized()*length_world(b)
        s = bone_stretch(target, bw.position, b.wiggle_stretch)
        if p and b.wiggle_chain and p.wiggle_tail: #ASSUMES P IS DIRECT PARENT?
            fac = bone_get_fac(b.wiggle_mass, p.wiggle_mass) #if i else p.wiggle_stretch
            if i == 0: fac = p.wiggle_stretch
            if (p == b.parent and b.bone.use_connect): #optimization with direct parent tail
                p.wiggle.position -= s*fac
            else:
                headpos = bw.matrix.translation
                v1 = headpos - p.wiggle.matrix.translation
                headpos -= s*fac
                v2 = headpos - p.wiggle.matrix.translation
                sc = v2.length/v1.length
                q = v1.rotation_difference(v2)
                v3 = q @ (p.wiggle.position - p.wiggle.matrix.translation)
                p.wiggle.position = p.wiggle.matrix.translation + v3*sc
                
            bw.position += s*(1-fac)
            update_p = True
        else:
            bw.position += s

    if update_p:
        collide(p,dg,dt)#would only be tail changing
        update_matrix(p)
    if b.wiggle_tail:
        collide(b,dg,dt)
    if b.wiggle_head:
        collide(b,dg,dt,True)
    update_matrix(b)
 
        
@persistent
def wiggle_pre(scene):
    if (scene.wiggle.lastframe == scene.frame_current and not scene.wiggle.reset) or scene.wiggle.is_rendering:
        return

    if not scene.wiggle_enable:
        reset_scene()
        return

    scene_objects = scene.objects
    scene_collection = scene.collection

    for wo in scene.wiggle.list:
        ob = scene.objects[wo.name]
        if not ob:
            build_list()
            return

        if getattr(ob, "wiggle_mute", False) or getattr(ob, "wiggle_freeze", False):
            reset_ob(ob)
            continue

        ob_pose_bones = getattr(ob.pose, "bones", {})

        for wb in wo.list:
            b = ob_pose_bones.get(wb.name)
            if not b:
                build_list()
                return
            if getattr(b, "wiggle_mute", False) or not (getattr(b, "wiggle_head", False) or getattr(b, "wiggle_tail", False)):
                reset_bone(b)
                continue
            bw = b.wiggle
            if not bw.collision_col:
                if b.wiggle_collider_collection:
                    b.wiggle_collider_collection = bpy.data.collections.get(b.wiggle_collider_collection.name)
                    bw.collision_col = scene.collection
                elif b.wiggle_collider_collection_head:
                    bpy.data.collections.get(b.wiggle_collider_collection_head.name)
                    bw.collision_col = scene.collection
                elif b.wiggle_collider:
                    bpy.data.objects.get(b.wiggle_collider.name)
                    bw.collision_col = scene.collection
                elif b.wiggle_collider_head:
                    bpy.data.objects.get(b.wiggle_collider_head.name)
                    bw.collision_col = scene.collection
            b.location = ZERO_VEC
            b.rotation_quaternion = Quaternion((1,0,0,0))
            b.rotation_euler = ZERO_VEC
            b.scale = ONE_VEC

@persistent                
def wiggle_post(scene,dg):
    wiggle = scene.wiggle

    if (wiggle.lastframe == scene.frame_current) and not wiggle.reset:
        return
    if not scene.wiggle_enable or wiggle.is_rendering:
        return

    lastframe = wiggle.lastframe
    frame_start, frame_end, frame_current = scene.frame_start, scene.frame_end, scene.frame_current
    frame_is_preroll = wiggle.is_preroll
    frame_loop = wiggle.loop

    if (frame_current == frame_start) and not frame_loop and not frame_is_preroll:
        bpy.ops.wiggle.reset()
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
    wiggle_iterations = wiggle.iterations

    objects = scene.objects
    for _ in range(accumulatedFrames):
        for wo in wiggle_list:
            ob = objects[wo.name]

            if getattr(ob, 'wiggle_mute', False) or getattr(ob, 'wiggle_freeze', False):
                continue

            pose_bones = ob.pose.bones

            bones = [
                pose_bones[wb.name]
                for wb in wo.list
                if not getattr(pose_bones[wb.name], 'wiggle_mute', False)
                and (getattr(pose_bones[wb.name], 'wiggle_head', False) or getattr(pose_bones[wb.name], 'wiggle_tail', False))
            ]
            for b in bones:
                b.wiggle.collision_normal = b.wiggle.collision_normal_head = ZERO_VEC.copy()
                move(b,dg,dt,dt2)
            for i in range(wiggle_iterations):
                idx = wiggle_iterations - 1 - i
                for b in bones:
                    constrain(b,idx,dg,dt,dt2,wiggle_iterations)
            for b in bones:
                update_matrix(b,True) #final update
            for b in bones:
                bw = b.wiggle

                vb = ZERO_VEC
                if bw.collision_normal.length:
                    vb = (
                        bw.velocity.reflect(bw.collision_normal)
                        .project(bw.collision_normal)*b.wiggle_bounce
                    )
                bw.velocity = (bw.position - bw.position_last) + vb

                vb_head = ZERO_VEC 
                if bw.collision_normal_head.length:
                    vb_head = (
                        bw.velocity_head.reflect(bw.collision_normal_head)
                        .project(bw.collision_normal_head)*b.wiggle_bounce_head
                    )
                bw.velocity_head = (bw.position_head - bw.position_last_head) + vb_head
                bw.position_last = bw.position
                bw.position_last_head = bw.position_head

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
            ob.wiggle_mute = b.wiggle_mute
            ob.wiggle_head = b.wiggle_head
            ob.wiggle_tail = b.wiggle_tail
            ob.wiggle_head_mute = b.wiggle_head_mute
            ob.wiggle_tail_mute = b.wiggle_tail_mute
            
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
#        return context.active_pose_bone and not context.active_pose_bone.bone.use_connect
        return context.scene.wiggle_enable and context.object and not context.object.wiggle_mute and context.active_pose_bone and not context.active_pose_bone.wiggle_mute and not context.active_pose_bone.bone.use_connect
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.prop(context.active_pose_bone, 'wiggle_head')
    
    def draw(self,context):
        b = context.active_pose_bone
        if not b.wiggle_head: return
    
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        def drawprops(layout,b,props):
            for p in props:
                layout.prop(b, p)
        
        col = layout.column(align=True)
        drawprops(col,b,['wiggle_mass_head','wiggle_stiff_head','wiggle_stretch_head','wiggle_damp_head'])
        col.separator()
        col.prop(b,'wiggle_gravity_head')
        row=col.row(align=True)
        row.prop(b,'wiggle_wind_ob_head')
        sub = row.row(align=True)
        sub.ui_units_x = 5
        sub.prop(b, 'wiggle_wind_head', text='')
        col.separator()
        col.prop(b, 'wiggle_collider_type_head',text='Collisions')
        collision = False
        if b.wiggle_collider_type_head == 'Object':
            row = col.row(align=True)
            row.prop_search(b, 'wiggle_collider_head', context.scene, 'objects',text=' ')
            if b.wiggle_collider_head:
                if b.wiggle_collider_head.name in context.scene.objects:
                    collision = True
                else:
                    row.label(text='',icon='UNLINKED')
        else:
            row = col.row(align=True)
            row.prop_search(b, 'wiggle_collider_collection_head', bpy.data, 'collections', text=' ')
            if b.wiggle_collider_collection_head:
                if b.wiggle_collider_collection_head in context.scene.collection.children_recursive:
                    collision = True
                else:
                    row.label(text='',icon='UNLINKED')
            
        if collision:
            col = layout.column(align=True)
            drawprops(col,b,['wiggle_radius_head','wiggle_friction_head','wiggle_bounce_head','wiggle_sticky_head'])
        layout.prop(b,'wiggle_chain_head')
            
class WIGGLE_PT_Tail(WigglePanel,bpy.types.Panel):
    bl_label = ''
    bl_parent_id = 'WIGGLE_PT_Settings'
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    @classmethod
    def poll(cls,context):
#        return context.active_pose_bone
        return context.scene.wiggle_enable and context.object and not context.object.wiggle_mute and context.active_pose_bone and not context.active_pose_bone.wiggle_mute
    
    def draw_header(self,context):
        row=self.layout.row(align=True)
        row.prop(context.active_pose_bone, 'wiggle_tail')
        
    def draw(self,context):
        b = context.active_pose_bone
        if not b.wiggle_tail: return
    
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        def drawprops(layout,b,props):
            for p in props:
                layout.prop(b, p)
        
        col = layout.column(align=True)
        drawprops(col,b,['wiggle_mass','wiggle_stiff','wiggle_stretch','wiggle_damp'])
        col.separator()
        col.prop(b,'wiggle_gravity')
        row=col.row(align=True)
        row.prop(b,'wiggle_wind_ob')
        sub = row.row(align=True)
        sub.ui_units_x = 5
        sub.prop(b, 'wiggle_wind', text='')
        col.separator()
        col.prop(b, 'wiggle_collider_type',text='Collisions')
        collision = False
        if b.wiggle_collider_type == 'Object':
            row = col.row(align=True)
            row.prop_search(b, 'wiggle_collider', context.scene, 'objects',text=' ')
            if b.wiggle_collider:
                if b.wiggle_collider.name in context.scene.objects:
                    collision = True
                else:
                    row.label(text='',icon='UNLINKED')
        else:
            row = col.row(align=True)
            row.prop_search(b, 'wiggle_collider_collection', bpy.data, 'collections', text=' ')
            if b.wiggle_collider_collection:
                if b.wiggle_collider_collection in context.scene.collection.children_recursive:
                    collision = True
                else:
                    row.label(text='',icon='UNLINKED')
        if collision:
            col = layout.column(align=True)
            drawprops(col,b,['wiggle_radius','wiggle_friction','wiggle_bounce','wiggle_sticky'])
        layout.prop(b,'wiggle_chain')

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
        layout.prop(context.scene.wiggle, 'loop')
        layout.prop(context.scene.wiggle, 'iterations')
        
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
    velocity: bpy.props.FloatVectorProperty(subtype='VELOCITY', override={'LIBRARY_OVERRIDABLE'})
    
    collision_point:bpy.props.FloatVectorProperty(subtype = 'TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    collision_ob: bpy.props.PointerProperty(type=bpy.types.Object, override={'LIBRARY_OVERRIDABLE'})
    collision_normal: bpy.props.FloatVectorProperty(subtype = 'TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    collision_col: bpy.props.PointerProperty(type=bpy.types.Collection,override={'LIBRARY_OVERRIDABLE'})
    
    position_head: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    position_last_head: bpy.props.FloatVectorProperty(subtype='TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    velocity_head: bpy.props.FloatVectorProperty(subtype='VELOCITY', override={'LIBRARY_OVERRIDABLE'})
    
    collision_point_head:bpy.props.FloatVectorProperty(subtype = 'TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    collision_ob_head: bpy.props.PointerProperty(type=bpy.types.Object, override={'LIBRARY_OVERRIDABLE'})
    collision_normal_head: bpy.props.FloatVectorProperty(subtype = 'TRANSLATION', override={'LIBRARY_OVERRIDABLE'})
    
class WiggleObject(bpy.types.PropertyGroup):
    list: bpy.props.CollectionProperty(type=WiggleItem, override={'LIBRARY_OVERRIDABLE'})
    
class WiggleScene(bpy.types.PropertyGroup):
    lastframe: bpy.props.IntProperty()
    iterations: bpy.props.IntProperty(name='Quality', description='Constraint solver interations for chain physics', min=1, default=2, soft_max=10, max=100)
    loop: bpy.props.BoolProperty(name='Loop Physics', description='Physics continues as timeline loops', default=True)
    list: bpy.props.CollectionProperty(type=WiggleItem, override={'LIBRARY_OVERRIDABLE','USE_INSERTION'})
    preroll: bpy.props.IntProperty(name = 'Preroll', description='Frames to run simulation before bake', min=0, default=0)
    is_preroll: bpy.props.BoolProperty(default=False)
    bake_overwrite: bpy.props.BoolProperty(name='Overwrite Current Action', description='Bake wiggle into current action, instead of creating a new one', default = False)
    bake_nla: bpy.props.BoolProperty(name='Current Action to NLA', description='Move existing animation on the armature into an NLA strip', default = False) 
    is_rendering: bpy.props.BoolProperty(default=False)
    reset: bpy.props.BoolProperty(default=False)

def register():
    
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
#        update=lambda s, c: update_prop(s, c, 'wiggle_enable')
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
    bpy.types.PoseBone.wiggle_enable = bpy.props.BoolProperty(
        name = 'Enable Bone',
        description = "Enable wiggle on this bone",
        default = False,
        options={'HIDDEN'},
        override={'LIBRARY_OVERRIDABLE'}
#        update=lambda s, c: update_prop(s, c, 'wiggle_enable')
    )
    bpy.types.PoseBone.wiggle_mute = bpy.props.BoolProperty(
        name = 'Mute Bone',
        description = "Mute wiggle for this bone.",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_mute')
    )
    bpy.types.PoseBone.wiggle_head = bpy.props.BoolProperty(
        name = 'Bone Head',
        description = "Enable wiggle on this bone's head",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
        update=lambda s, c: update_prop(s, c, 'wiggle_head')
    )
    bpy.types.PoseBone.wiggle_tail = bpy.props.BoolProperty(
        name = 'Bone Tail',
        description = "Enable wiggle on this bone's tail",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        options={'HIDDEN'},
        update=lambda s, c: update_prop(s, c, 'wiggle_tail')
    )
    
    bpy.types.PoseBone.wiggle_head_mute = bpy.props.BoolProperty(
        name = 'Bone Head Mute',
        description = "Mute wiggle on this bone's head",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_head_mute')
    )
    bpy.types.PoseBone.wiggle_tail_mute = bpy.props.BoolProperty(
        name = 'Bone Tail Mute',
        description = "Mute wiggle on this bone's tail",
        default = False,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_tail_mute')
    )
    
    #TAIL PROPS
    
    bpy.types.PoseBone.wiggle_mass = bpy.props.FloatProperty(
        name = 'Mass',
        description = 'Mass of bone',
        min = 0.01,
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_mass')
    )
    bpy.types.PoseBone.wiggle_stiff = bpy.props.FloatProperty(
        name = 'Stiff',
        description = 'Spring stiffness coefficient, can be large numbers',
        min = 0,
        default = 400,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_stiff')
    )
    bpy.types.PoseBone.wiggle_stretch = bpy.props.FloatProperty(
        name = 'Stretch',
        description = 'Bone stretchiness factor, 0 to 1 range',
        min = 0,
        default = 0,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_stretch')
    )
    bpy.types.PoseBone.wiggle_damp = bpy.props.FloatProperty(
        name = 'Damp',
        description = 'Dampening coefficient, can be greater than 1',
        min = 0,
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_damp')
    )
    bpy.types.PoseBone.wiggle_gravity = bpy.props.FloatProperty(
        name = 'Gravity',
        description = 'Multiplier for scene gravity',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_gravity')
    )
    bpy.types.PoseBone.wiggle_wind_ob = bpy.props.PointerProperty(
        name='Wind', 
        description='Wind force field object', 
        type=bpy.types.Object, 
        poll = wind_poll, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_prop(s, c, 'wiggle_wind_ob')
    )
    bpy.types.PoseBone.wiggle_wind = bpy.props.FloatProperty(
        name = 'Wind Multiplier',
        description = 'Multiplier for wind forces',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_wind')
    )
    bpy.types.PoseBone.wiggle_chain = bpy.props.BoolProperty(
        name = 'Chain',
        description = 'Bone affects its parent creating a physics chain',
        default = True,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_chain')
    )
    
    #HEAD PROPS
    
    bpy.types.PoseBone.wiggle_mass_head = bpy.props.FloatProperty(
        name = 'Mass',
        description = 'Mass of bone',
        min = 0.01,
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_mass_head')
    )
    bpy.types.PoseBone.wiggle_stiff_head = bpy.props.FloatProperty(
        name = 'Stiff',
        description = 'Spring stiffness coefficient, can be large numbers',
        min = 0,
        default = 400,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_stiff_head')
    )
    bpy.types.PoseBone.wiggle_stretch_head = bpy.props.FloatProperty(
        name = 'Stretch',
        description = 'Bone stretchiness factor, 0 to 1 range',
        min = 0,
        default = 0,
        max=1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_stretch_head')
    )
    bpy.types.PoseBone.wiggle_damp_head = bpy.props.FloatProperty(
        name = 'Damp',
        description = 'Dampening coefficient, can be greater than 1',
        min = 0,
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_damp_head')
    )
    bpy.types.PoseBone.wiggle_gravity_head = bpy.props.FloatProperty(
        name = 'Gravity',
        description = 'Multiplier for scene gravity',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_gravity_head')
    )
    bpy.types.PoseBone.wiggle_wind_ob_head = bpy.props.PointerProperty(
        name='Wind', 
        description='Wind force field object', 
        type=bpy.types.Object, 
        poll = wind_poll, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_prop(s, c, 'wiggle_wind_ob_head')
    )
    bpy.types.PoseBone.wiggle_wind_head = bpy.props.FloatProperty(
        name = 'Wind',
        description = 'Multiplier for wind forces',
        default = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_wind_head')
    )
    bpy.types.PoseBone.wiggle_chain_head = bpy.props.BoolProperty(
        name = 'Chain',
        description = 'Bone affects its parent creating a physics chain',
        default = True,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_chain_head')
    )
    
    #TAIL COLLISION
    
    bpy.types.PoseBone.wiggle_collider_type = bpy.props.EnumProperty(
        name='Collider Type',
        items=[('Object','Object','Collide with a selected mesh'),('Collection','Collection','Collide with all meshes in selected collection')],
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_collider_type')
    )
    bpy.types.PoseBone.wiggle_collider = bpy.props.PointerProperty(
        name='Collider Object', 
        description='Mesh object to collide with', 
        type=bpy.types.Object, 
        poll = collider_poll, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_prop(s, c, 'wiggle_collider')
    )
    bpy.types.PoseBone.wiggle_collider_collection = bpy.props.PointerProperty(
        name = 'Collider Collection', 
        description='Collection to collide with', 
        type=bpy.types.Collection, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_prop(s, c, 'wiggle_collider_collection')
    )
    
    bpy.types.PoseBone.wiggle_radius = bpy.props.FloatProperty(
        name = 'Radius',
        description = 'Collision radius',
        min = 0,
        default = 0,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_radius')
    )
    bpy.types.PoseBone.wiggle_friction = bpy.props.FloatProperty(
        name = 'Friction',
        description = 'Friction when colliding',
        min = 0,
        default = 0.5,
        soft_max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_friction')
    )
    bpy.types.PoseBone.wiggle_bounce = bpy.props.FloatProperty(
        name = 'Bounce',
        description = 'Bounciness when colliding',
        min = 0,
        default = 0.5,
        soft_max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_bounce')
    )
    bpy.types.PoseBone.wiggle_sticky = bpy.props.FloatProperty(
        name = 'Sticky',
        description = 'Margin beyond radius to keep item stuck to surface',
        min = 0,
        default = 0,
        soft_max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_sticky')
    )
    
    #HEAD COLLISION
    
    bpy.types.PoseBone.wiggle_collider_type_head = bpy.props.EnumProperty(
        name='Collider Type',
        items=[('Object','Object','Collide with a selected mesh'),('Collection','Collection','Collide with all meshes in selected collection')],
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_collider_type_head')
    )
    bpy.types.PoseBone.wiggle_collider_head = bpy.props.PointerProperty(
        name='Collider Object', 
        description='Mesh object to collide with', 
        type=bpy.types.Object, 
        poll = collider_poll, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_prop(s, c, 'wiggle_collider_head')
    )
    bpy.types.PoseBone.wiggle_collider_collection_head = bpy.props.PointerProperty(
        name = 'Collider Collection', 
        description='Collection to collide with', 
        type=bpy.types.Collection, 
        override={'LIBRARY_OVERRIDABLE'}, 
        update=lambda s, c: update_prop(s, c, 'wiggle_collider_collection_head')
    )
    
    bpy.types.PoseBone.wiggle_radius_head = bpy.props.FloatProperty(
        name = 'Radius',
        description = 'Collision radius',
        min = 0,
        default = 0,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_radius_head')
    )
    bpy.types.PoseBone.wiggle_friction_head = bpy.props.FloatProperty(
        name = 'Friction',
        description = 'Friction when colliding',
        min = 0,
        default = 0.5,
        soft_max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_friction_head')
    )
    bpy.types.PoseBone.wiggle_bounce_head = bpy.props.FloatProperty(
        name = 'Bounce',
        description = 'Bounciness when colliding',
        min = 0,
        default = 0.5,
        soft_max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_bounce_head')
    )
    bpy.types.PoseBone.wiggle_sticky_head = bpy.props.FloatProperty(
        name = 'Sticky',
        description = 'Margin beyond radius to keep item stuck to surface',
        min = 0,
        default = 0,
        soft_max = 1,
        override={'LIBRARY_OVERRIDABLE'},
        update=lambda s, c: update_prop(s, c, 'wiggle_sticky_head')
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
    bpy.utils.register_class(WiggleCopy)
    bpy.utils.register_class(WiggleSelect)
    bpy.utils.register_class(WiggleBake)
    bpy.utils.register_class(WIGGLE_PT_Settings)
    bpy.utils.register_class(WIGGLE_PT_Head)
    bpy.utils.register_class(WIGGLE_PT_Tail)
    bpy.utils.register_class(WIGGLE_PT_Utilities)
    bpy.utils.register_class(WIGGLE_PT_Bake)
    
    bpy.app.handlers.frame_change_pre.append(wiggle_pre)
    bpy.app.handlers.frame_change_post.append(wiggle_post)
    bpy.app.handlers.render_pre.append(wiggle_render_pre)
    bpy.app.handlers.render_post.append(wiggle_render_post)
    bpy.app.handlers.render_cancel.append(wiggle_render_cancel)
    bpy.app.handlers.load_post.append(wiggle_load)

def unregister():
    bpy.utils.unregister_class(WiggleBoneItem)
    bpy.utils.unregister_class(WiggleItem)
    bpy.utils.unregister_class(WiggleBone)
    bpy.utils.unregister_class(WiggleObject)
    bpy.utils.unregister_class(WiggleScene)
    bpy.utils.unregister_class(WiggleReset)
    bpy.utils.unregister_class(WiggleCopy)
    bpy.utils.unregister_class(WiggleSelect)
    bpy.utils.unregister_class(WiggleBake)
    bpy.utils.unregister_class(WIGGLE_PT_Settings)
    bpy.utils.unregister_class(WIGGLE_PT_Head)
    bpy.utils.unregister_class(WIGGLE_PT_Tail)
    bpy.utils.unregister_class(WIGGLE_PT_Utilities)
    bpy.utils.unregister_class(WIGGLE_PT_Bake)
    
    bpy.app.handlers.frame_change_pre.remove(wiggle_pre)
    bpy.app.handlers.frame_change_post.remove(wiggle_post)
    bpy.app.handlers.render_pre.remove(wiggle_render_pre)
    bpy.app.handlers.render_post.remove(wiggle_render_post)
    bpy.app.handlers.render_cancel.remove(wiggle_render_cancel)
    bpy.app.handlers.load_post.remove(wiggle_load)
    
if __name__ == "__main__":
    register()
