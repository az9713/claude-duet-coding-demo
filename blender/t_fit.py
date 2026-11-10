"""Head bone and skullcap (HairBack material) bounds of the HairDeck avatar (Regrowth) and of Base_Female, both turned to face -Y.
Prints JSON lines; tools/fit_head.py turns them into assets/fit.json."""
import bpy, os, json, math, shutil
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for key, vrm in (('hd', os.path.join(ROOT, '..', 'regrowth', 'assets', 'avatar', 'avatar.vrm')),
                 ('woman', os.path.join(ROOT, 'assets', 'avatar_src', 'base_female', 'Base_Female.vrm'))):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    g = vrm[:-4] + '_copy.glb'
    if not os.path.exists(g):
        shutil.copy(vrm, g)
    bpy.ops.import_scene.gltf(filepath=g)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    arm.rotation_mode = 'XYZ'
    arm.rotation_euler[2] = math.pi
    bpy.context.view_layer.update()
    head = list(arm.matrix_world @ arm.pose.bones['J_Bip_C_Head'].head)
    P = []
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        idx = [i for i, m in enumerate(o.data.materials) if m and 'HairBack' in m.name]
        if not idx:
            continue
        vs = {v for p in o.data.polygons if p.material_index in idx for v in p.vertices}
        P += [list(o.matrix_world @ o.data.vertices[v].co) for v in vs]
    P = np.array(P)
    print('FIT', json.dumps(dict(key=key, head=head, lo=P.min(0).tolist(), hi=P.max(0).tolist(), n=len(P),
                                 mats=[m.name for m in bpy.data.materials])))
