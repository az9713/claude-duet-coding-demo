"""Print Base_Female's most forward y per height (all meshes, posed, 0.04 < |x| < 0.12 where the front locks hang) -> rig.CHEST_Z/CHEST_Y."""
import bpy, sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'blender')]
import scene
d = scene.build(res=(64, 64))
dg = bpy.context.evaluated_depsgraph_get()
V = []
for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    for mod in o.modifiers:
        if mod.type == 'SOLIDIFY':
            mod.show_viewport = False
    dg.update()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    V += [list(e.matrix_world @ v.co) for v in me.vertices]
V = np.array(V)
for z in np.arange(1.37, 0.98, -.03):
    s = V[(abs(V[:, 2] - z) < .015) & (abs(V[:, 0]) > .04) & (abs(V[:, 0]) < .12)]
    if len(s):
        print('CHEST z %.2f ymin %.3f n %d' % (z, s[:, 1].min(), len(s)))
