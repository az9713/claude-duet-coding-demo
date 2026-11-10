"""Hair on Base_Female in Blender.
  blender -b --factory-startup -P blender/morph_render.py -- picks [W]           six picked styles, front/side/back -> out/tests/picks/
  blender -b --factory-startup -P blender/morph_render.py -- morph NPZ OUT [W [F,F,..]]   hold 2 s, morph (NPZ frames), hold 2 s at 30 fps -> OUT/fNNNNN.png
The morph render skips frames that exist, so a stopped render resumes."""
import math, os, sys, time
import bpy
import numpy as np
from mathutils import Vector
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import rig, scene

PICKS = [('s1_b', 0), ('s2', 0), ('s3_a', 0), ('s4', 0), ('s5', 0), ('s6', 1)]
FPS, HOLD = 30, 60


def resample(s, K):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1))]
    t = np.linspace(0, d[-1], K)
    return np.stack([np.interp(t, d, s[:, c]) for c in range(s.shape[1])], 1)


def to_lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class Hair:
    """All tracks as one mesh with fixed topology: T cards x K points, 4 vertices per point."""

    def __init__(self, d, T, K):
        self.T, self.K = T, K
        self.k = np.clip(np.random.default_rng(0).normal(0.93, 0.07, T), 0.78, 1.0)  # per-card brightness (scene.add_hair)
        F = np.concatenate([rig.card_faces(K) + 4 * K * j for j in range(T)])
        m = bpy.data.meshes.new('hair')
        m.from_pydata(np.zeros((4 * K * T, 3)).tolist(), [], F.tolist())
        m.polygons.foreach_set('use_smooth', [True] * len(m.polygons))
        m.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
        m.materials.append(scene.hair_material(d))
        self.obj = bpy.data.objects.new('hair', m)
        bpy.context.scene.collection.objects.link(self.obj)
        scene.add_outline(self.obj, d['outline'], 0.0012)
        self.b = rig.body()

    def set(self, P, C):
        """P (T, K, 8) head frame, C (T, 16, 3) sRGB."""
        W = rig.place(P, self.b)
        V = np.concatenate([rig.card_verts(W[j], 'square') for j in range(self.T)])
        Cv = np.concatenate([rig.card_colors(W[j], C[j], self.k[j]) for j in range(self.T)])
        me = self.obj.data
        me.vertices.foreach_set('co', V.ravel())
        me.color_attributes['col'].data.foreach_set('color', np.c_[Cv, np.ones(len(Cv))].ravel())
        me.update()
        return to_lin(np.median(C[:, 0], axis=0)) * 0.8  # scalp colour: the median root colour, darker


def setup(res):
    d = scene.build(res=res)
    sc = bpy.context.scene
    sc.eevee.taa_render_samples = 16
    sc.render.fps = FPS
    # backdrop: a flat warm-grey wall behind her (she faces -Y)
    m = bpy.data.materials.new('wall')
    m.use_nodes = True
    m.node_tree.nodes.clear()
    e, o = m.node_tree.nodes.new('ShaderNodeEmission'), m.node_tree.nodes.new('ShaderNodeOutputMaterial')
    e.inputs[0].default_value = (0.62, 0.58, 0.55, 1)
    m.node_tree.links.new(e.outputs[0], o.inputs[0])
    me = bpy.data.meshes.new('wall')
    me.from_pydata([(-12, 1.5, -3), (12, 1.5, -3), (12, 1.5, 6), (-12, 1.5, 6)], [], [(0, 1, 2, 3)])
    me.materials.append(m)
    wall = bpy.data.objects.new('wall', me)
    sc.collection.objects.link(wall)
    return d, sc, wall


def shot(d, sc, path, az, dist=0.95, elev=4.0, dz=-0.02):
    h = scene.head_pos(d['arm'])
    scene.aim(d['cam'], (h.x, h.y, h.z + dz), az, dist, elev)
    a, e = math.radians(az + 18), math.radians(30)  # key light on the camera side (Regrowth film.py)
    d['sun'].rotation_euler = Vector((-math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), -math.sin(e))).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def picks(W):
    import json
    d, sc, wall = setup((W, W * 9 // 16))
    out = os.path.join(ROOT, 'out', 'tests', 'picks')
    os.makedirs(out, exist_ok=True)
    hairs = []
    for name, seed in PICKS:
        cards = json.load(open(os.path.join(ROOT, 'out', 'gen', name, f'cards_seed{seed}.json')))
        K = 48  # every card resampled to K points by arc length (tools/morph.py does the same)
        P = np.stack([resample(np.asarray(c['seq'], np.float64), K) for c in cards])
        h = Hair(d, len(cards), K)
        d['cap_col'].default_value = (*h.set(P, np.array([c['colors'] for c in cards])), 1)
        h.obj.hide_render = True
        hairs.append((name, h, d['cap_col'].default_value[:]))
    for name, h, cap in hairs:
        h.obj.hide_render = False
        d['cap_col'].default_value = cap
        for view, az in (('front', 20), ('side', 90), ('back', 180)):
            wall.hide_render = az > 100  # the wall stands behind her; the back view looks at it from the wrong side
            shot(d, sc, os.path.join(out, f'{name}_{view}.png'), az)
        h.obj.hide_render = True


def morph(npz, out, W, only=None):
    z = np.load(npz)
    P, C = z['P'], z['C']
    F, T, K = P.shape[:3]
    d, sc, wall = setup((W, W * 9 // 16))
    h = Hair(d, T, K)
    os.makedirs(out, exist_ok=True)
    N = HOLD + F + HOLD
    for f in only or range(N):
        p = os.path.join(out, f'f{f:05d}.png')
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        t0 = time.time()
        i = min(max(f - HOLD, 0), F - 1)
        d['cap_col'].default_value = (*h.set(P[i], C[i]), 1)
        az = 30 + 25 * f / (N - 1)  # a slow move round her, 30 to 55 degrees, so the back of the head shows
        shot(d, sc, p + '.tmp.png', az, dist=1.7, dz=0.02)  # whole hair in frame: the head bone sits at the skull base, 0.21 m under the skullcap top
        os.replace(p + '.tmp.png', p)
        print('FRAME', f, round(time.time() - t0, 2), flush=True)
    print('DONE')


a = sys.argv[sys.argv.index('--') + 1:]
if a[0] == 'picks':
    picks(int(a[1]) if len(a) > 1 else 960)
else:
    morph(os.path.abspath(a[1]), os.path.abspath(a[2]), int(a[3]) if len(a) > 3 else 1920,
          [int(x) for x in a[4].split(',')] if len(a) > 4 else None)  # optional frame list for test stills
