"""Metamorphosis: one life told by hair. Every frame is a function of its index, so a stopped render resumes.

blender -b --factory-startup -P blender/film.py -- stills 0,300,900 [W]   -> out/tests/film/fNNNNN.png
blender -b --factory-startup -P blender/film.py -- render START END [W]   -> out/frames/fNNNNN.png, existing files skipped
Inputs: out/final/m_*.npz (tools/morph.py), assets/final/*_chains.json, assets/plates/p1..p6.png.
Delete out/sim/*.npz after a change to head(), wind() or colliders().
"""
import json, math, os, shutil, sys, time
import bpy
import numpy as np
from mathutils import Matrix, Vector
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'blender')]
import rig, scene
sim = rig.sim

FPS, DUR = 30, 90
N = FPS * DUR
PIV = np.array([-0.000255, 0.007965, 1.387561])  # Base_Female head joint (blender/t_fit.py)
STYLES = ['s1_b', 's2', 's3_a', 's4', 's5', 's6']
MORPHS = [(13, 19), (27, 33), (40, 46), (54, 64), (70, 76)]  # morph i turns STYLES[i] into STYLES[i + 1]
FADES = [(14.5, 17.5), (28.5, 31.5), (41.5, 44.5), (56, 62), (71.5, 74.5)]  # plate i -> i + 1
KEY = dict(blink='target_13', fun='target_2', sorrow='target_4')  # VRM blendShapeGroups of Base_Female.vrm
BLINKS = [2.2, 6.1, 11.8, 17.0, 22.6, 25.9, 31.4, 36.8, 43.3, 49.0, 52.6, 58.4, 66.1, 69.2, 74.8, 79.9, 86.3]


def ss(a, b, t):
    x = min(max((t - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def bump(a, b, c, d, t):
    return ss(a, b, t) * (1 - ss(c, d, t))


# --- performance ---------------------------------------------------------------------------------------------------
def head(t):
    """World rotation of her head. She faces -Y. Yaw > 0 turns her face toward +X (screen right), pitch > 0 looks down."""
    yaw = 32 * bump(20.3, 20.9, 23.3, 23.9, t) - 10 * bump(78, 80, 88, 90, t)  # the ponytail swing; toward the child
    pitch = 8 * bump(46, 48, 53, 55, t) + 4 * bump(58, 62, 74, 77, t) - 3 * bump(34, 35.5, 38.5, 40, t)
    yaw += 1.5 * math.sin(.45 * t)
    pitch += .7 * math.sin(.33 * t + 1)
    roll = 1.0 * math.sin(.27 * t) + 4 * bump(34, 35.5, 38.5, 40, t)  # a happy tilt at the wedding
    a = Matrix.Rotation(math.radians(yaw), 3, 'Z') @ Matrix.Rotation(math.radians(pitch), 3, 'X') @ Matrix.Rotation(math.radians(roll), 3, 'Y')
    return np.array(a)


def face(t):
    close = max([0.0] + [1 - abs(t - b) / .09 for b in BLINKS])
    fun = .7 * bump(34, 35.5, 38.5, 40.5, t) + .45 * bump(3, 4.5, 9, 10.5, t) + .8 * bump(84, 86, 89, 90.5, t)
    sorrow = .35 * bump(46, 48, 53, 55, t) + .2 * bump(60, 64, 72, 75, t)
    return {KEY['blink']: close, KEY['fun']: fun, KEY['sorrow']: sorrow}


WIND_DIR = np.array([.9, .4, .05])


def wind(t):
    a = .015 * bump(-1, 0, 11, 13, t) + .01 * bump(25, 27, 32, 34, t) + .015 * bump(76, 78, 91, 92, t)  # 0.07 blew the pigtail straight out
    return a * (.65 + .35 * math.sin(.9 * t) * math.sin(.37 * t + 2))


# camera keys: time, azimuth, distance, height offset of the target from the head joint
CAM = [(0, 22, 1.75, .03), (13, 28, 1.7, .03), (19, 36, 1.75, .04), (27, 34, 1.7, .03), (33, 25, 1.55, .04), (40, 24, 1.6, .03),
       (46, 18, 1.65, .02), (54, 16, 1.75, .0), (64, 20, 1.9, -.03), (70, 18, 1.95, -.03), (77, 4, 2.0, -.05), (82, -6, 1.9, -.06),
       (90, -14, 1.85, .0)]


def camera(t):
    for (t0, a0, d0, h0), (t1, a1, d1, h1) in zip(CAM, CAM[1:]):
        if t <= t1:
            x = ss(t0, t1, t)
            return a0 + (a1 - a0) * x, d0 + (d1 - d0) * x, h0 + (h1 - h0) * x
    return CAM[-1][1:]


def cam_target(t):
    """Her head; in the last shot the target slides toward the grandchild's face."""
    a, d, h = camera(t)
    p = PIV + np.array([0, 0, h])
    return p + ss(82, 89, t) * (CHILD_FACE - p) * .55


def colliders(t):
    """(centre, radius): skull, chest, shoulders. Only the skull turns with the head."""
    R = head(t)
    return [(R @ (np.array([0, .013, 1.476]) - PIV) + PIV, .10), (np.array([0, -.01, 1.19]), .125),
            (np.array([.135, .01, 1.27]), .06), (np.array([-.135, .01, 1.27]), .06)]


def simulate(chains, t0, t1, path):
    """Regrowth film.py simulate: VRM spring update with wind and sphere colliders, 3 steps per frame."""
    if os.path.exists(path):
        z = np.load(path)
        return [(z[f'h{i}'], z[f'r{i}']) for i in range(len(chains))]
    dt = 1 / (3 * FPS)
    f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    rng = np.random.default_rng(5)
    st = []
    C0 = colliders(t0)
    for c in chains:
        J = c['J']
        d = np.diff(J, axis=0)
        L = np.linalg.norm(d, axis=1)
        mask = np.array([[np.linalg.norm(p - s) > r for s, r in C0] for p in J[1:]])
        st.append(dict(L=L, D=d / L[:, None], cur=J[1:].copy(), prev=J[1:].copy(), mask=mask, ph=rng.uniform(0, 6.28),
                       h=np.zeros((f1 - f0, len(L), 3)), r=np.zeros((f1 - f0, len(L), 3, 3))))
    for s in range((f1 - f0) * 3):
        t = t0 + s * dt
        R, S, W = head(t), colliders(t), wind(t)
        f, store = s // 3, s % 3 == 0
        for c, q in zip(chains, st):
            Wp, hd = R, R @ (c['J'][0] - PIV) + PIV
            for k in range(len(q['L'])):
                rest = Wp @ q['D'][k]
                w = WIND_DIR * W * (1 + .45 * math.sin(2.3 * t + q['ph'] + k * .8))
                nxt = q['cur'][k] + (q['cur'][k] - q['prev'][k]) * (1 - c['drag']) + rest * (c['stiff'] * dt) + w * dt
                for i, (ce, ra) in enumerate(S):
                    v = nxt - ce
                    n = np.linalg.norm(v)
                    if q['mask'][k, i] and n < ra:
                        nxt = ce + v / max(n, 1e-9) * ra
                v = nxt - hd
                n = np.linalg.norm(v)
                v = v / n if n > 1e-12 else rest
                nxt = hd + v * q['L'][k]
                q['prev'][k], q['cur'][k] = q['cur'][k], nxt
                Wk = sim.rotdiff(rest, v) @ Wp
                if store:
                    q['h'][f, k], q['r'][f, k] = hd, Wk
                Wp, hd = Wk, nxt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f'h{i}': q['h'] for i, q in enumerate(st)}, **{f'r{i}': q['r'] for i, q in enumerate(st)})
    return [(q['h'], q['r']) for q in st]


# --- hair ----------------------------------------------------------------------------------------------------------
def card_k(roots):
    """Per-card brightness 0.8..1 from the root position, so the same card has the same shade in two meshes."""
    h = np.sin(roots @ np.array([127.1, 311.7, 74.7]) * 40.0) * 43758.5453
    return 0.8 + 0.2 * (h - np.floor(h))


def world_verts(P, b):
    """(T, K, 8) head-frame cards -> (T*4K, 3) world ring vertices on Base_Female, front locks kept on her chest."""
    W = rig.place(P, b)
    W[..., :3] = rig.on_chest(W[..., :3])
    return np.concatenate([rig.card_verts(W[j], 'square') for j in range(len(W))]), W


class Style:
    """Chains and their simulation for one style, over the time it is visible."""

    def __init__(self, i, b):
        name = STYLES[i]
        chains = []
        for c in json.load(open(os.path.join(ROOT, 'assets', 'final', name + '_chains.json'))):
            J = rig.on_chest(rig.to_woman(rig.to_world(np.asarray(c['joints'], np.float64), b)))
            J = sim.clean(J)
            if len(J) >= 2:
                chains.append(dict(J=J, stiff=float(c['stiffness']), drag=0.4, grav=np.zeros(3)))
        self.chains = chains
        self.t0 = MORPHS[i - 1][0] if i > 0 else 0
        t1 = MORPHS[i][1] if i < len(MORPHS) else DUR
        self.f0 = int(round(self.t0 * FPS))
        self.sim = simulate(chains, self.t0, t1 + 1 / FPS, os.path.join(ROOT, 'out', 'sim', name + '.npz'))
        self.nf = len(self.sim[0][0]) if self.sim else 0

    def local(self, V, binding, f):
        """Local spring displacement (m, 3) of rest vertices V at film frame f, without the rigid head turn."""
        if not self.chains:
            return np.zeros_like(V)
        i = min(max(f - self.f0, 0), self.nf - 1)
        s = [(h[i:i + 1], r[i:i + 1]) for h, r in self.sim]
        R = head(f / FPS)[None]
        X = sim.deform(V, binding, self.chains, s, R, np.zeros((1, 3)), PIV)
        return sim.local_disp(X, V[None], R, np.zeros((1, 3)), PIV)[0]


class Morph:
    """One fixed-topology mesh for morph i; it also shows the hold before it (frame 0) and, for the last one, after it."""

    def __init__(self, d, i, b, styles):
        z = np.load(os.path.join(ROOT, 'out', 'final', f'm_{STYLES[i]}_{STYLES[i + 1]}.npz'))
        self.P, self.C, self.W = z['P'], z['C'], z['W']
        self.F, self.T, self.K = self.P.shape[:3]
        self.b, self.A, self.B = b, styles[i], styles[i + 1]
        self.t0, self.t1 = MORPHS[i]
        self.kA, self.kB = card_k(self.P[0, :, 0, :3]), card_k(self.P[-1, :, 0, :3])
        VA, _ = world_verts(self.P[0], b)
        VB, _ = world_verts(self.P[-1], b)
        cards = lambda V: list(V.reshape(self.T, 4 * self.K, 3))
        self.VA, self.VB = VA, VB
        self.bA, self.bB = sim.bind(cards(VA), self.A.chains, .17), sim.bind(cards(VB), self.B.chains, .17)
        F = np.concatenate([rig.card_faces(self.K) + 4 * self.K * j for j in range(self.T)])
        m = bpy.data.meshes.new(f'hair{i}')
        m.from_pydata(VA.tolist(), [], F.tolist())
        m.polygons.foreach_set('use_smooth', [True] * len(m.polygons))
        m.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
        m.materials.append(scene.hair_material(d))
        self.obj = bpy.data.objects.new(f'hair{i}', m)
        bpy.context.scene.collection.objects.link(self.obj)
        scene.add_outline(self.obj, d['outline'], 0.0012)

    def update(self, f):
        """Show the morph state for film frame f; returns the median root colour (sRGB)."""
        j = min(max(f - int(round(self.t0 * FPS)), 0), self.F - 1)
        w = self.W[j]
        V, Wd = world_verts(self.P[j], self.b)
        wv = np.repeat(w, 4 * self.K)[:, None]
        L = (1 - wv) * (self.A.local(self.VA, self.bA, f) if (wv < 1).any() else 0) + wv * (self.B.local(self.VB, self.bB, f) if (wv > 0).any() else 0)
        R = head(f / FPS)
        X = (V + L - PIV) @ R.T + PIV
        k = (1 - w) * self.kA + w * self.kB
        Cv = np.concatenate([rig.card_colors(Wd[t], self.C[j, t], k[t]) for t in range(self.T)])
        me = self.obj.data
        me.vertices.foreach_set('co', X.ravel())
        me.color_attributes['col'].data.foreach_set('color', np.c_[Cv, np.ones(len(Cv))].ravel())
        me.update()
        return np.median(self.C[j, :, 0], axis=0)


def to_lin(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# --- extra avatars ---------------------------------------------------------------------------------------------------
def import_avatar(rel):
    vrm = os.path.join(ROOT, 'assets', 'avatar_src', rel)
    g = vrm[:-4] + '_copy.glb'
    if not os.path.exists(g):
        shutil.copy(vrm, g)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=g)
    new = [o for o in bpy.data.objects if o not in before]
    for o in [o for o in new if o.type == 'MESH' and (o.name.startswith('Hair') or o.name.startswith('Icosphere'))]:
        bpy.data.objects.remove(o)
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == 'ARMATURE')
    arm.rotation_mode = 'XYZ'
    return arm, [o for o in new if o.type == 'MESH']


def place_bone(arm, bone, point):
    """Move the armature so that the bone's head sits at the world point."""
    bpy.context.view_layer.update()
    arm.location += Vector(point) - arm.matrix_world @ arm.pose.bones[bone].head
    bpy.context.view_layer.update()


def fix_arms(arm, sides):
    """Point each arm bone's tail at the next joint. The glTF importer makes short bones (0.17 m for both arm bones of
    AvatarSample_F), so an IK chain on them reached for the wrong point and the arm swung across her chest."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    for s in sides:
        eb[f'J_Bip_{s}_UpperArm'].tail = eb[f'J_Bip_{s}_LowerArm'].head
        eb[f'J_Bip_{s}_LowerArm'].tail = eb[f'J_Bip_{s}_Hand'].head
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()


def ik(arm, bone, name, pole=None):
    """IK on bone (chain 2: forearm and upper arm) toward a new empty. Returns (constraint, target empty, pole empty)."""
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    c = arm.pose.bones[bone].constraints.new('IK')
    c.target, c.chain_count = e, 2
    p = None
    if pole is not None:
        p = bpy.data.objects.new(name + '_pole', None)
        bpy.context.scene.collection.objects.link(p)
        p.location = pole
        c.pole_target, c.pole_angle = p, math.radians(-90)
    return c, e, p


def keep_groups(obj, keep):
    """Delete every vertex whose weight in the bone groups named by keep(name) is below 0.25."""
    import bmesh
    names = {g.index: g.name for g in obj.vertex_groups}
    me = obj.data
    drop = [v.index for v in me.vertices if sum(g.weight for g in v.groups if keep(names.get(g.group, ''))) < 0.25]  # 0.5 cut the elbow out
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in drop], context='VERTS')
    bm.to_mesh(me)
    bm.free()


def tint_cap(meshes, rgb):
    """Paint a VRoid skullcap (HairBack material) with one colour before the toon pass."""
    for o in meshes:
        for m in o.data.materials:
            if not (m and 'HairBack' in m.name and m.use_nodes):
                continue
            t = m.node_tree
            em = next((x for x in t.nodes if x.type == 'EMISSION'), None)
            if em and em.inputs['Color'].is_linked:
                src = em.inputs['Color'].links[0].from_socket
                mx = t.nodes.new('ShaderNodeMix')
                mx.data_type = 'RGBA'
                mx.inputs[0].default_value = 1.0
                t.links.new(src, mx.inputs['A'])
                mx.inputs['B'].default_value = (*rgb, 1)
                t.links.new(mx.outputs['Result'], em.inputs['Color'])


CHILD_FACE = np.array([0.27, -0.13, 1.28])  # updated in Film.__init__ from the placed child


class Film:
    def __init__(self, res):
        global CHILD_FACE
        self.d = d = scene.build(res=res)
        sc = self.sc = bpy.context.scene
        sc.eevee.taa_render_samples = 16
        sc.render.fps = FPS
        b = rig.body()
        self.styles = [Style(i, b) for i in range(len(STYLES))]
        self.morphs = [Morph(d, i, b, self.styles) for i in range(len(MORPHS))]
        arm = self.arm = d['arm']
        self.pb = arm.pose.bones[scene.HEAD_BONE]
        self.base = self.pb.matrix.copy()
        self.mw = arm.matrix_world.copy()
        self.keys = bpy.data.objects['Face'].data.shape_keys.key_blocks
        s1 = np.load(os.path.join(ROOT, 'out', 'final', 'm_s1_b_s2.npz'))
        root_col = to_lin(np.median(s1['C'][0, :, 0], axis=0)) * .8
        # the off-screen arm (AvatarSample_F, right arm only): faces -X from behind her left shoulder
        harm, hm = import_avatar('avatarsample_f/AvatarSample_F.vrm')
        harm.rotation_euler[2] = math.pi / 2
        for o in hm:
            keep_groups(o, lambda n: n.startswith('J_Bip_R_') and not any(k in n for k in ('Shoulder', 'Leg', 'Foot', 'Toe')))
        fix_arms(harm, 'R')
        place_bone(harm, 'J_Bip_R_UpperArm', (0.56, 0.22, 1.50))
        self.h_ik, self.h_t, _ = ik(harm, 'J_Bip_R_LowerArm', 'helper_t', pole=(0.5, 0.45, 1.1))
        self.h_rest = np.array([0.56, 0.22, 1.50]) + np.array([0.40, 0.05, -0.30])  # arm held out to the right, outside the frame (0.15 put the hand at the frame edge)
        # her own left arm for the push in stage 4: new bone tails, then the arms-down pose again (scene.build)
        fix_arms(arm, 'L')
        for bn in ('J_Bip_L_UpperArm', 'J_Bip_L_LowerArm', 'J_Bip_L_Hand'):
            arm.pose.bones[bn].matrix_basis = Matrix()
        bpy.context.view_layer.update()
        scene.world_rotate_bone(arm, 'J_Bip_L_UpperArm', (0, 1, 0), math.radians(68) * scene.side(arm, 'J_Bip_L_UpperArm'))
        self.w_ik, self.w_t, _ = ik(arm, 'J_Bip_L_LowerArm', 'her_t', pole=(0.45, 0.15, 1.05))
        self.w_rest = np.array(arm.matrix_world @ arm.pose.bones['J_Bip_L_Hand'].head)
        self.w_ik.influence = 0
        # the grandchild (AvatarSample_E) at her left side, where the s6 lock hangs (LOCK x = +0.066), raised so her hands reach it
        carm, cm = import_avatar('avatarsample_e/AvatarSample_E.vrm')
        carm.rotation_euler[2] = 2 * math.pi / 3  # she faces (-0.87, -0.5): toward the woman and partly toward the camera
        tint_cap(cm, root_col)
        fix_arms(carm, 'LR')
        place_bone(carm, 'J_Bip_C_Head', (0.27, -0.13, 1.22))
        self.c_ik = [ik(carm, f'J_Bip_{s}_LowerArm', f'child_{s}', pole=p) for s, p in (('L', (0.42, -0.39, 0.9)), ('R', (0.12, 0.13, 0.9)))]
        # rest targets: hands hanging 0.3 m under each shoulder (the IK stays on, so her T-pose never shows)
        self.c_rest = [np.array(carm.matrix_world @ carm.pose.bones[f'J_Bip_{s}_UpperArm'].head) + np.array([0, 0, -0.3]) for s in ('L', 'R')]
        self.helper_meshes = hm
        self.carm, self.carm_loc = carm, carm.location.copy()
        self.child_meshes = cm
        scene.toon_avatar(d['toon'])
        for o in hm + cm:
            scene.outline_with_alpha(o, d['outline'], 0.0016)
        cap = np.array([list(o.matrix_world @ o.data.vertices[v].co) for o in cm
                        for v in {v for p in o.data.polygons if o.data.materials[p.material_index] and 'HairBack' in o.data.materials[p.material_index].name for v in p.vertices}])
        cc = (cap.min(0) + cap.max(0)) / 2
        CHILD_FACE = cc + np.array([-0.05, -0.03, -0.06])
        self.child_hair(d, b, cc, s1)
        self.lock = self.side_lock()
        self.plate, self.vals = self.make_plate()
        self.cam = d['cam']

    def child_hair(self, d, b, cc, s1):
        """The s1_b pigtails on the child: her skullcap is 0.913 times Base_Female's; she is turned -60 degrees from Base_Female."""
        P = s1['P'][0]
        W = rig.place(P, b)
        R = np.array(Matrix.Rotation(-math.pi / 3, 3, 'Z'))
        W[..., :3] = (W[..., :3] - rig.WOMAN_C) @ R.T * 0.913 + cc
        W[..., 3:5] *= 0.913
        W[..., 5:8] = W[..., 5:8] @ R.T
        cards = [dict(seq=W[j], shape='square', colors=s1['C'][0, j]) for j in range(len(W)) if W[j, :, 3].max() > 1e-5]
        self.child_hair_obj = scene.add_hair(d, cards, 'child_hair')

    def side_lock(self):
        """World point on the s6 side lock where the child braids: the mean of the lowest 20% of its points."""
        P = self.morphs[-1].P[-1]
        V = rig.place(P, rig.body())[..., :3].reshape(-1, 3)
        low = V[V[:, 2] < np.quantile(V[:, 2], .2)]
        p = low.mean(0)
        print('LOCK', p.round(3))
        return p

    def make_plate(self):
        m = bpy.data.materials.new('plate')
        m.use_nodes = True
        t = m.node_tree
        t.nodes.clear()
        em, out = t.nodes.new('ShaderNodeEmission'), t.nodes.new('ShaderNodeOutputMaterial')
        tex = []
        for i in range(1, 7):
            n = t.nodes.new('ShaderNodeTexImage')
            n.image = bpy.data.images.load(os.path.join(ROOT, 'assets', 'plates', f'p{i}.png'))
            n.extension = 'EXTEND'
            tex.append(n)
        sock, vals = tex[0].outputs[0], []
        for src in tex[1:]:
            mx, v = t.nodes.new('ShaderNodeMix'), t.nodes.new('ShaderNodeValue')
            mx.data_type = 'RGBA'
            t.links.new(v.outputs[0], mx.inputs[0])
            t.links.new(sock, mx.inputs['A'])
            t.links.new(src.outputs[0], mx.inputs['B'])
            sock = mx.outputs['Result']
            vals.append(v)
        t.links.new(sock, em.inputs[0])
        t.links.new(em.outputs[0], out.inputs[0])
        me = bpy.data.meshes.new('plate')
        me.from_pydata([(-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0)], [], [(0, 1, 2, 3)])
        me.uv_layers.new().data.foreach_set('uv', [0, 0, 1, 0, 1, 1, 0, 1])
        me.materials.append(m)
        o = bpy.data.objects.new('plate', me)
        self.sc.collection.objects.link(o)
        return o, vals

    def hands(self, t):
        R = head(t)
        on_head = lambda p: R @ (np.asarray(p) - PIV) + PIV
        # 1. the off-screen hand ties the ribbon on her left pigtail (+X, near the camera)
        rib = on_head(PIV + np.array([0.105, 0.03, 0.10]))
        a = ss(2.5, 4.2, t) * (1 - ss(8.6, 10.6, t))
        tie = np.array([0.0, 0.02 * math.sin(7 * t), 0.018 * math.cos(7 * t)]) * bump(4.2, 4.6, 8.0, 8.6, t)
        tug = np.array([0.03, 0.0, -0.02]) * bump(8.0, 8.4, 8.6, 9.0, t)
        self.h_t.location = Vector(self.h_rest + a * (rib + np.array([0.08, 0.03, 0.02]) + tie + tug - self.h_rest))
        self.h_ik.influence = 1.0
        # 2. her own left hand pushes the bob back behind her ear
        g = bump(47.5, 48.7, 51.2, 52.6, t)
        sweep = ss(48.7, 51.0, t)
        temple = on_head(PIV + np.array([0.085, -0.075 + 0.13 * sweep, 0.09 - 0.03 * sweep]))
        self.w_t.location = Vector(self.w_rest + g * (temple - self.w_rest))
        self.w_ik.influence = bump(46.8, 47.2, 53.0, 53.4, t)  # blending IK with the FK pose swung the arm out sideways
        # 3. the grandchild's hands braid the side lock: they cross over and back
        c = bump(76.5, 78, 85, 86.5, t)
        for k, (con, e, _) in enumerate(self.c_ik):
            ph = 4.5 * t + k * math.pi
            p = self.lock + np.array([0.03 + 0.012 * math.sin(ph), -0.035 + 0.018 * math.cos(ph) + 0.02 * k, 0.04 + 0.03 * math.sin(.8 * t)])
            r = self.c_rest[k] + getattr(self, 'c_off', 0)
            e.location = Vector(r + c * (p - r))
            con.influence = 1.0

    def set(self, f):
        t = f / FPS
        R = head(t)
        P = Vector(PIV.tolist())
        self.pb.matrix = self.mw.inverted() @ (Matrix.Translation(P) @ Matrix(R.tolist()).to_4x4() @ Matrix.Translation(-P)) @ self.mw @ self.base
        for k, v in face(t).items():
            self.keys[k].value = v
        # hair: which mesh shows at t
        show = 0
        for j, (m0, m1) in enumerate(MORPHS):
            if t >= m0:
                show = j  # morph j, or the hold before it (its frame 0)
            if t >= m1 and j + 1 < len(MORPHS):
                show = j + 1
        for j, m in enumerate(self.morphs):
            m.obj.hide_render = j != show
        rc = self.morphs[show].update(f)
        self.d['cap_col'].default_value = (*(to_lin(rc) * .8), 1)
        self.child_vis(t)
        self.hands(t)
        az, dist, _ = camera(t)
        tgt = cam_target(t)
        scene.aim(self.cam, tuple(tgt), az, dist, 4.0)
        a, e = math.radians(az + 18), math.radians(30)
        self.d['sun'].rotation_euler = Vector((-math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), -math.sin(e))).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        M = self.cam.matrix_world
        D = dist + 2.6
        self.plate.matrix_world = M @ Matrix.Translation((0, 0, -D)) @ Matrix.Diagonal((1.15 * 2 * D * .3, 1.15 * 2 * D * .16875, 1, 1))
        for v, (a0, a1) in zip(self.vals, FADES):
            v.outputs[0].default_value = ss(a0, a1, t)
        # light through the decades: exposure follows the plates (rainy evening darker, winter softer)
        self.sc.view_settings.exposure = -.25 * bump(41.5, 44.5, 56, 62, t) - .1 * bump(56, 62, 71.5, 74.5, t)

    def child_vis(self, t):
        for o in self.child_meshes + [self.child_hair_obj]:
            o.hide_render = t < 74.4
        # she steps in from screen right, 74.5-77.5 s
        o = np.array([0.45, 0.10, 0.0]) * (1 - ss(74.5, 77.5, t))
        self.carm.location = self.carm_loc + Vector(o)
        self.child_hair_obj.location = Vector(o)
        self.c_off = o
        for o in self.helper_meshes:
            o.hide_render = t > 11.5

    def render(self, f, path):
        self.set(f)
        self.sc.render.filepath = path
        bpy.ops.render.render(write_still=True)


def main():
    a = sys.argv[sys.argv.index('--') + 1:]
    if a[0] == 'stills':
        W = int(a[2]) if len(a) > 2 else 960
        film = Film((W, W * 9 // 16))
        os.makedirs(os.path.join(ROOT, 'out', 'tests', 'film'), exist_ok=True)
        for f in [int(x) for x in a[1].split(',')]:
            t = time.time()
            film.render(f, os.path.join(ROOT, 'out', 'tests', 'film', f'f{f:05d}.png'))
            print('FRAME', f, round(time.time() - t, 2), flush=True)
    else:
        W = int(a[3]) if len(a) > 3 else 1920
        film = Film((W, W * 9 // 16))
        out = os.path.join(ROOT, 'out', 'frames')
        os.makedirs(out, exist_ok=True)
        for f in range(int(a[1]), min(int(a[2]), N)):
            p = os.path.join(out, f'f{f:05d}.png')
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
            t = time.time()
            film.render(f, p + '.tmp.png')
            os.replace(p + '.tmp.png', p)
            print('FRAME', f, round(time.time() - t, 2), flush=True)
    print('DONE')


main()
