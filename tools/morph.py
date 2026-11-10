"""python -I tools/morph.py A.json B.json OUT.npz [FRAMES]: latent morph of card set A into card set B (head frame).

1. Encode every card of A and B with the HairDeck card VAE (posterior means, zdim 64).
2. Pair cards with the Hungarian method on cost = latent distance + root distance (each divided by its median).
   The cost matrix is padded, so a card can also take a "dissolve" (A) or "spawn" (B) entry. A pair that costs more
   than 2 * GATE becomes one dissolve plus one spawn.
3. Per frame, blend z and the root of each pair with a per-track weight (staggered start, smoothstep), decode,
   resample to K points. A dissolving card shrinks into its root, a spawning card grows out of its root.
   Colours blend in CIE Lab; root normals blend linearly and are normalised.
OUT.npz: P (F, T, K, 8) float32 cards (xyz, width, thickness, root normal), C (F, T, 16, 3) sRGB, kind (T,) 0 pair 1 dissolve 2 spawn,
W (F, T) blend weight of each track.
"""
import json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'HairDeck'))
K, GATE, STAGGER = 48, 1.5, 0.45  # points per card; dissolve/spawn cost in median units; share of the morph used for start offsets


def srgb_to_lab(c):
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T / [0.95047, 1.0, 1.08883]
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389) * [0.95047, 1.0, 1.08883]
    c = xyz @ np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]).T
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def match(zA, zB, rA, rB):
    """-> list of (i or None, j or None): pairs, dissolves (j None), spawns (i None)."""
    lat = np.linalg.norm(zA[:, None] - zB[None], axis=-1)
    root = np.linalg.norm(rA[:, None] - rB[None], axis=-1)
    cost = lat / np.median(lat) + root / max(np.median(root), 1e-6)
    a, b = len(zA), len(zB)
    big = 1e9
    M = np.zeros((a + b, b + a))
    M[:a, :b] = cost
    M[:a, b:] = np.where(np.eye(a, dtype=bool), GATE, big)  # A card i -> dissolve
    M[a:, :b] = np.where(np.eye(b, dtype=bool), GATE, big)  # spawn -> B card j
    r, c = linear_sum_assignment(M)
    out = []
    for i, j in zip(r, c):
        if i < a and j < b:
            out.append((i, j))
        elif i < a:
            out.append((i, None))
        elif j < b:
            out.append((None, j))
    return out


def weights(t, n, seed=0):
    """(n,) blend weight of each track at morph time t in [0, 1]: staggered starts, smoothstep ramps."""
    d = np.random.default_rng(seed).uniform(0, STAGGER, n)
    x = np.clip((t - d) / (1 - STAGGER), 0, 1)
    return x * x * (3 - 2 * x)


def morph(A, B, frames, vae):
    import torch
    from hairdeck.cards.vae import pad_seqs
    from hairdeck.cards.codec import resample
    sA = [np.asarray(c['seq'], np.float64) for c in A]
    sB = [np.asarray(c['seq'], np.float64) for c in B]
    zA, zB = (vae.encode_padded(*pad_seqs(s, vae.L)) for s in (sA, sB))
    rA, rB = np.array([s[0, :3] for s in sA]), np.array([s[0, :3] for s in sB])
    nA, nB = np.array([s[0, 5:8] for s in sA]), np.array([s[0, 5:8] for s in sB])
    lA, lB = (srgb_to_lab(np.asarray([c['colors'] for c in X], np.float64)) for X in (A, B))
    tracks = match(zA, zB, rA, rB)
    T = len(tracks)
    kind = np.array([0 if i is not None and j is not None else (1 if j is None else 2) for i, j in tracks], np.int8)
    print('cards A', len(A), 'B', len(B), 'tracks', T, 'pairs', (kind == 0).sum(), 'dissolve', (kind == 1).sum(), 'spawn', (kind == 2).sum())
    # per track: start and end latent, root, normal, Lab colours; a dissolve/spawn keeps its own card at both ends
    pick = lambda xa, xb, i, j: (xa[i] if i is not None else xb[j], xb[j] if j is not None else xa[i])
    z0, z1, r0, r1, n0, n1, c0, c1 = (np.array([pick(xa, xb, i, j)[e] for i, j in tracks])
                                      for xa, xb in ((zA, zB), (rA, rB), (nA, nB), (lA, lB)) for e in (0, 1))
    P = np.zeros((frames, T, K, 8), np.float32)
    C = np.zeros((frames, T, 16, 3), np.float32)
    W = np.zeros((frames, T), np.float32)
    for f in range(frames):
        w = W[f] = weights(f / (frames - 1), T)
        z = (1 - w[:, None]) * z0 + w[:, None] * z1
        root = (1 - w[:, None]) * r0 + w[:, None] * r1
        dec = vae.decode_cards(torch.from_numpy(z.astype(np.float32)).to(vae.device), root)
        grow = np.where(kind == 1, 1 - w, np.where(kind == 2, w, 1.0))  # dissolve shrinks, spawn grows, a pair stays
        for k, s in enumerate(dec):
            s = resample(s, K)
            s[:, :3] = root[k] + (s[:, :3] - root[k]) * max(grow[k], 1e-3)
            s[:, 3] *= grow[k]
            nrm = (1 - w[k]) * n0[k] + w[k] * n1[k]
            P[f, k, :, :5] = s
            P[f, k, :, 5:8] = nrm / max(np.linalg.norm(nrm), 1e-9)
        C[f] = lab_to_srgb((1 - w[:, None, None]) * c0 + w[:, None, None] * c1)
    return P, C, kind, W


if __name__ == '__main__':
    c = np.random.default_rng(1).uniform(0, 1, (50, 3))
    assert np.abs(lab_to_srgb(srgb_to_lab(c)) - c).max() < 2e-3, 'Lab round trip'
    zA, zB = np.eye(3), np.eye(4)[:, :3] * 1.0
    assert sorted(m for m in match(zA, zB[:4], zA, zB[:4]) if None not in m) == [(0, 0), (1, 1), (2, 2)], 'identity match'
    if len(sys.argv) < 4:
        print('self-check ok'); sys.exit()
    from hairdeck.cards.vae import load_vae
    vae = load_vae(os.path.join(ROOT, 'hd_ws', 'ckpt', 'card_vae'))
    A, B = (json.load(open(p)) for p in sys.argv[1:3])
    frames = int(sys.argv[4]) if len(sys.argv) > 4 else 180
    P, C, kind, W = morph(A, B, frames, vae)
    np.savez_compressed(sys.argv[3], P=P, C=C, kind=kind, W=W)
    print(sys.argv[3], P.shape)
