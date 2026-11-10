"""python tools/card_sheet.py OUT.png DIR...: front and side line drawings of every cards_seed*.json in each DIR, in the card colours."""
import glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
S = 260
def draw(cards, ax):
    im = Image.new('RGB', (S, S), (235, 235, 235)); d = ImageDraw.Draw(im)
    allp = np.concatenate([np.array(c['seq'])[:, :3] for c in cards])
    uv = lambda p: np.c_[p[:, 0] if ax == 0 else -p[:, 2], -p[:, 1]]  # y is up in the head frame
    lo, hi = uv(allp).min(0), uv(allp).max(0); sc = .9 * S / (hi - lo).max(); mid = (lo + hi) / 2  # fit the whole set
    for c in sorted(cards, key=lambda c: np.array(c['seq'])[:, 2].mean() * (1 if ax == 0 else 0)):
        p = np.array(c['seq'])[:, :3]; col = np.array(c.get('colors', [[60, 40, 30]]))
        col = col if col.max() > 1 else col * 255
        xy = S / 2 + (uv(p) - mid) * sc
        for i in range(len(xy) - 1):
            k = min(int(i / max(len(xy) - 1, 1) * len(col)), len(col) - 1)
            d.line([tuple(xy[i]), tuple(xy[i + 1])], fill=tuple(int(v) for v in col[k]), width=2)
    return im
out, dirs = sys.argv[1], sys.argv[2:]
rows = []
for dr in dirs:
    for f in sorted(glob.glob(os.path.join(dr, 'cards_seed*.json'))):
        cards = json.load(open(f)); r = Image.new('RGB', (2 * S, S + 16), 'white'); r.paste(draw(cards, 0), (0, 16)); r.paste(draw(cards, 2), (S, 16))
        ImageDraw.Draw(r).text((4, 2), f'{os.path.basename(dr)} {os.path.basename(f)[6:-5]}: {len(cards)} cards', fill='black'); rows.append(r)
cols = 4; sh = Image.new('RGB', (cols * 2 * S, -(-len(rows) // cols) * (S + 16)), 'white')
for i, r in enumerate(rows): sh.paste(r, ((i % cols) * 2 * S, (i // cols) * (S + 16)))
sh.save(out); print(out, len(rows))
