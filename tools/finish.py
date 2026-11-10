"""python -I tools/finish.py: title and credit PNGs, 90 s music edit, final encode -> out/metamorphosis.mp4 (copied from Regrowth)."""
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = os.environ.get('FFMPEG') or __import__('imageio_ffmpeg').get_ffmpeg_exe()
out = os.path.join(ROOT, 'out')
FONT = r'C:\Windows\Fonts\georgia.ttf'


def text(path, lines, x, y):
    """White text with a dark shadow, so it reads on the sky and on the tree. lines: (text, size)."""
    im = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for s, size in lines:
        f = ImageFont.truetype(FONT, size)
        w = d.textlength(s, font=f)
        for dx, dy, c in ((3, 3, (40, 28, 32, 170)), (0, 0, (255, 252, 248, 255))):
            d.text((x - w / 2 + dx, y + dy), s, font=f, fill=c)
        y += int(size * 1.35)
    im.save(path)


text(os.path.join(out, 'title.png'), [('Metamorphosis', 96)], 1480, 120)
text(os.path.join(out, 'credit.png'), [('Hair made with HairDeck', 40), ('six styles, five morphs in the card latent space', 28)], 1500, 70)  # x 470 covered her head in the last shot

# Music: the score is 176.5 s. Its first 65 s carry the childhood-to-motherhood part; its last 26.5 s are the quiet ending.
# Film time: 0-65 from the start, 63-90 from 149.5 s, with a 2 s crossfade.
music = os.path.join(ROOT, 'assets', 'music', 'score.mp3')
fc = ('[0:a]atrim=0:65,asetpts=N/SR/TB[s1];[0:a]atrim=149.5:176.5,asetpts=N/SR/TB[s2];'
      '[s1][s2]acrossfade=d=2,afade=t=in:d=1,afade=t=out:st=86:d=4,atrim=0:90,volume=-1dB[a]')
audio = os.path.join(out, 'music.m4a')
subprocess.run([FF, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', music, '-filter_complex', fc, '-map', '[a]', '-c:a', 'aac', '-b:a', '192k', audio], check=True)

fv = ('[0:v]fade=t=in:st=0:d=1.2,fade=t=out:st=88.6:d=1.4[b];'
      '[1:v]format=rgba,fade=t=in:st=0.8:d=1.5:alpha=1,fade=t=out:st=4.5:d=1.5:alpha=1[t];'
      '[2:v]format=rgba,fade=t=in:st=84.5:d=1.5:alpha=1,fade=t=out:st=88.2:d=1:alpha=1[c];'
      '[b][t]overlay=shortest=1[x];[x][c]overlay=shortest=1,format=yuv420p[v]')
mp4 = os.path.join(out, 'metamorphosis.mp4')
subprocess.run([FF, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-framerate', '30', '-i', os.path.join(out, 'frames', 'f%05d.png'),
                '-loop', '1', '-framerate', '30', '-t', '90', '-i', os.path.join(out, 'title.png'),
                '-loop', '1', '-framerate', '30', '-t', '90', '-i', os.path.join(out, 'credit.png'),
                '-i', audio, '-filter_complex', fv, '-map', '[v]', '-map', '3:a', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
                '-c:a', 'copy', '-movflags', '+faststart', '-shortest', mp4], check=True)
print('done', mp4, os.path.getsize(mp4))
