"""Chiller schematic: the refrigerant cycle, its two water circuits and the instruments the model reads.

Run from the repository root: python -m src.visualization.drawing [light|dark]
The PNG is rendered from the SVG with: rsvg-convert -z 2 docs/chiller.svg -o docs/chiller.png
"""

import sys

THEME = sys.argv[1] if len(sys.argv) > 1 else "light"
PALETTES = {
    "light": dict(bg="#F5F5F7", ink="#1D1D1F", line="#8A8A8F", sec="#6E6E73", arrow="#B9401B",
                  hot="#F7C2A8", refrigerant="#FBE6DC", cold="#E3E3E8"),
    "dark": dict(bg="#242427", ink="#F5F5F7", line="#7C7C82", sec="#A1A1A6", arrow="#FF9466",
                 hot="#6B3423", refrigerant="#3E2B25", cold="#38383D"),
}
P = PALETTES[THEME]
BG, INK, LINE, SEC = P["bg"], P["ink"], P["line"], P["sec"]
W, H = 1200, 820
o = []
a = o.append


def text(x, y, s, size=11, c=SEC, anchor="middle", weight=400):
    a(f'<text x="{x}" y="{y}" font-size="{size}" fill="{c}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def line(x1, y1, x2, y2, w=0.9, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="{w}"{d} stroke-linecap="round"/>')


def pipe(points, fill, width=18):
    pts = " ".join(f"{x},{y}" for x, y in points)
    a(f'<polyline points="{pts}" fill="none" stroke="{LINE}" stroke-width="{width}" stroke-linejoin="miter"/>')
    a(f'<polyline points="{pts}" fill="none" stroke="{fill}" stroke-width="{width - 3.2}" stroke-linejoin="miter"/>')


def arrow(x, y, direction):
    s = 5.5
    shape = {"up": f"{x-s},{y+s} {x},{y-s} {x+s},{y+s}", "down": f"{x-s},{y-s} {x},{y+s} {x+s},{y-s}",
             "left": f"{x+s},{y-s} {x-s},{y} {x+s},{y+s}", "right": f"{x-s},{y-s} {x+s},{y} {x-s},{y+s}"}[direction]
    a(f'<polygon points="{shape}" fill="{P["arrow"]}" fill-opacity="0.55"/>')


def bubble(x, y, tag, number):
    a(f'<circle cx="{x}" cy="{y}" r="18" fill="{BG}" stroke="{LINE}" stroke-width="1.2"/>')
    text(x, y - 1, tag, 10.5, INK, weight=600)
    text(x, y + 11, number, 9.5, INK)


def shell(x, y, w, h, fill, name):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}" stroke="{LINE}" stroke-width="1.6"/>')
    for i in range(2):  # the tube bundle
        ty = y + h * (0.26 + 0.48 * i)
        line(x + h / 2, ty, x + w - h / 2, ty, 0.8, "6 5")
    text(x + w / 2, y + h / 2 + 4, name, 12, INK, weight=500)


a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter, SF Pro Text, Helvetica Neue, Arial">')
a(f'<rect width="{W}" height="{H}" rx="24" fill="{BG}"/>')
text(W / 2, 54, "Chiller", 27, INK, weight=600)
text(W / 2, 80, "Water-cooled centrifugal chiller on the vapour-compression cycle", 13)

# refrigerant circuit: condenser, expansion valve, evaporator, compressor
cx0, cx1, cy, ey, sh = 300, 860, 250, 520, 84
pipe([(cx0 + 30, cy + sh), (cx0 + 30, 385), (220, 385), (220, ey + sh / 2), (cx0, ey + sh / 2)], P["refrigerant"], 14)  # liquid line
pipe([(cx1, ey + sh / 2), (1000, ey + sh / 2), (1000, 420)], P["refrigerant"], 14)   # suction
pipe([(1000, 360), (1000, cy + sh / 2), (cx1, cy + sh / 2)], P["refrigerant"], 14)   # discharge
arrow(1000, 470, "up"); arrow(930, cy + sh / 2, "left"); arrow(220, 470, "down")
# expansion valve
a(f'<polygon points="208,{430-12} 232,{430-12} 220,430" fill="{BG}" stroke="{LINE}" stroke-width="1.2"/>')
a(f'<polygon points="208,{430+12} 232,{430+12} 220,430" fill="{BG}" stroke="{LINE}" stroke-width="1.2"/>')
text(196, 434, "Expansion valve", 10.5, SEC, "end")

# water circuits: condenser water above, chilled water below
for x, label, direction in ((450, "in", "down"), (710, "out", "up")):
    pipe([(x, cy), (x, 140)], P["hot"])
    arrow(x, 175, direction)
    text(x, 128, f"Condenser water {label}", 10.5)
    pipe([(x, ey + sh), (x, 700)], P["cold"])
    arrow(x, 670, "up" if label == "in" else "down")
    text(x, 720, f"Chilled water {label}", 10.5)

shell(cx0, cy, cx1 - cx0, sh, P["refrigerant"], "Condenser")
shell(cx0, ey, cx1 - cx0, sh, P["refrigerant"], "Evaporator")

# compressor and motor
a(f'<circle cx="1000" cy="390" r="32" fill="{BG}" stroke="{LINE}" stroke-width="1.6"/>')
a(f'<polygon points="982,372 1018,380 1018,400 982,408" fill="none" stroke="{LINE}" stroke-width="1.2" stroke-linejoin="round"/>')
text(1016, 445, "Compressor", 10.5, SEC, "start")
line(1032, 387, 1072, 387, 0.8); line(1032, 393, 1072, 393, 0.8)
a(f'<circle cx="1096" cy="390" r="24" fill="{BG}" stroke="{LINE}" stroke-width="1.6"/>')
text(1096, 395, "M", 14, INK, weight=500)

# instruments, ISA-5.1 tags
for x, number in ((450, "103"), (710, "104")):
    line(x - 9, 195, x - 40, 195); bubble(x - 58, 195, "TT", number)
for x, number in ((450, "101"), (710, "102")):
    line(x - 9, 640, x - 40, 640); bubble(x - 58, 640, "TT", number)
line(710 + 9, 160, 750, 160); bubble(768, 160, "FT", "102")
line(710 + 9, 675, 750, 675); bubble(768, 675, "FT", "101")
line(800, cy, 800, cy - 22); bubble(800, cy - 40, "PT", "102")
line(800, ey + sh, 800, ey + sh + 22); bubble(800, ey + sh + 40, "PT", "101")
line(cx0 + 30 + 7, 345, 360, 345); bubble(378, 345, "TT", "105")
line(968, 390, 946, 390); bubble(928, 390, "TT", "106")
line(1096, 366, 1096, 338); bubble(1096, 320, "JT", "101")

# legend of what each tag is
notes = [("TT 101–104", "Water temperatures"), ("FT 101, 102", "Water flows"), ("PT 101, 102", "Refrigerant pressures"),
         ("TT 105", "Liquid line, for subcooling"), ("TT 106", "Oil sump"), ("JT 101", "Compressor power")]
cw = 1020 / 3
for i, (tag, meaning) in enumerate(notes):
    x, y = 100 + (i % 3) * cw, 758 + (i // 3) * 26
    text(x, y, tag, 11, INK, "start", 600)
    text(x + 92, y, meaning, 11, SEC, "start")

a("</svg>")
open(f"docs/chiller{'' if THEME == 'light' else '-dark'}.svg", "w").write("\n".join(o))
