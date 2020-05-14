import numpy as np

# Image dimensions and grid lattice size, a.
width, height = 600, 400
a = 5

# a.sin(60) and a.cos(60)
gx, gy = a * np.sqrt(3)/2, a/2
# The number of unit cells, horizontally and vertically.
nx, ny = int(width / 2 / gx)+1, int(height / (a + gy))+1
# We'll need the coordinates of the centre of the image when we come to rotate.
cx, cy = width / 2, height / 2

def svg_line(x0, y0, x1, y1, th=0, cls=None):
    """Return the SVG for a single, line possibly rotated by th radians."""

    def rotate(x, y, th):
        """Rotate the coordinates (x,y) about the centre (cx,cy)."""
        c, s = np.cos(th), np.sin(th)
        xp, yp = x-cx, y-cy
        x, y = c*xp - s*yp, s*xp + c*yp
        return x+cx, y+cy

    if th != 0:
        x0, y0 = rotate(x0, y0, th)
        x1, y1 = rotate(x1, y1, th)

    # If an SVG class has been provided, add it to the line element.
    s_cls = 'class="{}"'.format(cls) if cls else ''
    return '<line x1="{}" y1="{}" x2="{}" y2="{}" {}/>'.format(
                                                    x0, y0, x1, y1, s_cls)

def add_unit_cell(s, x0, y0, th=0, cls=None):
    """Add a unit cell from the lattice to the SVG output.

    The "unit cell" consists of the arrangement of lines: \ /
                                                           |
    centred at the vertex where they meet. th is the angle of rotation, in
    radians and cls is an optional SVG class to add to the <line> element.

    """

    s.append(svg_line(x0, y0, x0, y0+a, th, cls))
    s.append(svg_line(x0, y0, x0-gx, y0-gy, th, cls))
    s.append(svg_line(x0, y0, x0+gx, y0-gy, th, cls))

def svg_preamble(s):
    """The usual SVG preamble and style definitions."""

    s.append('<?xml version="1.0" encoding="utf-8"?>')
    s.append('<svg xmlns="http://www.w3.org/2000/svg"\n' + ' '*5 +
         'xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}" >'
            .format(width, height))
    s.append("""<defs>
    <style type="text/css"><![CDATA[
    line {
        stroke-width: 2px;
        stroke: #000;
    }
    .lattice1 {
        stroke: #824e4e;
    }
    .lattice2 {
        stroke: #4a5f70;
    }
    >]]></style>
    </defs>
    """)

s = []
svg_preamble(s)

# Angle of rotation of the lattices (degrees)
th = 5.0
# An internal variable to offset every other row of the unit cells when drawing
_ph = 0
for iy in range(ny):
    _ph = 0 if _ph else gx
    for ix in range(nx):
        add_unit_cell(s, ix*2*gx + _ph, iy*(a+gy), cls='lattice1')
        add_unit_cell(s, ix*2*gx + _ph, iy*(a+gy), np.radians(th), 'lattice2')

s.append('</svg>')

with open('lattice.svg', 'w') as fo:
    print('\n'.join(s), file=fo)
