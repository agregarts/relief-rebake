# -*- coding: utf-8 -*-
#
#  Relief Rebake - convert a messy relief mesh into a clean height field
#  Copyright (C) 2026  Agregart
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  Developed with AI assistance.
#
"""
================================================================================
 RELIEF REBAKE
================================================================================

 WHAT IT DOES
   Takes a messy relief mesh - AI-generated, photogrammetry, remeshed, or
   just badly distorted - and rebuilds it as a clean quad height field.

   It ray-casts the object from above onto a regular grid, cleans up the
   resulting height map, then builds a fresh watertight mesh with an even
   quad topology, straight side walls and a flat base.

 WHY YOU MIGHT NEED IT
   Image-to-3D tools produce relief panels with unusable topology: uneven
   triangle density, stretched faces, wasted geometry on the back plate.
   Retopology tools smooth them out and lose the detail. This keeps every
   height value and just rebuilds the grid underneath it.

   The output is a plain quad grid, so it displaces, subdivides, sculpts,
   bakes and exports to CAM cleanly.

 REQUIREMENTS
   Blender 3.0+ with numpy (bundled with Blender - nothing to install).
   The relief must face +Z. If it does not, rotate it and apply the
   rotation with Ctrl+A > Rotation before running.

 HOW TO USE
   1. Select the source mesh.
   2. Scripting workspace > Run Script (Alt+P).
   3. Window > Toggle System Console to watch progress and ETA.

 START SMALL
   The default RES of 1575 takes a few minutes. Run that first and check
   the result before raising it. Resolution cost is QUADRATIC:

     RES    grid           quads     time      RAM
     1000   1000 x  667     0.7M     ~3 min    ~0.5 GB
     1575   1575 x 1050     1.7M     ~7 min    ~1 GB
     3150   3150 x 2100     6.6M     ~30 min   ~4 GB
     4725   4725 x 3150    14.9M     ~60 min   ~8-12 GB
     6300   6300 x 4200    26.5M     ~2 hours  ~16-24 GB

   Doubling RES quadruples the work. Times are rough and depend on the
   source mesh complexity.

 SETTINGS
   RES             output grid width. See table above.
   DESPECKLE       removes single-pixel spikes from the ray cast.
   SMOOTH          gaussian blur on the height field. 0 = off (default).
                   Anything above 0 will soften your detail.
   DETAIL_BOOST    only used when SMOOTH > 0; re-adds fine detail.
   FLATTEN_FLOOR   heights below this fraction snap to the base plane.
                   0.015 removes ray-cast noise in the background.
   RELIEF_DEPTH    0 = keep the original depth. Set a value to rescale.
   BASE            thickness of the solid base under the relief.
   MAKE_UV         off by default. At high RES the UV layer alone can
                   cost hundreds of MB.
   FAST_BUILD      writes geometry straight from numpy arrays. If it ever
                   fails the script falls back to from_pydata automatically.

 NOTES
   The source object is hidden, not deleted. Unhide it from the Outliner
   if you need it back.

   Overhangs are lost by design - a height field stores one Z per XY
   position. For relief panels this is not a limitation; for undercut
   sculpture it is.

================================================================================
"""

import bpy
import bmesh
import math
import time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# ==============================================================================
CFG = {
    "RES":            1575,    # start here, raise once you like the result
    "DESPECKLE":      True,
    "SMOOTH":         0.0,     # 0 = surface is never touched
    "DETAIL_BOOST":   0.0,
    "FLATTEN_FLOOR":  0.015,
    "RELIEF_DEPTH":   0.0,     # 0 = keep original depth
    "BASE":           3.0,
    "OBJECT_NAME":    "Relief_Clean",
    "SHADE_SMOOTH":   True,
    "TRIANGULATE":    False,
    "MAKE_UV":        False,   # costs memory at high RES
    "FAST_BUILD":     True,    # numpy direct write; set False if it misbehaves
}
# ==============================================================================


def gaussian_blur(a, sigma):
    if sigma <= 0:
        return a
    r = max(1, int(math.ceil(sigma * 3)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode='edge')
    o = np.zeros_like(a)
    for i in range(k.size):
        o += k[i] * p[i:i + a.shape[0], :]
    p = np.pad(o, ((0, 0), (r, r)), mode='edge')
    o2 = np.zeros_like(o)
    for i in range(k.size):
        o2 += k[i] * p[:, i:i + o.shape[1]]
    return o2


def median3(a):
    """3x3 median filter, written into a single preallocated buffer."""
    p = np.pad(a, 1, mode='edge')
    st = np.empty((9,) + a.shape, dtype=np.float32)
    n = 0
    for i in range(3):
        for j in range(3):
            st[n] = p[i:i + a.shape[0], j:j + a.shape[1]]
            n += 1
    out = np.median(st, axis=0).astype(np.float32)
    del st
    return out


def sample_heightfield(obj, res):
    """Ray-cast the object from above onto a regular grid."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0)
    vs = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    ev.to_mesh_clear()
    del verts, polys

    xmin, ymin, zmin = vs.min(0)
    xmax, ymax, zmax = vs.max(0)
    del vs
    sx, sy = xmax - xmin, ymax - ymin
    if sx <= 0 or sy <= 0:
        raise ValueError("Object has no XY extent. The relief must face +Z.")

    if sx >= sy:
        W = int(res)
        H = max(2, int(round(res * sy / sx)))
    else:
        H = int(res)
        W = max(2, int(round(res * sx / sy)))

    print("[rebake] grid: %d x %d = %.2fM rays" % (W, H, W * H / 1e6))

    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)
    top = zmax + max(sx, sy) * 0.1
    down = Vector((0.0, 0.0, -1.0))

    hf = np.full((H, W), np.nan, dtype=np.float32)
    hit = 0
    t0 = time.time()
    step = max(1, H // 50)
    origin = Vector((0.0, 0.0, top))
    cast = tree.ray_cast

    for j in range(H):
        origin.y = float(ys[j])
        row = hf[j]
        for i in range(W):
            origin.x = float(xs[i])
            loc = cast(origin, down)[0]
            if loc is not None:
                row[i] = loc.z
                hit += 1
        if j % step == 0 and j:
            el = time.time() - t0
            eta = el * (H - j) / j
            print("[rebake] %d%%  elapsed %.1f min  remaining ~%.1f min"
                  % (int(100 * j / H), el / 60.0, eta / 60.0))

    print("[rebake] ray casting done: %.1f min" % ((time.time() - t0) / 60.0))
    print("[rebake] hit rate: %.1f%%" % (100.0 * hit / (W * H)))
    if hit == 0:
        raise ValueError("No rays hit the object. Check the facing direction.")

    miss = np.isnan(hf)
    if miss.any():
        print("[rebake] filling gaps: %d cells" % int(miss.sum()))
        hf[miss] = zmin
        for _ in range(3):
            f = median3(hf)
            hf[miss] = f[miss]
    return hf, (zmin, zmax)


def process(hf, zr, c):
    zmin, zmax = zr
    rng = zmax - zmin
    if rng <= 0:
        raise ValueError("Height range is zero.")
    h = (hf - zmin) / rng

    if c["DESPECKLE"]:
        m = median3(h)
        spike = np.abs(h - m) > 0.06
        h = np.where(spike, m, h)
        print("[rebake] spikes removed: %d px" % int(spike.sum()))
        del m, spike

    if c["SMOOTH"] > 0:
        bl = gaussian_blur(h, c["SMOOTH"])
        if c["DETAIL_BOOST"] > 0:
            fine = gaussian_blur(h, c["SMOOTH"] * 0.4)
            h = bl + (fine - bl) * c["DETAIL_BOOST"]
        else:
            h = bl

    t = c["FLATTEN_FLOOR"]
    if t > 0:
        h = np.clip((h - t) / (1.0 - t), 0.0, 1.0)
    return np.clip(h, 0.0, 1.0).astype(np.float32), rng


def _geometry_arrays(hm, size_x, size_y, depth, base):
    """Build all geometry as numpy arrays. No Python lists."""
    H, W = hm.shape
    xs = np.linspace(-size_x * .5, size_x * .5, W, dtype=np.float32)
    ys = np.linspace(-size_y * .5, size_y * .5, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z = base + hm * depth

    top_co = np.stack([X, Y, Z], -1).reshape(-1, 3)
    del X, Y, Z

    idx = np.arange(W * H, dtype=np.int32).reshape(H, W)
    top_f = np.stack([idx[:-1, :-1], idx[:-1, 1:],
                      idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    n_top = top_f.shape[0]

    # perimeter ring: top edge -> right -> bottom -> left
    per = np.concatenate([
        idx[0, :],
        idx[1:, W - 1],
        idx[H - 1, W - 2::-1],
        idx[H - 2:0:-1, 0],
    ]).astype(np.int32)
    P = per.size
    bs = W * H

    skirt_co = top_co[per].copy()
    skirt_co[:, 2] = 0.0

    nxt = np.roll(np.arange(P, dtype=np.int32), -1)
    side_f = np.stack([per, bs + np.arange(P, dtype=np.int32),
                       bs + nxt, per[nxt]], -1)

    cen = bs + P
    cen_co = np.zeros((1, 3), dtype=np.float32)
    bot_f = np.stack([np.full(P, cen, dtype=np.int32),
                      bs + nxt,
                      bs + np.arange(P, dtype=np.int32)], -1)

    co = np.concatenate([top_co, skirt_co, cen_co], 0).astype(np.float32)
    return co, top_f, side_f, bot_f, n_top


def build_mesh_fast(hm, size_x, size_y, depth, c):
    co, top_f, side_f, bot_f, n_top = _geometry_arrays(
        hm, size_x, size_y, depth, float(c["BASE"]))

    nv = co.shape[0]
    n_quad = top_f.shape[0] + side_f.shape[0]
    n_tri = bot_f.shape[0]
    n_poly = n_quad + n_tri
    n_loop = n_quad * 4 + n_tri * 3
    print("[rebake] %d verts, %d polys, %d loops" % (nv, n_poly, n_loop))

    mesh = bpy.data.meshes.new(c["OBJECT_NAME"])
    mesh.vertices.add(nv)
    mesh.loops.add(n_loop)
    mesh.polygons.add(n_poly)

    mesh.vertices.foreach_set("co", co.ravel())
    del co

    loop_verts = np.concatenate([top_f.ravel(), side_f.ravel(), bot_f.ravel()])
    mesh.loops.foreach_set("vertex_index", loop_verts)
    del loop_verts

    totals = np.empty(n_poly, dtype=np.int32)
    totals[:n_quad] = 4
    totals[n_quad:] = 3
    starts = np.zeros(n_poly, dtype=np.int32)
    np.cumsum(totals[:-1], out=starts[1:])

    mesh.polygons.foreach_set("loop_start", starts)
    try:
        mesh.polygons.foreach_set("loop_total", totals)
    except Exception:
        pass          # Blender 4.1+ derives loop_total from offsets

    if c["SHADE_SMOOTH"]:
        sm = np.zeros(n_poly, dtype=bool)
        sm[:n_top] = True
        mesh.polygons.foreach_set("use_smooth", sm)

    mesh.update(calc_edges=True)
    ok = mesh.validate(verbose=False)
    if len(mesh.polygons) < n_poly * 0.99:
        raise RuntimeError("fast path failed validation (%d/%d polys)"
                           % (len(mesh.polygons), n_poly))
    if ok:
        print("[rebake] note: validate corrected some geometry")

    if c["MAKE_UV"]:
        _add_uv(mesh, hm.shape[1], hm.shape[0])
    return mesh


def build_mesh_slow(hm, size_x, size_y, depth, c):
    """Fallback: from_pydata. Slower and memory hungry, but reliable."""
    print("[rebake] falling back to from_pydata - this will be slow")
    co, top_f, side_f, bot_f, n_top = _geometry_arrays(
        hm, size_x, size_y, depth, float(c["BASE"]))
    faces = top_f.tolist() + side_f.tolist() + bot_f.tolist()
    mesh = bpy.data.meshes.new(c["OBJECT_NAME"])
    mesh.from_pydata(co.astype(np.float64).tolist(), [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    if c["SHADE_SMOOTH"]:
        sm = np.zeros(len(mesh.polygons), dtype=bool)
        sm[:n_top] = True
        mesh.polygons.foreach_set("use_smooth", sm)
        mesh.update()
    if c["MAKE_UV"]:
        _add_uv(mesh, hm.shape[1], hm.shape[0])
    return mesh


def _add_uv(mesh, W, H):
    uvl = mesh.uv_layers.new(name="UVMap")
    nl = len(mesh.loops)
    lv = np.empty(nl, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", lv)
    uv = np.zeros((nl, 2), dtype=np.float32)
    m = lv < (W * H)
    tv = lv[m]
    uv[m, 0] = (tv % W) / float(W - 1)
    uv[m, 1] = (tv // W) / float(H - 1)
    uvl.data.foreach_set("uv", uv.ravel())


def main():
    c = CFG
    src = bpy.context.active_object
    if src is None or src.type != 'MESH':
        raise ValueError("Select the source mesh object first.")

    t_all = time.time()
    print("=" * 60)
    print("[rebake] source:", src.name, "| RES =", c["RES"])
    print("=" * 60)

    hf, zr = sample_heightfield(src, int(c["RES"]))
    hm, rng = process(hf, zr, c)
    del hf

    bb = [src.matrix_world @ Vector(v) for v in src.bound_box]
    xs = [v.x for v in bb]
    ys = [v.y for v in bb]
    size_x, size_y = max(xs) - min(xs), max(ys) - min(ys)
    depth = float(c["RELIEF_DEPTH"]) if c["RELIEF_DEPTH"] > 0 else rng

    print("[rebake] grid %dx%d - building mesh..." % (hm.shape[1], hm.shape[0]))
    t0 = time.time()
    mesh = None
    if c["FAST_BUILD"]:
        try:
            mesh = build_mesh_fast(hm, size_x, size_y, depth, c)
        except Exception as ex:
            print("[rebake] fast path failed:", ex)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            mesh = None
    if mesh is None:
        mesh = build_mesh_slow(hm, size_x, size_y, depth, c)
    print("[rebake] mesh build: %.1f s" % (time.time() - t0))

    if c["TRIANGULATE"]:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    name = c["OBJECT_NAME"]
    old = bpy.data.objects.get(name)
    if old and old.type == 'MESH':
        od = old.data
        old.data = mesh
        if od.users == 0:
            bpy.data.meshes.remove(od)
        obj = old
    else:
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)

    src.hide_set(True)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print("=" * 60)
    print("[rebake] DONE ->", obj.name, "|", len(mesh.vertices), "verts")
    print("[rebake] total time: %.1f min" % ((time.time() - t_all) / 60.0))
    print("[rebake] source object hidden (unhide from the Outliner)")
    print("=" * 60)


if __name__ == "__main__":
    main()
