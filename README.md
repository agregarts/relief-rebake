# Relief Rebake

A Blender add-on that rebuilds messy relief meshes as clean quad
height fields.

Ray-casts the object from above onto a regular grid, cleans the height
map, then builds a fresh watertight mesh with even quads, straight side
walls and a flat base. Every height value is kept; only the grid
underneath is replaced.

Useful for AI-generated, scanned or remeshed relief panels where
retopology tools would smooth away the detail.

## Install

Download the zip and use Edit > Preferences > Get Extensions >
Install from Disk.

## Notes

- The relief must face +Z
- Resolution cost is quadratic; start at the default 1575
- Overhangs are lost by design (a height field stores one Z per XY)
- Requires only numpy, which ships with Blender

## License

GPL-3.0-or-later
