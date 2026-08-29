# relief-rebake
Rebuilds messy relief meshes as clean quad height fields. Ray-casts the object from above, cleans the height map, then builds a fresh watertight mesh with even quads, straight walls and a flat base. Keeps every height value; only the grid underneath is replaced. Useful for AI-generated, scanned or remeshed relief panels.
