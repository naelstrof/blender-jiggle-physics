all: jiggle-physics.zip

jiggle-physics.zip: __init__.py blender_manifest.toml
	blender --command extension build
