all: jiggle-physics.zip

jiggle-physics.zip: __init__.py blender_manifest.toml
	python generate_info.py -- blender_manifest.toml __init__.py
	blender --command extension build
