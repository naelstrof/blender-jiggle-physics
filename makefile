all: __jiggle_info.py jiggle-physics.zip

jiggle-physics.zip: __init__.py blender_manifest.toml __jiggle_info.py
	blender --command extension build

__jiggle_info.py: blender_manifest.toml
	python generate_info.py -- blender_manifest.toml __jiggle_info.py
