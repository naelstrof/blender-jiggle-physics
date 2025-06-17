#!/usr/bin/env python3

"""
Reads a blender_manifest.toml file and replaces the bl_info dictionary in the target python file
"""

import sys, toml, ast, re

def toml_to_bl_info_dict(toml_data):
    bl_info = {}
    bl_info["name"] = toml_data.get("name", "Unknown Add-on")
    bl_info["description"] = toml_data.get("tagline", "No description provided")
    bl_info["author"] = toml_data.get("maintainer", "Unknown Author")
    bl_info["blender"] = tuple(map(int, toml_data.get("blender_version_min", "4.2.0").split(".")))
    bl_info["version"] = tuple(map(int, toml_data.get("version", "0.0.0").split(".")))
    bl_info["wiki_url"] = toml_data.get("website", "")
    if (toml_data.get("tags") is not None):
        bl_info["category"] = toml_data.get("tags")[0]
    return bl_info

def bl_info_dict_to_str(bl_info):
    items = []
    for key, value in bl_info.items():
        items.append(f"    {repr(key)}: {repr(value)},")
    return "bl_info = {\n" + "\n".join(items) + "\n}"

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate bl_info.py from blender_manifest.toml")
    parser.add_argument("input", nargs="?", default="blender_manifest.toml", help="Input TOML file")
    parser.add_argument("output", nargs="?", default="__init__.py", help="Target Python file to edit")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            toml_data = toml.load(f)
    except Exception as e:
        print(f"Error reading TOML: {e}", file=sys.stderr)
        sys.exit(1)

    bl_info_str = bl_info_dict_to_str(toml_to_bl_info_dict(toml_data))

    try:
        with open(args.output, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = re.compile(r'bl_info\s*=\s*\{.*?\}', re.DOTALL)
        new_content = pattern.sub(bl_info_str, content)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
