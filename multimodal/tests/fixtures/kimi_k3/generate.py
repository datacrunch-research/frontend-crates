# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Kimi K3 image geometry goldens from pinned SGLang, stdlib only.

The pinned source is `python/sglang/srt/multimodal/processors/kimi_k25.py` at
the SGLang commit below, whose `navit_resize_config` and
`_grid_thw_from_resize_config` are shared by the Kimi K2.5 and K3 processors
(kimi_k3.py imports them). Only those two pure functions are executed; the
rest of the module needs torch.

`--verify-hf` additionally runs the installed Kimi K3 image processor on
synthetic images and asserts its `image_grid_thw` matches the planned grid; it
needs transformers, torch and Pillow plus the checkpoint's processor code and
is not required to regenerate the fixture.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
from urllib.request import urlopen

SOURCE = (
    "https://raw.githubusercontent.com/sgl-project/sglang/"
    "faa56e2b5c5c5dee60eb5dd7b6251944a21a6104/"
    "python/sglang/srt/multimodal/processors/kimi_k25.py"
)
FUNCTIONS = {"navit_resize_config", "_grid_thw_from_resize_config"}

# (patch_size, merge_kernel_size, in_patch_limit, patch_limit_on_one_side,
#  fixed_output_tokens). The first row is moonshotai/Kimi-K3's media_proc_cfg;
# the rest are mathematical test inputs, not model defaults.
SPECS = [
    (14, 2, 65536, 512, None),
    (14, 2, 65536, 512, 64),
    (14, 2, 1024, 512, None),
    (14, 2, 65536, 64, None),
    (16, 2, 65536, 512, None),
    (14, 3, 65536, 512, None),
    (14, 2, 16, 2, None),
    (28, 1, 4096, 128, None),
]
DIMENSIONS = [
    (1, 1), (13, 15), (14, 14), (27, 28), (28, 28), (29, 29), (56, 56),
    (100, 76), (448, 448), (640, 480), (1024, 768), (768, 1024),
    (1920, 1080), (1080, 1920), (3840, 2160), (2160, 3840), (4000, 3000),
    (4000, 4000), (8000, 8000), (1, 10000), (10000, 1), (20000, 100),
    (1000, 250), (1001, 250), (1000, 251), (2800, 2800), (3360, 3360),
    (7168, 7168), (7169, 7169), (10000, 10000), (12345, 6789), (65535, 65535),
    # The widest u32 sides; every operand stays far below 2**53.
    (2**32 - 1, 1), (1, 2**32 - 1), (2**32 - 1, 2**32 - 1),
    # Zero sides: Python divides by them.
    (0, 1), (1, 0),
]


def load_functions(source):
    # Execute only the pure geometry functions, avoiding torch imports.
    tree = ast.parse(source)
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    assert {node.name for node in tree.body} == FUNCTIONS
    # The signatures use `int | None`; keep them as strings on older Pythons.
    tree.body.insert(0, ast.parse("from __future__ import annotations").body[0])
    namespace = {"math": math}
    exec(compile(tree, SOURCE, "exec"), namespace)
    return namespace["navit_resize_config"], namespace["_grid_thw_from_resize_config"]


def spec_dict(row):
    keys = ("patch_size", "merge_kernel_size", "in_patch_limit",
            "patch_limit_on_one_side", "fixed_output_tokens")
    return dict(zip(keys, row))


def plan(navit_resize_config, grid_thw, width, height, spec):
    """`prepare_kimi_k3_encoder_inputs`: the resize config, then the grid."""
    config = navit_resize_config(
        width, height, spec["patch_size"], spec["merge_kernel_size"],
        spec["in_patch_limit"], spec["patch_limit_on_one_side"],
        spec["fixed_output_tokens"],
    )
    _, grid_h, grid_w = grid_thw(config, spec["patch_size"])
    return dict(
        resized_width=config["new_width"], resized_height=config["new_height"],
        pad_width=config["pad_width"], pad_height=config["pad_height"],
        grid_h=grid_h, grid_w=grid_w, num_tokens=config["num_tokens"],
    )


def generate(source):
    navit_resize_config, grid_thw = load_functions(source)
    rng = random.Random(3)
    dimensions = DIMENSIONS + [
        (rng.randrange(1, 10000), rng.randrange(1, 10000)) for _ in range(24)
    ]
    groups = []
    for row in SPECS:
        spec = spec_dict(row)
        cases = []
        for width, height in dimensions:
            try:
                expected = plan(navit_resize_config, grid_thw, width, height, spec)
            except (ValueError, ZeroDivisionError) as error:
                expected = {"error": f"{type(error).__name__}: {error}"}
            cases.append([width, height, expected])
        groups.append(
            '    {"spec": ' + json.dumps(spec) + ', "cases": [\n'
            + ',\n'.join('      ' + json.dumps(case) for case in cases)
            + '\n    ]}'
        )
    return (
        '{\n  "source": ' + json.dumps(SOURCE) + ',\n  "source_sha256": '
        + json.dumps(hashlib.sha256(source).hexdigest()) + ',\n  "groups": [\n'
        + ',\n'.join(groups)
        + '\n  ]\n}\n'
    )


def verify_hf(source, max_side=2048):
    """Cross-check the plan against the installed processor's image_grid_thw."""
    from PIL import Image
    from transformers import AutoImageProcessor

    navit_resize_config, grid_thw = load_functions(source)
    processor = AutoImageProcessor.from_pretrained("moonshotai/Kimi-K3", trust_remote_code=True)
    spec = dict(processor.media_proc_cfg)
    spec = {k: spec[k] for k in ("patch_size", "merge_kernel_size", "in_patch_limit",
                                 "patch_limit_on_one_side", "fixed_output_tokens")}
    checked = 0
    for width, height in DIMENSIONS:
        if not (0 < width <= max_side and 0 < height <= max_side):
            continue
        expected = plan(navit_resize_config, grid_thw, width, height, spec)
        out = processor(images=[Image.new("RGB", (width, height), (90, 60, 30))], return_tensors="pt")
        grid = out["image_grid_thw"][0].tolist()
        assert grid == [1, expected["grid_h"], expected["grid_w"]], (width, height, grid, expected)
        checked += 1
    print(f"verified {checked} images against the installed processor")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hf", action="store_true")
    args = parser.parse_args()
    source = urlopen(SOURCE).read()
    out = Path(__file__).with_name("geometry.json")
    out.write_text(generate(source))
    print(f"wrote {out}")
    if args.verify_hf:
        verify_hf(source)


if __name__ == "__main__":
    main()
