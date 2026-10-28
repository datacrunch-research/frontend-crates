// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::kimi_k3::{
    KimiK3GeometrySpec, KimiK3ImagePlan, navit_resize_config,
};

/// moonshotai/Kimi-K3 `preprocessor_config.json`, `media_proc_cfg`.
fn spec() -> KimiK3GeometrySpec {
    KimiK3GeometrySpec {
        patch_size: 14,
        merge_kernel_size: 2,
        in_patch_limit: 65_536,
        patch_limit_on_one_side: 512,
        fixed_output_tokens: None,
    }
}

#[test]
fn matches_pinned_sglang_geometry() {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Expected {
        Plan(KimiK3ImagePlan),
        Error { error: String },
    }
    #[derive(serde::Deserialize)]
    struct Group {
        spec: KimiK3GeometrySpec,
        // Original width/height and the expected plan.
        cases: Vec<(u32, u32, Expected)>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        groups: Vec<Group>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/kimi_k3/geometry.json")).unwrap();
    let mut checked = 0;
    for group in fixture.groups {
        for (width, height, expected) in group.cases {
            let actual = navit_resize_config(width, height, &group.spec);
            match expected {
                Expected::Plan(plan) => {
                    assert_eq!(actual.unwrap(), plan, "{width}x{height}, {:?}", group.spec);
                }
                Expected::Error { error } => assert!(
                    matches!(actual, Err(MmError::InvalidInput { .. })),
                    "{width}x{height}, {:?}: Python raised {error:?}",
                    group.spec
                ),
            }
            checked += 1;
        }
    }
    assert!(checked >= 400, "fixture shrank to {checked} cases");
}

/// The released configuration on the sizes the planner's own tests derive by
/// hand, so a fixture regeneration that drifted would show up here too.
#[test]
fn released_configuration_landmarks() {
    let plan = |w, h| navit_resize_config(w, h, &spec()).unwrap();
    assert_eq!(plan(640, 480).num_tokens, 414);
    assert_eq!((plan(640, 480).grid_h, plan(640, 480).grid_w), (36, 46));
    assert_eq!(plan(4000, 3000).num_tokens, 15444);
    assert_eq!(plan(20000, 100).num_tokens, 512);
    assert_eq!(plan(10000, 10000).num_tokens, 16641);
    // The patch budget scales a huge square to 256 whole patches a side.
    assert_eq!(plan(u32::MAX, u32::MAX).num_tokens, 16384);
    assert_eq!(plan(1, 1).num_tokens, 1);
}
