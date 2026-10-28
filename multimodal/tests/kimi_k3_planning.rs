// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::kimi_k3::{KimiK3Framing, KimiK3Processor, KimiK3Spec};
use dynamo_multimodal::processor::{DecodedMedia, MediaMetadata, MmFamilyProcessor};
use dynamo_multimodal::registry::{
    ProcessorSpec, build_processor, processor_from_spec, spec_from_hf_configs, spec_from_model_dir,
};
use dynamo_multimodal::token_layout::apply_layout;

// Trimmed from moonshotai/Kimi-K3 config.json.
const K3_CONFIG: &str = r#"{
    "architectures": ["KimiK3ForConditionalGeneration"],
    "model_type": "kimi_k3",
    "media_placeholder_token_id": 163605,
    "text_config": {"model_type": "kimi_linear"},
    "vision_config": {
        "merge_kernel_size": [2, 2], "merge_type": "sd2_tpool", "patch_size": 14,
        "vt_hidden_size": 1024, "vt_num_hidden_layers": 27
    }
}"#;
// moonshotai/Kimi-K3 preprocessor_config.json.
const K3_PREPROCESSOR_CONFIG: &str = r#"{
    "media_proc_cfg": {
        "in_patch_limit": 65536, "patch_size": 14,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "merge_kernel_size": 2, "fixed_output_tokens": null,
        "patch_limit_on_one_side": 512,
        "in_patch_limit_each_frame": 16384, "in_patch_limit_video": 655360,
        "sample_fps": 8.0, "max_num_frames_each_video": null,
        "temporal_merge_kernel_size": 4, "timestamp_mode": "hh:mm:ss.fff",
        "transparent_bg_config": {"pattern": "chessboard", "chessboard_square_size": 8},
        "transparent_bg_fill_stage": "after_resize",
        "config_type": "media_proc.processors.moonvit.MoonViTMediaProcessorConfig"
    },
    "auto_map": {
        "AutoProcessor": "kimi_k3_processor.KimiK3Processor",
        "AutoImageProcessor": "kimi_k3_vision_processing.KimiK3VisionProcessor"
    }
}"#;

/// The checkpoint's added tokens (tokenizer_config.json), which config.json
/// does not carry.
const MEDIA_BEGIN: i32 = 163601;
const MEDIA_CONTENT: i32 = 163602;
const MEDIA_END: i32 = 163603;
const MEDIA_PAD: i32 = 163605;

fn k3_spec(config: &str, pre: &str) -> dynamo_multimodal::Result<KimiK3Spec> {
    spec_from_hf_configs(config, pre).map(|spec| match spec {
        ProcessorSpec::KimiK3(spec) => spec,
        _ => panic!("expected a Kimi K3 spec"),
    })
}

/// One id per byte: a stand-in for the checkpoint's tokenizer on the size literal.
fn bytes_tokenizer(text: &str) -> Vec<i32> {
    text.bytes().map(i32::from).collect()
}

fn framed() -> KimiK3Processor {
    let mut spec = k3_spec(K3_CONFIG, K3_PREPROCESSOR_CONFIG).unwrap();
    spec.framing = Some(KimiK3Framing {
        media_begin_token_id: MEDIA_BEGIN,
        media_content_token_id: MEDIA_CONTENT,
        media_end_token_id: MEDIA_END,
    });
    KimiK3Processor::new(spec)
        .unwrap()
        .with_size_tokenizer(bytes_tokenizer)
}

#[test]
fn resolves_the_checkpoint_from_its_config_files() {
    let spec = k3_spec(K3_CONFIG, K3_PREPROCESSOR_CONFIG).unwrap();
    assert_eq!(spec.media_pad_token_id, MEDIA_PAD);
    assert!(spec.framing.is_none(), "framing ids live in the tokenizer");
    assert_eq!(
        (
            spec.geometry.patch_size,
            spec.geometry.merge_kernel_size,
            spec.geometry.in_patch_limit,
            spec.geometry.patch_limit_on_one_side,
            spec.geometry.fixed_output_tokens,
        ),
        (14, 2, 65536, 512, None)
    );
    let family = build_processor(ProcessorSpec::KimiK3(spec)).unwrap();
    let image = MediaMetadata::Image {
        width: 640,
        height: 480,
    };
    assert_eq!(family.num_media_tokens(&image).unwrap(), 414);

    // From a model directory too.
    let dir =
        std::env::temp_dir().join(format!("dynamo-multimodal-kimi-k3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), K3_CONFIG).unwrap();
    std::fs::write(dir.join("preprocessor_config.json"), K3_PREPROCESSOR_CONFIG).unwrap();
    let from_dir = spec_from_model_dir(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        matches!(from_dir, ProcessorSpec::KimiK3(spec) if spec.media_pad_token_id == MEDIA_PAD)
    );
}

#[test]
fn drifted_or_unknown_configuration_is_refused() {
    // A knob the planner does not know.
    let mut pre: serde_json::Value = serde_json::from_str(K3_PREPROCESSOR_CONFIG).unwrap();
    pre["media_proc_cfg"]["crop_to_square"] = serde_json::Value::Bool(true);
    assert!(matches!(
        k3_spec(K3_CONFIG, &pre.to_string()),
        Err(MmError::Unsupported { .. })
    ));
    // Another processor type's config.
    let mut pre: serde_json::Value = serde_json::from_str(K3_PREPROCESSOR_CONFIG).unwrap();
    pre["media_proc_cfg"]["config_type"] = "media_proc.processors.other.Config".into();
    assert!(matches!(
        k3_spec(K3_CONFIG, &pre.to_string()),
        Err(MmError::Unsupported { .. })
    ));
    // The vision tower disagrees with the processor.
    let mut config: serde_json::Value = serde_json::from_str(K3_CONFIG).unwrap();
    config["vision_config"]["patch_size"] = 16.into();
    assert!(matches!(
        k3_spec(&config.to_string(), K3_PREPROCESSOR_CONFIG),
        Err(MmError::InvalidInput { .. })
    ));
    // No pad id.
    let mut config: serde_json::Value = serde_json::from_str(K3_CONFIG).unwrap();
    config
        .as_object_mut()
        .unwrap()
        .remove("media_placeholder_token_id");
    assert!(matches!(
        k3_spec(&config.to_string(), K3_PREPROCESSOR_CONFIG),
        Err(MmError::InvalidInput { .. })
    ));
    // The nested object is required.
    assert!(k3_spec(K3_CONFIG, r#"{"patch_size": 14}"#).is_err());
}

#[test]
fn layout_frames_each_pad_with_the_engine_tokens() {
    let processor = framed();
    let media =
        [(640, 480), (13, 15)].map(|(width, height)| MediaMetadata::Image { width, height });
    let counts = media
        .iter()
        .map(|item| processor.num_media_tokens(item).unwrap())
        .collect::<Vec<_>>();
    // 13x15 keeps its size, pads to 28x28: one merged cell.
    assert_eq!(counts, [414, 1]);
    // Two chat-template placeholders around text, one pad each.
    let ids = [7, MEDIA_PAD, 8, MEDIA_PAD, 9];
    let layout = processor.layout_from_metadata(&ids, &media).unwrap();
    let expanded = apply_layout(&ids, &layout, &counts, 1024).unwrap();
    let mut expected = vec![7, MEDIA_BEGIN];
    expected.extend(bytes_tokenizer("image 640x480"));
    expected.push(MEDIA_CONTENT);
    expected.extend(std::iter::repeat_n(MEDIA_PAD, 414));
    expected.extend([MEDIA_END, 8, MEDIA_BEGIN]);
    expected.extend(bytes_tokenizer("image 13x15"));
    expected.push(MEDIA_CONTENT);
    expected.push(MEDIA_PAD);
    expected.extend([MEDIA_END, 9]);
    assert_eq!(expanded.input_ids, expected);
    // Offsets span the framing; feature ranges are the pads alone.
    assert_eq!(expanded.offsets, [(1, 430), (432, 446)]);
    assert_eq!(expanded.feature_ranges[0], vec![16..430]);
    assert_eq!(expanded.feature_ranges[1], vec![445..446]);
    assert_eq!(processor.expanded_len(640, 480).unwrap(), 430);
    assert_eq!(processor.expanded_len(13, 15).unwrap(), 15);
    // A placeholder count that does not match the media list is an error.
    assert!(
        processor
            .layout_from_metadata(&[MEDIA_PAD], &media)
            .is_err()
    );
}

#[test]
fn layouts_need_framing_and_a_size_tokenizer() {
    let image = MediaMetadata::Image {
        width: 640,
        height: 480,
    };
    let bare = build_processor(ProcessorSpec::KimiK3(
        k3_spec(K3_CONFIG, K3_PREPROCESSOR_CONFIG).unwrap(),
    ))
    .unwrap();
    assert_eq!(bare.num_media_tokens(&image).unwrap(), 414);
    assert!(matches!(
        bare.layout_from_metadata(&[MEDIA_PAD], &[image]),
        Err(MmError::Unsupported { .. })
    ));
    let mut spec = k3_spec(K3_CONFIG, K3_PREPROCESSOR_CONFIG).unwrap();
    spec.framing = Some(KimiK3Framing {
        media_begin_token_id: MEDIA_BEGIN,
        media_content_token_id: MEDIA_CONTENT,
        media_end_token_id: MEDIA_END,
    });
    let without_tokenizer = KimiK3Processor::new(spec).unwrap();
    assert!(matches!(
        without_tokenizer.layout_from_metadata(&[MEDIA_PAD], &[image]),
        Err(MmError::Unsupported { .. })
    ));
}

#[test]
fn serialized_spec_keeps_pixel_execution_and_other_modalities_unsupported() {
    let family = processor_from_spec(
        r#"{
            "family": "kimi_k3", "media_pad_token_id": 163605,
            "framing": {"media_begin_token_id": 163601, "media_content_token_id": 163602, "media_end_token_id": 163603},
            "patch_size": 14, "merge_kernel_size": [2, 2],
            "in_patch_limit": 65536, "patch_limit_on_one_side": 512
        }"#,
    )
    .unwrap();
    let image = DecodedMedia::Image {
        rgb: vec![],
        height: 42,
        width: 42,
    };
    assert!(matches!(
        family.process_item(&image),
        Err(MmError::Unsupported { .. })
    ));
    assert!(matches!(
        family.layout(&[MEDIA_PAD], &[]),
        Err(MmError::Unsupported { .. })
    ));
    for other in [
        MediaMetadata::Video {
            width: 448,
            height: 448,
            num_frames: 16,
        },
        MediaMetadata::Audio {
            num_samples: 16000,
            sample_rate: 16000,
        },
    ] {
        assert!(matches!(
            family.num_media_tokens(&other),
            Err(MmError::Unsupported { .. })
        ));
    }
    assert!(processor_from_spec(r#"{"family": "kimi_k3", "media_pad_token_id": 1}"#).is_err());
}
