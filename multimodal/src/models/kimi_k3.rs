// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Geometry math adapted from SGLang's kimi_k25.py and kimi_k3.py (Apache-2.0).
// Pinned source: sgl-project/sglang@faa56e2b5c5c5dee60eb5dd7b6251944a21a6104.

//! Pixel-free Kimi K3 (`kimi_k3`) image geometry, shared by routers and
//! engines.
//!
//! The registered processor supports metadata-only token counting and prompt
//! layout. Kimi K3's chat template renders exactly one `<|media_pad|>` per
//! image (`image_prompts`); the engine then replaces that one token with the
//! checkpoint's media framing, `<|media_begin|>image {w}x{h}<|media_content|>`,
//! the pad repeated once per NaViT output token, and `<|media_end|>`
//! (`_expand_k3_image_prompt_token_ids`). The token count comes from
//! [`navit_resize_config`]: the image is scaled down to fit `in_patch_limit`
//! patches and `patch_limit_on_one_side` patches per side, then padded to the
//! merge kernel, and each merged cell is one token. The engine's
//! `image_grid_thw` is `[1, h, w]` in patches of that padded canvas.
//!
//! The framing carries the original dimensions as text, so a layout needs a
//! tokenizer for `image {w}x{h}` ([`KimiK3Processor::with_size_tokenizer`]) and
//! the three framing ids ([`KimiK3Framing`], added tokens of the checkpoint's
//! tokenizer, not `config.json`). Without them [`MmFamilyProcessor::num_media_tokens`]
//! still answers (the pad count alone) and
//! [`MmFamilyProcessor::layout_from_metadata`] is unsupported.
//!
//! It does not decode, resize pixels, or produce vision embeddings, and it
//! does not cover video or audio.

use std::sync::Arc;

use crate::processor::{
    DecodedMedia, ExpansionPart, MediaMetadata, MmFamilyProcessor, ProcessedItem, Segment,
    TokenLayout,
};
use crate::{MmError, Result};

/// `media_proc_cfg` knobs that decide the canvas, with the checkpoint's
/// field names (`preprocessor_config.json`, `MoonViTMediaProcessorConfig`).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct KimiK3GeometrySpec {
    pub patch_size: u32,
    /// One number, or the `[h, w]` pair `config.json`'s `vision_config`
    /// spells when both sides are equal.
    #[serde(deserialize_with = "merge_kernel_size")]
    pub merge_kernel_size: u32,
    /// Patch budget of the whole image before the merge.
    pub in_patch_limit: u32,
    /// Patch budget of either side.
    pub patch_limit_on_one_side: u32,
    /// When set, every image occupies this many tokens whatever its size.
    #[serde(default)]
    pub fixed_output_tokens: Option<u32>,
}

fn merge_kernel_size<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<u32, D::Error> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Kernel {
        One(u32),
        Pair([u32; 2]),
    }
    match <Kernel as serde::Deserialize>::deserialize(deserializer)? {
        Kernel::One(size) => Ok(size),
        Kernel::Pair([h, w]) if h == w => Ok(h),
        Kernel::Pair([h, w]) => Err(serde::de::Error::custom(format!(
            "kimi_k3: merge_kernel_size [{h}, {w}] is not square; the planner needs one size"
        ))),
    }
}

/// The three framing tokens the engine inserts around each image, added tokens
/// of the checkpoint's tokenizer.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct KimiK3Framing {
    /// `<|media_begin|>`
    pub media_begin_token_id: i32,
    /// `<|media_content|>`
    pub media_content_token_id: i32,
    /// `<|media_end|>`
    pub media_end_token_id: i32,
}

/// Planning parameters resolved from the checkpoint, without serving defaults.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct KimiK3Spec {
    /// `<|media_pad|>`: `config.json`'s `media_placeholder_token_id`, the
    /// chat template's placeholder and the expanded feature id.
    pub media_pad_token_id: i32,
    /// The framing ids, when the consumer read them from the tokenizer. Needed
    /// for layouts, not for counts.
    #[serde(default)]
    pub framing: Option<KimiK3Framing>,
    #[serde(flatten)]
    pub geometry: KimiK3GeometrySpec,
}

/// Tokenizes the size literal `image {w}x{h}` as plain text, with the
/// checkpoint's tokenizer (no special tokens inside it).
pub type SizeTokenizer = Arc<dyn Fn(&str) -> Vec<i32> + Send + Sync>;

/// Kimi K3 metadata-only processor. Pixel preprocessing is explicitly
/// unsupported.
///
/// ```
/// use dynamo_multimodal::{Result, registry, token_layout};
/// use dynamo_multimodal::models::kimi_k3::{KimiK3Processor, KimiK3Spec};
/// use dynamo_multimodal::processor::{MediaMetadata, MmFamilyProcessor};
/// # fn main() -> Result<()> {
/// // moonshotai/Kimi-K3: config.json's media_placeholder_token_id and
/// // preprocessor_config.json's media_proc_cfg.
/// let spec = r#"{
///     "family": "kimi_k3", "media_pad_token_id": 163605,
///     "patch_size": 14, "merge_kernel_size": 2,
///     "in_patch_limit": 65536, "patch_limit_on_one_side": 512
/// }"#;
/// let family = registry::processor_from_spec(spec)?;
/// let media = [MediaMetadata::Image { width: 640, height: 480 }];
/// // 640x480 stays as is, pads to 644x504: 23x18 merged cells.
/// assert_eq!(family.num_media_tokens(&media[0])?, 414);
///
/// // A layout needs the framing ids and a tokenizer for "image 640x480".
/// let mut spec: KimiK3Spec = serde_json::from_str(spec).unwrap();
/// spec.framing = Some(dynamo_multimodal::models::kimi_k3::KimiK3Framing {
///     media_begin_token_id: 163601, media_content_token_id: 163602, media_end_token_id: 163603,
/// });
/// let processor = KimiK3Processor::new(spec)?
///     .with_size_tokenizer(|text| text.bytes().map(i32::from).collect()); // one id per byte, for the example
/// let ids = [1, 163605, 2]; // the chat template's one pad
/// let layout = processor.layout_from_metadata(&ids, &media)?;
/// let expanded = token_layout::apply_layout(&ids, &layout, &[414], 1024)?;
/// // 1 + begin + "image 640x480" (13) + content + 414 pads + end + 2
/// assert_eq!(expanded.input_ids.len(), 432);
/// assert_eq!(expanded.offsets, [(1, 430)]);
/// assert_eq!(expanded.feature_ranges[0][0], 16..430); // the pads alone take embeddings
/// # Ok(())
/// # }
/// ```
pub struct KimiK3Processor {
    spec: KimiK3Spec,
    size_tokenizer: Option<SizeTokenizer>,
}

impl KimiK3Processor {
    pub fn new(spec: KimiK3Spec) -> Result<Self> {
        spec.geometry.validate()?;
        if spec.media_pad_token_id < 0 {
            return Err(MmError::invalid_input(
                "kimi_k3: media_pad_token_id must be nonnegative",
            ));
        }
        if let Some(framing) = &spec.framing
            && [
                framing.media_begin_token_id,
                framing.media_content_token_id,
                framing.media_end_token_id,
            ]
            .iter()
            .any(|id| *id < 0)
        {
            return Err(MmError::invalid_input(
                "kimi_k3: framing token ids must be nonnegative",
            ));
        }
        Ok(Self {
            spec,
            size_tokenizer: None,
        })
    }

    /// Supplies the tokenizer for the size literal, enabling layouts.
    pub fn with_size_tokenizer(
        mut self,
        tokenizer: impl Fn(&str) -> Vec<i32> + Send + Sync + 'static,
    ) -> Self {
        self.size_tokenizer = Some(Arc::new(tokenizer));
        self
    }

    pub fn spec(&self) -> &KimiK3Spec {
        &self.spec
    }

    pub fn plan_image(&self, width: u32, height: u32) -> Result<KimiK3ImagePlan> {
        plan_image_grid(width, height, &self.spec.geometry)
    }

    /// The text between `<|media_begin|>` and `<|media_content|>`: the
    /// original dimensions, as the engine spells them.
    pub fn size_literal(width: u32, height: u32) -> String {
        format!("image {width}x{height}")
    }

    /// Everything the engine inserts for one image: the ids before the pads
    /// and after them. Needs the framing ids and the size tokenizer.
    pub fn framing_ids(&self, width: u32, height: u32) -> Result<(Vec<i32>, Vec<i32>)> {
        let framing = self.spec.framing.as_ref().ok_or_else(|| {
            MmError::unsupported(
                "kimi_k3: the layout needs the framing token ids (media_begin, media_content, \
                 media_end), which come from the tokenizer, not config.json",
            )
        })?;
        let tokenizer = self.size_tokenizer.as_ref().ok_or_else(|| {
            MmError::unsupported(
                "kimi_k3: the layout needs a tokenizer for the size literal; see \
                 KimiK3Processor::with_size_tokenizer",
            )
        })?;
        let mut before = vec![framing.media_begin_token_id];
        before.extend(tokenizer(&Self::size_literal(width, height)));
        before.push(framing.media_content_token_id);
        Ok((before, vec![framing.media_end_token_id]))
    }

    /// Tokens one image occupies in the expanded prompt, framing included.
    pub fn expanded_len(&self, width: u32, height: u32) -> Result<usize> {
        let (before, after) = self.framing_ids(width, height)?;
        let pads = usize::try_from(self.plan_image(width, height)?.num_tokens)
            .map_err(|_| MmError::limit_exceeded("kimi_k3: token count exceeds usize"))?;
        pads.checked_add(before.len() + after.len())
            .ok_or_else(|| MmError::limit_exceeded("kimi_k3: expanded length exceeds usize"))
    }
}

impl MmFamilyProcessor for KimiK3Processor {
    /// The pad tokens of one image: the positions that receive embeddings.
    /// The framing the engine adds around them is text; [`Self::expanded_len`]
    /// counts both.
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        match media {
            MediaMetadata::Image { width, height } => {
                usize::try_from(self.plan_image(*width, *height)?.num_tokens)
                    .map_err(|_| MmError::limit_exceeded("kimi_k3: token count exceeds usize"))
            }
            _ => Err(MmError::unsupported("kimi_k3 supports only images")),
        }
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout> {
        let pad = self.spec.media_pad_token_id;
        let found = input_ids.iter().filter(|&&id| id == pad).count();
        if found != media.len() {
            return Err(MmError::invalid_input(format!(
                "prompt has {found} media placeholder(s) but {} media item(s)",
                media.len()
            )));
        }
        let mut segments = Vec::new();
        let mut text_start = 0;
        let mut item = 0;
        for (pos, &id) in input_ids.iter().enumerate() {
            if id != pad {
                continue;
            }
            let (width, height) = match media[item] {
                MediaMetadata::Image { width, height } => (width, height),
                _ => return Err(MmError::unsupported("kimi_k3 supports only images")),
            };
            let count = self.num_media_tokens(&media[item])?;
            let (before, after) = self.framing_ids(width, height)?;
            if text_start < pos {
                segments.push(Segment::Text(text_start..pos));
            }
            segments.push(Segment::Media {
                item,
                src: pos..pos + 1,
                expansion: vec![
                    ExpansionPart::Literal(before),
                    ExpansionPart::Feature { id: pad, n: count },
                    ExpansionPart::Literal(after),
                ],
            });
            item += 1;
            text_start = pos + 1;
        }
        if text_start < input_ids.len() {
            segments.push(Segment::Text(text_start..input_ids.len()));
        }
        Ok(TokenLayout { segments })
    }

    fn process_item(&self, _media: &DecodedMedia) -> Result<ProcessedItem> {
        Err(MmError::unsupported(
            "kimi_k3 pixel preprocessing is not implemented; use metadata-only planning",
        ))
    }

    fn layout(&self, _input_ids: &[i32], _items: &[ProcessedItem]) -> Result<TokenLayout> {
        Err(MmError::unsupported(
            "kimi_k3 processed-item layout is not implemented; use layout_from_metadata",
        ))
    }
}

/// The resize the engine applies and what it yields: the scaled image, the
/// padding to the merge kernel, the patch grid of the padded canvas
/// (`image_grid_thw` is `[1, grid_h, grid_w]`) and the token count.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct KimiK3ImagePlan {
    pub resized_width: u32,
    pub resized_height: u32,
    pub pad_width: u32,
    pub pad_height: u32,
    pub grid_h: u32,
    pub grid_w: u32,
    /// Merged cells of the padded canvas, or `fixed_output_tokens`: the
    /// `<|media_pad|>` copies in the expanded prompt.
    pub num_tokens: u32,
}

impl KimiK3GeometrySpec {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.patch_size == 0
            || self.merge_kernel_size == 0
            || self.in_patch_limit == 0
            || self.patch_limit_on_one_side == 0
        {
            return Err(MmError::invalid_input(
                "kimi_k3 geometry: patch size, merge kernel and patch limits must be positive",
            ));
        }
        if self.fixed_output_tokens == Some(0) {
            return Err(MmError::invalid_input(
                "kimi_k3 geometry: fixed_output_tokens must be positive when set",
            ));
        }
        self.side_pixels()?;
        self.merge_factor()?;
        Ok(())
    }

    /// `patch_limit_on_one_side * patch_size`: the longest side after the resize.
    fn side_pixels(&self) -> Result<u32> {
        self.patch_limit_on_one_side
            .checked_mul(self.patch_size)
            .ok_or_else(|| MmError::invalid_input("kimi_k3 geometry: side limit overflows u32"))
    }

    /// `merge_kernel_size * patch_size`: the pixel multiple the canvas pads to.
    fn merge_factor(&self) -> Result<u32> {
        self.merge_kernel_size
            .checked_mul(self.patch_size)
            .ok_or_else(|| MmError::invalid_input("kimi_k3 geometry: merge factor overflows u32"))
    }
}

/// Mirror of the pinned `navit_resize_config` and `_grid_thw_from_resize_config`.
///
/// Python's float steps are reproduced exactly: the scale is the smallest of
/// one, the square root of the patch budget over the whole-patch count, and
/// each side limit over the side; the scaled sides are truncated to integers
/// and clamped to the side limit; padding completes the merge kernel. Every
/// operand is below `2**53`, so the f64 arithmetic is the same as Python's.
pub fn navit_resize_config(
    width: u32,
    height: u32,
    spec: &KimiK3GeometrySpec,
) -> Result<KimiK3ImagePlan> {
    spec.validate()?;
    if width == 0 || height == 0 {
        // Python divides the side limit by the dimension.
        return Err(MmError::invalid_input(
            "kimi_k3 geometry: image dimensions must be positive",
        ));
    }
    let patch = f64::from(spec.patch_size);
    // `max(1.0, width // patch_size)`: whole patches per side, at least one.
    let patches_w = f64::from((width / spec.patch_size).max(1));
    let patches_h = f64::from((height / spec.patch_size).max(1));
    let s1 = (f64::from(spec.in_patch_limit) / (patches_w * patches_h)).sqrt();
    let side = f64::from(spec.patch_limit_on_one_side) * patch;
    let s2 = side / f64::from(width);
    let s3 = side / f64::from(height);
    let scale = 1.0_f64.min(s1).min(s2).min(s3);
    let side_px = spec.side_pixels()?;
    // `int(width * scale)` truncates toward zero, as the cast does for a
    // nonnegative value that fits.
    let resized_width = ((f64::from(width) * scale) as u32).max(1).min(side_px);
    let resized_height = ((f64::from(height) * scale) as u32).max(1).min(side_px);
    let factor = spec.merge_factor()?;
    let pad_height = (factor - resized_height % factor) % factor;
    let pad_width = (factor - resized_width % factor) % factor;
    let canvas_h = resized_height + pad_height;
    let canvas_w = resized_width + pad_width;
    let num_tokens = match spec.fixed_output_tokens {
        Some(fixed) => fixed,
        None => u32::try_from(u64::from(canvas_h / factor) * u64::from(canvas_w / factor))
            .map_err(|_| MmError::limit_exceeded("kimi_k3 geometry: token count exceeds u32"))?,
    };
    Ok(KimiK3ImagePlan {
        resized_width,
        resized_height,
        pad_width,
        pad_height,
        grid_h: canvas_h / spec.patch_size,
        grid_w: canvas_w / spec.patch_size,
        num_tokens,
    })
}

/// [`navit_resize_config`] under the name the other families use.
pub fn plan_image_grid(
    width: u32,
    height: u32,
    spec: &KimiK3GeometrySpec,
) -> Result<KimiK3ImagePlan> {
    navit_resize_config(width, height, spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// moonshotai/Kimi-K3 `preprocessor_config.json`, `media_proc_cfg`.
    fn kimi_k3() -> KimiK3GeometrySpec {
        KimiK3GeometrySpec {
            patch_size: 14,
            merge_kernel_size: 2,
            in_patch_limit: 65_536,
            patch_limit_on_one_side: 512,
            fixed_output_tokens: None,
        }
    }

    /// Hand-derived from the pinned Python: a small image keeps its size and
    /// pads to the merge kernel; a 4000x3000 image stays under both budgets;
    /// a very wide image is cut by the side limit; a huge square by the patch
    /// budget.
    #[test]
    fn plans_the_navit_branches() {
        let plan = navit_resize_config(640, 480, &kimi_k3()).unwrap();
        assert_eq!((plan.resized_width, plan.resized_height), (640, 480));
        assert_eq!((plan.pad_width, plan.pad_height), (4, 24));
        assert_eq!((plan.grid_h, plan.grid_w, plan.num_tokens), (36, 46, 414));

        let plan = navit_resize_config(4000, 3000, &kimi_k3()).unwrap();
        assert_eq!((plan.resized_width, plan.resized_height), (4000, 3000));
        assert_eq!(
            (plan.grid_h, plan.grid_w, plan.num_tokens),
            (216, 286, 15444)
        );

        // s2 = 7168 / 20000: the width is cut to the side limit.
        let plan = navit_resize_config(20000, 100, &kimi_k3()).unwrap();
        assert_eq!((plan.resized_width, plan.resized_height), (7168, 35));
        assert_eq!((plan.grid_h, plan.grid_w, plan.num_tokens), (4, 512, 512));

        // 714x714 whole patches exceed the budget: s1 = sqrt(65536 / 509796).
        let plan = navit_resize_config(10000, 10000, &kimi_k3()).unwrap();
        assert_eq!((plan.resized_width, plan.resized_height), (3585, 3585));
        assert_eq!(
            (plan.grid_h, plan.grid_w, plan.num_tokens),
            (258, 258, 16641)
        );

        let fixed = KimiK3GeometrySpec {
            fixed_output_tokens: Some(64),
            ..kimi_k3()
        };
        let plan = navit_resize_config(640, 480, &fixed).unwrap();
        assert_eq!(plan.num_tokens, 64);
        assert_eq!(
            (plan.grid_h, plan.grid_w),
            (36, 46),
            "the grid is unchanged"
        );
    }

    #[test]
    fn merge_kernel_size_reads_a_number_or_a_square_pair() {
        let one: KimiK3GeometrySpec = serde_json::from_str(
            r#"{"patch_size": 14, "merge_kernel_size": 2, "in_patch_limit": 65536, "patch_limit_on_one_side": 512}"#,
        )
        .unwrap();
        let pair: KimiK3GeometrySpec = serde_json::from_str(
            r#"{"patch_size": 14, "merge_kernel_size": [2, 2], "in_patch_limit": 65536, "patch_limit_on_one_side": 512}"#,
        )
        .unwrap();
        assert_eq!(one, pair);
        assert!(
            serde_json::from_str::<KimiK3GeometrySpec>(
                r#"{"patch_size": 14, "merge_kernel_size": [2, 3], "in_patch_limit": 65536, "patch_limit_on_one_side": 512}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_invalid_inputs_and_configuration() {
        for (w, h) in [(0, 1), (1, 0), (0, 0)] {
            assert!(matches!(
                navit_resize_config(w, h, &kimi_k3()),
                Err(MmError::InvalidInput { .. })
            ));
        }
        let reject = |spec: KimiK3GeometrySpec| {
            assert!(matches!(
                navit_resize_config(28, 28, &spec),
                Err(MmError::InvalidInput { .. })
            ));
        };
        for field in 0..5 {
            let mut spec = kimi_k3();
            match field {
                0 => spec.patch_size = 0,
                1 => spec.merge_kernel_size = 0,
                2 => spec.in_patch_limit = 0,
                3 => spec.patch_limit_on_one_side = 0,
                _ => spec.fixed_output_tokens = Some(0),
            }
            reject(spec);
        }
        reject(KimiK3GeometrySpec {
            patch_size: u32::MAX,
            ..kimi_k3()
        });
        // The largest dimensions still plan: the patch budget scales a square
        // down to sqrt(65536) = 256 whole patches a side, 3584 pixels.
        let plan = navit_resize_config(u32::MAX, u32::MAX, &kimi_k3()).unwrap();
        assert_eq!((plan.resized_width, plan.resized_height), (3584, 3584));
        assert_eq!(plan.num_tokens, 128 * 128);
    }

    fn processor(framing: bool) -> KimiK3Processor {
        KimiK3Processor::new(KimiK3Spec {
            media_pad_token_id: 163605,
            framing: framing.then_some(KimiK3Framing {
                media_begin_token_id: 163601,
                media_content_token_id: 163602,
                media_end_token_id: 163603,
            }),
            geometry: kimi_k3(),
        })
        .unwrap()
    }

    #[test]
    fn counts_without_a_tokenizer_but_lays_out_only_with_one() {
        let image = MediaMetadata::Image {
            width: 640,
            height: 480,
        };
        let bare = processor(false);
        assert_eq!(bare.num_media_tokens(&image).unwrap(), 414);
        assert!(matches!(
            bare.layout_from_metadata(&[1, 163605, 2], &[image]),
            Err(MmError::Unsupported { .. })
        ));
        let framed = processor(true);
        assert!(matches!(
            framed.layout_from_metadata(&[1, 163605, 2], &[image]),
            Err(MmError::Unsupported { .. })
        ));
        let full =
            processor(true).with_size_tokenizer(|text| text.bytes().map(i32::from).collect());
        assert_eq!(full.expanded_len(640, 480).unwrap(), 1 + 13 + 1 + 414 + 1);
        let layout = full
            .layout_from_metadata(&[1, 163605, 2], &[image])
            .unwrap();
        let expanded =
            crate::token_layout::apply_layout(&[1, 163605, 2], &layout, &[414], 1024).unwrap();
        let mut expected = vec![1, 163601];
        expected.extend("image 640x480".bytes().map(i32::from));
        expected.push(163602);
        expected.extend(std::iter::repeat_n(163605, 414));
        expected.extend([163603, 2]);
        assert_eq!(expanded.input_ids, expected);
        assert_eq!(expanded.offsets, [(1, 430)]);
        assert_eq!(expanded.feature_ranges[0], vec![16..430]);
        // Placeholder and item counts must agree; other modalities are refused.
        assert!(full.layout_from_metadata(&[1, 2], &[image]).is_err());
        let video = MediaMetadata::Video {
            width: 640,
            height: 480,
            num_frames: 8,
        };
        assert!(matches!(
            full.num_media_tokens(&video),
            Err(MmError::Unsupported { .. })
        ));
        assert!(matches!(
            full.layout_from_metadata(&[163605], &[video]),
            Err(MmError::Unsupported { .. })
        ));
    }

    #[test]
    fn negative_ids_are_refused() {
        assert!(
            KimiK3Processor::new(KimiK3Spec {
                media_pad_token_id: -1,
                framing: None,
                geometry: kimi_k3(),
            })
            .is_err()
        );
        assert!(
            KimiK3Processor::new(KimiK3Spec {
                media_pad_token_id: 163605,
                framing: Some(KimiK3Framing {
                    media_begin_token_id: -1,
                    media_content_token_id: 163602,
                    media_end_token_id: 163603,
                }),
                geometry: kimi_k3(),
            })
            .is_err()
        );
    }
}
