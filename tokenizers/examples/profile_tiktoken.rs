// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSONL profiler for a local Kimi tiktoken model. Input: {"text":"...", "budget_seconds":1.0}.
//! Reports complete IDs and original regex piece sizes; optional budgets fail the process.
use std::io::{BufRead, Write};
use std::time::Instant;

use dynamo_tokenizers::{EncodeSegment, TikTokenTokenizer, traits::Encoder};
use fancy_regex::Regex;
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("local tiktoken.model path");
    let tokenizer = TikTokenTokenizer::from_file_auto(&path)?;
    let regex = Regex::new(fastokens::tiktoken::KIMI_PATTERN)?;
    let mut output = std::io::stdout().lock();
    let mut exceeded = false;
    for line in std::io::stdin().lock().lines() {
        let job: Value = serde_json::from_str(&line?)?;
        let text = job["text"].as_str().expect("text string");
        let started = Instant::now();
        let mut segments = Vec::new();
        if job["kimi_chunked"].as_bool().unwrap_or(false) {
            let (mut start, mut run, mut was_space) = (0, 0, false);
            for (count, (offset, ch)) in text.char_indices().enumerate() {
                let space = ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
                run = if space == was_space { run + 1 } else { 1 };
                if (count > 0 && count % 400_000 == 0) || run > 25_000 {
                    segments.push(EncodeSegment::new(&text[start..offset], true));
                    (start, run) = (offset, 1);
                }
                was_space = space;
            }
            segments.push(EncodeSegment::new(&text[start..], true));
        } else {
            segments.push(EncodeSegment::new(text, true));
        }
        let encoding = tokenizer.encode_segments(&segments)?;
        let seconds = started.elapsed().as_secs_f64();
        let pieces = segments
            .iter()
            .flat_map(|segment| regex.find_iter(segment.text))
            .collect::<Result<Vec<_>, _>>()?;
        let budget = job["budget_seconds"].as_f64();
        exceeded |= budget.is_some_and(|limit| seconds > limit);
        writeln!(
            output,
            "{}",
            json!({"id":job["id"], "seconds":seconds,
            "characters":text.chars().count(), "bytes":text.len(), "pieces":pieces.len(),
            "segments":segments.len(),
            "max_piece_bytes":pieces.iter().map(|m|m.end() - m.start()).max().unwrap_or(0),
            "ids":encoding.token_ids(), "budget_met":budget.map(|limit|seconds<=limit)})
        )?;
        output.flush()?;
    }
    assert!(
        !exceeded,
        "tokenization exceeded the supplied reference budget"
    );
    Ok(())
}
