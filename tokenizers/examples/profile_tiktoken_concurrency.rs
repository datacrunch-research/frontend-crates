// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compare shared and isolated encoder scratch under concurrent calls.
use dynamo_tokenizers::{TikTokenTokenizer, traits::Encoder};
use std::sync::Barrier;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let threads: usize = args[2].parse()?;
    let repeats: usize = args[3].parse()?;
    let isolated = args.get(4).is_some_and(|s| s == "isolated");
    let models = (0..if isolated { threads } else { 1 })
        .map(|_| TikTokenTokenizer::from_file_auto(&args[1]))
        .collect::<Result<Vec<_>, _>>()?;
    let text = "The quick brown fox jumps over the lazy dog.\n".repeat(1000);
    let expected = models[0].encode(&text)?.token_ids().to_vec();
    let barrier = Barrier::new(threads);
    let started = Instant::now();
    std::thread::scope(|scope| {
        for i in 0..threads {
            let (models, text, expected, barrier) = (&models, &text, &expected, &barrier);
            scope.spawn(move || {
                let model = &models[if isolated { i } else { 0 }];
                barrier.wait();
                for _ in 0..repeats {
                    assert_eq!(model.encode(text).unwrap().token_ids(), expected);
                }
            });
        }
    });
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"threads":threads,"repeats":repeats,
        "isolated":isolated,"seconds":seconds})
    );
    if let Some(budget) = args.get(5) {
        let budget: f64 = budget.parse()?;
        assert!(
            seconds <= budget,
            "concurrent encoding exceeded the reference budget"
        );
    }
    Ok(())
}
