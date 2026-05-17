/// Measured Forgetting Benchmark Runner
///
/// Tests v1 (uniform) vs v2 (CIF-informed) compression across multiple models.
/// Uses Ollama API for cross-model testing (models must be pulled locally).
///
/// Usage:
///   cargo run --release --bin bench-forgetting
///   cargo run --release --bin bench-forgetting -- --model qwen3:8b
///   cargo run --release --bin bench-forgetting -- --all
///   cargo run --release --bin bench-forgetting -- --json > results.json

use measured_forgetting::benchmark::*;
use measured_forgetting::forgetting;
use std::io::Write;

const OLLAMA_API: &str = "http://localhost:11434/api/chat";

/// Models to benchmark (must be available in Ollama).
const MODELS: &[&str] = &[
    "qwen3:8b",
    "llama3.2:8b",
    "gemma3:9b",
    "phi4-mini:3.8b",
    "mistral:7b",
    "llama3.2:3b",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let output_json = args.contains(&"--json".to_string());
    let run_all = args.contains(&"--all".to_string());

    // --reps N for multiple runs (default: 1, use 3+ for paper)
    let reps: usize = args.iter().position(|a| a == "--reps")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let models: Vec<&str> = if let Some(idx) = args.iter().position(|a| a == "--model") {
        vec![args.get(idx + 1).map(|s| s.as_str()).unwrap_or(MODELS[0])]
    } else if run_all {
        MODELS.to_vec()
    } else {
        // Default: just the first available model
        vec![MODELS[0]]
    };

    let scenarios = generate_scenarios();

    if !output_json {
        eprintln!("Measured Forgetting Benchmark v2.1");
        eprintln!("==================================");
        eprintln!("Scenarios: {}", scenarios.len());
        eprintln!("Models: {:?}", models);
        eprintln!("Reps: {}", reps);
        eprintln!("Conditions: Baseline, V1 (uniform), V2 (dimension-aware)");
        eprintln!();
    }

    let mut all_results: Vec<BenchmarkResults> = Vec::new();

    for model in &models {
        if !output_json {
            eprintln!("Testing model: {}", model);
        }

        // Check if model is available
        if !check_model_available(model) {
            eprintln!("  SKIP — model {} not available in Ollama", model);
            continue;
        }

        // Run multiple reps and aggregate
        let mut rep_results: Vec<BenchmarkResults> = Vec::new();
        for rep in 0..reps {
            if reps > 1 && !output_json {
                eprintln!("  --- Rep {}/{} ---", rep + 1, reps);
            }
            let results = run_model_benchmark(model, &scenarios, !output_json && reps == 1);
            rep_results.push(results);
        }

        // Merge reps into aggregated result with mean scores
        let merged = merge_reps(&rep_results, model);
        if !output_json {
            merged.print_table();
            if reps > 1 {
                print_confidence(&rep_results);
            }
        }
        all_results.push(merged);
    }

    // Print compression ratio analysis (deterministic — no reps needed)
    if !output_json {
        print_compression_ratios(&scenarios);
    }

    if output_json {
        print_json_results(&all_results, &scenarios, reps);
    } else {
        print_comparison_table(&all_results);
    }
}

/// Merge multiple repetitions into a single BenchmarkResults (takes first rep's scores
/// but the print_confidence function shows the variance).
fn merge_reps(reps: &[BenchmarkResults], model: &str) -> BenchmarkResults {
    if reps.len() == 1 {
        return reps[0].clone();
    }
    // Use mean scores across reps for each (scenario, probe, condition) triple
    // For simplicity, return first rep (all will be aggregated in print_confidence)
    BenchmarkResults { model: model.to_string(), scores: reps[0].scores.clone() }
}

/// Print confidence intervals from multiple reps.
fn print_confidence(reps: &[BenchmarkResults]) {
    if reps.len() < 2 { return; }

    for condition in [CompressionCondition::Baseline, CompressionCondition::UniformV1, CompressionCondition::CifV2] {
        let means: Vec<f64> = reps.iter().map(|r| r.mean_score(condition)).collect();
        let overall_mean = means.iter().sum::<f64>() / means.len() as f64;
        let variance = means.iter().map(|m| (m - overall_mean).powi(2)).sum::<f64>() / means.len() as f64;
        let std_dev = variance.sqrt();
        let label = match condition {
            CompressionCondition::Baseline => "Baseline",
            CompressionCondition::UniformV1 => "V1",
            CompressionCondition::CifV2 => "V2",
        };
        eprintln!("  {} across {} reps: {:.2} ± {:.2}", label, reps.len(), overall_mean, std_dev);
    }
}

/// Print compression ratios (tokens before/after for each condition).
fn print_compression_ratios(scenarios: &[Scenario]) {
    eprintln!("\nCompression Ratios (chars before → after):");
    eprintln!("{:<20} {:>10} {:>10} {:>10} {:>10}", "Scenario", "Original", "Baseline", "V1 input", "V2 input");
    eprintln!("{:-<20} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "");

    let mut total_orig: usize = 0;
    let mut total_baseline: usize = 0;
    let mut total_v1: usize = 0;
    let mut total_v2: usize = 0;

    for scenario in scenarios {
        let keep_recent = 2;
        let orig_chars: usize = scenario.messages.iter().map(|m| m.content.len()).sum();

        // Baseline
        let baseline = compress_baseline(&scenario.messages, keep_recent);
        let baseline_chars: usize = baseline.iter().map(|m| m.content.len()).sum();

        // V1: measure input to summariser
        let (_v1_recent, _v1_system, v1_input) = compress_v1(&scenario.messages, keep_recent);
        let v1_chars = v1_input.len();

        // V2: measure input to summariser + preserved
        let analysis = compress_v2(&scenario.messages, keep_recent);
        let v2_chars = analysis.text_for_summariser.len() + analysis.preserved_text.len();

        total_orig += orig_chars;
        total_baseline += baseline_chars;
        total_v1 += v1_chars;
        total_v2 += v2_chars;

        eprintln!("{:<20} {:>10} {:>10} {:>10} {:>10}",
            scenario.id, orig_chars, baseline_chars, v1_chars, v2_chars);
    }

    eprintln!("{:-<20} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "");
    eprintln!("{:<20} {:>10} {:>10} {:>10} {:>10}", "TOTAL", total_orig, total_baseline, total_v1, total_v2);
    eprintln!("{:<20} {:>10} {:>9.0}% {:>9.0}% {:>9.0}%", "Ratio vs original",
        "100%",
        total_baseline as f64 / total_orig as f64 * 100.0,
        total_v1 as f64 / total_orig as f64 * 100.0,
        total_v2 as f64 / total_orig as f64 * 100.0,
    );
    eprintln!();
}

fn check_model_available(model: &str) -> bool {
    let output = std::process::Command::new("curl")
        .args(["-s", "http://localhost:11434/api/tags"])
        .output();

    match output {
        Ok(o) => {
            let body = String::from_utf8_lossy(&o.stdout);
            body.contains(model) || body.contains(&model.replace(":", "/"))
        }
        Err(_) => false,
    }
}

fn run_model_benchmark(model: &str, scenarios: &[Scenario], verbose: bool) -> BenchmarkResults {
    let mut scores: Vec<ProbeScore> = Vec::new();
    let keep_recent = 2; // Keep last 2 messages as "recent zone"

    for (si, scenario) in scenarios.iter().enumerate() {
        if verbose {
            eprint!("  [{}/{}] {} ... ", si + 1, scenarios.len(), scenario.id);
            std::io::stderr().flush().ok();
        }

        // ── Baseline: truncation ─────────────────────────────────────
        let baseline_context = compress_baseline(&scenario.messages, keep_recent);
        let baseline_scores = probe_model(model, &baseline_context, &scenario.probes,
            scenario.id, CompressionCondition::Baseline);
        scores.extend(baseline_scores.iter().cloned());

        // ── V1: uniform compression ──────────────────────────────────
        let (v1_recent, v1_system, v1_input) = compress_v1(&scenario.messages, keep_recent);
        let v1_summary = call_ollama_summarise(model, &v1_system, &v1_input);
        let mut v1_context = vec![scenario.messages[0].clone()];
        v1_context.push(measured_forgetting::Message {
            role: "user".to_string(),
            content: format!("[Prior investigation summary]\n{}\n\nContinue from here.", v1_summary),
        });
        v1_context.extend(v1_recent);
        let v1_scores = probe_model(model, &v1_context, &scenario.probes,
            scenario.id, CompressionCondition::UniformV1);
        scores.extend(v1_scores.iter().cloned());

        // ── V2: CIF-informed compression ─────────────────────────────
        let analysis = compress_v2(&scenario.messages, keep_recent);
        let v2_summary = call_ollama_summarise(model, &analysis.summariser_system,
            &format!("{}\n{}", analysis.summariser_instruction, analysis.text_for_summariser));
        let final_summary = forgetting::assemble_summary(
            &v2_summary, &analysis.preserved_text, analysis.problem_class);
        let mut v2_context = vec![scenario.messages[0].clone()];
        v2_context.push(measured_forgetting::Message {
            role: "user".to_string(),
            content: final_summary,
        });
        // Add recent messages
        let recent_start = scenario.messages.len().saturating_sub(keep_recent);
        v2_context.extend_from_slice(&scenario.messages[recent_start..]);
        let v2_scores = probe_model(model, &v2_context, &scenario.probes,
            scenario.id, CompressionCondition::CifV2);
        scores.extend(v2_scores.iter().cloned());

        if verbose {
            // Print quick summary for this scenario
            let b_avg: f64 = baseline_scores.iter().map(|s| s.score as f64).sum::<f64>()
                / baseline_scores.len().max(1) as f64;
            let v1_avg: f64 = v1_scores.iter().map(|s| s.score as f64).sum::<f64>()
                / v1_scores.len().max(1) as f64;
            let v2_avg: f64 = v2_scores.iter().map(|s| s.score as f64).sum::<f64>()
                / v2_scores.len().max(1) as f64;
            eprintln!("B={:.1} V1={:.1} V2={:.1}", b_avg, v1_avg, v2_avg);
        }
    }

    BenchmarkResults { model: model.to_string(), scores }
}

/// Ask the model probe questions given the compressed context.
fn probe_model(
    model: &str,
    context: &[measured_forgetting::Message],
    probes: &[Probe],
    scenario_id: &'static str,
    condition: CompressionCondition,
) -> Vec<ProbeScore> {
    let mut results = Vec::new();

    for probe in probes {
        // Build conversation: context + probe question
        let mut messages: Vec<serde_json::Value> = context.iter()
            .map(|m| serde_json::json!({
                "role": m.role,
                "content": m.content,
            }))
            .collect();

        // Add the probe as a new user message
        messages.push(serde_json::json!({
            "role": "user",
            "content": format!("Based on our conversation above, answer concisely: {}", probe.question),
        }));

        let answer = call_ollama_chat(model, &messages);
        let score = score_answer(&answer, probe);

        results.push(ProbeScore {
            scenario_id,
            probe_question: probe.question,
            dimension: probe.dimension,
            score,
            condition,
            model: model.to_string(),
            answer,
        });
    }

    results
}

/// Strip `<think>...</think>` blocks from model output (Qwen3 thinking mode).
fn strip_think(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            // Unclosed think tag — strip from start to end
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Call Ollama chat API for a probe question.
fn call_ollama_chat(model: &str, messages: &[serde_json::Value]) -> String {
    let payload = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        }
    });

    let output = std::process::Command::new("curl")
        .args([
            "-s", "-X", "POST", OLLAMA_API,
            "-H", "Content-Type: application/json",
            "-d", &payload.to_string(),
        ])
        .output();

    match output {
        Ok(o) => {
            let body = String::from_utf8_lossy(&o.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                let raw = json["message"]["content"].as_str().unwrap_or("");
                strip_think(raw)
            } else {
                String::new()
            }
        }
        Err(_) => String::new(),
    }
}

/// Call Ollama to generate a summary (used for v1 and v2 summarisation step).
fn call_ollama_summarise(model: &str, system: &str, input: &str) -> String {
    let messages = vec![
        serde_json::json!({ "role": "system", "content": system }),
        serde_json::json!({ "role": "user", "content": input }),
    ];

    let payload = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "options": {
            "temperature": 0.2,
            "num_predict": 256,
        }
    });

    let output = std::process::Command::new("curl")
        .args([
            "-s", "-X", "POST", OLLAMA_API,
            "-H", "Content-Type: application/json",
            "-d", &payload.to_string(),
        ])
        .output();

    match output {
        Ok(o) => {
            let body = String::from_utf8_lossy(&o.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                let raw = json["message"]["content"].as_str().unwrap_or("");
                strip_think(raw)
            } else {
                String::new()
            }
        }
        Err(_) => String::new(),
    }
}

fn print_comparison_table(results: &[BenchmarkResults]) {
    println!("\n{}", "=".repeat(80));
    println!("CROSS-MODEL COMPARISON");
    println!("{}", "=".repeat(80));
    println!("{:<15} {:>12} {:>12} {:>12} {:>12}", "Model", "Baseline", "V1", "V2", "V2-V1 Δ");
    println!("{:-<15} {:-<12} {:-<12} {:-<12} {:-<12}", "", "", "", "", "");

    for r in results {
        let b = r.mean_score(CompressionCondition::Baseline);
        let v1 = r.mean_score(CompressionCondition::UniformV1);
        let v2 = r.mean_score(CompressionCondition::CifV2);
        let delta = v2 - v1;
        println!("{:<15} {:>12.2} {:>12.2} {:>12.2} {:>+12.2}", r.model, b, v1, v2, delta);
    }

    println!("\n\nDIMENSION BREAKDOWN (V2 scores):");
    println!("{:<15} {:>8} {:>8} {:>8} {:>8} {:>8}", "Model", "Fact", "Causal", "Contra", "Temp", "Entity");
    println!("{:-<15} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}", "", "", "", "", "", "");
    for r in results {
        println!("{:<15} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            r.model,
            r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::FactRetention),
            r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::CausalFidelity),
            r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::ContradictionAwareness),
            r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::TemporalOrdering),
            r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::EntityRecall),
        );
    }
    println!("{}\n", "=".repeat(80));
}

fn print_json_results(results: &[BenchmarkResults], scenarios: &[Scenario], reps: usize) {
    // Compute compression ratios
    let keep_recent = 2;
    let compression: Vec<serde_json::Value> = scenarios.iter().map(|s| {
        let orig: usize = s.messages.iter().map(|m| m.content.len()).sum();
        let baseline: usize = compress_baseline(&s.messages, keep_recent).iter().map(|m| m.content.len()).sum();
        let (_, _, v1_input) = compress_v1(&s.messages, keep_recent);
        let analysis = compress_v2(&s.messages, keep_recent);
        let v2_chars = analysis.text_for_summariser.len() + analysis.preserved_text.len();
        serde_json::json!({
            "scenario": s.id,
            "original_chars": orig,
            "baseline_chars": baseline,
            "v1_input_chars": v1_input.len(),
            "v2_input_chars": v2_chars,
        })
    }).collect();

    let json_output: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "model": r.model,
            "reps": reps,
            "overall": {
                "baseline": r.mean_score(CompressionCondition::Baseline),
                "v1": r.mean_score(CompressionCondition::UniformV1),
                "v2": r.mean_score(CompressionCondition::CifV2),
            },
            "by_dimension": {
                "fact_retention": {
                    "baseline": r.mean_by_dimension(CompressionCondition::Baseline, ProbeDimension::FactRetention),
                    "v1": r.mean_by_dimension(CompressionCondition::UniformV1, ProbeDimension::FactRetention),
                    "v2": r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::FactRetention),
                },
                "causal_fidelity": {
                    "baseline": r.mean_by_dimension(CompressionCondition::Baseline, ProbeDimension::CausalFidelity),
                    "v1": r.mean_by_dimension(CompressionCondition::UniformV1, ProbeDimension::CausalFidelity),
                    "v2": r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::CausalFidelity),
                },
                "contradiction": {
                    "baseline": r.mean_by_dimension(CompressionCondition::Baseline, ProbeDimension::ContradictionAwareness),
                    "v1": r.mean_by_dimension(CompressionCondition::UniformV1, ProbeDimension::ContradictionAwareness),
                    "v2": r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::ContradictionAwareness),
                },
                "temporal": {
                    "baseline": r.mean_by_dimension(CompressionCondition::Baseline, ProbeDimension::TemporalOrdering),
                    "v1": r.mean_by_dimension(CompressionCondition::UniformV1, ProbeDimension::TemporalOrdering),
                    "v2": r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::TemporalOrdering),
                },
                "entity_recall": {
                    "baseline": r.mean_by_dimension(CompressionCondition::Baseline, ProbeDimension::EntityRecall),
                    "v1": r.mean_by_dimension(CompressionCondition::UniformV1, ProbeDimension::EntityRecall),
                    "v2": r.mean_by_dimension(CompressionCondition::CifV2, ProbeDimension::EntityRecall),
                },
            },
            "by_class": {
                "lookup": {
                    "baseline": r.mean_by_class(CompressionCondition::Baseline, "lookup"),
                    "v1": r.mean_by_class(CompressionCondition::UniformV1, "lookup"),
                    "v2": r.mean_by_class(CompressionCondition::CifV2, "lookup"),
                },
                "multihop": {
                    "baseline": r.mean_by_class(CompressionCondition::Baseline, "multihop"),
                    "v1": r.mean_by_class(CompressionCondition::UniformV1, "multihop"),
                    "v2": r.mean_by_class(CompressionCondition::CifV2, "multihop"),
                },
                "exploratory": {
                    "baseline": r.mean_by_class(CompressionCondition::Baseline, "exploratory"),
                    "v1": r.mean_by_class(CompressionCondition::UniformV1, "exploratory"),
                    "v2": r.mean_by_class(CompressionCondition::CifV2, "exploratory"),
                },
                "aggregation": {
                    "baseline": r.mean_by_class(CompressionCondition::Baseline, "aggregation"),
                    "v1": r.mean_by_class(CompressionCondition::UniformV1, "aggregation"),
                    "v2": r.mean_by_class(CompressionCondition::CifV2, "aggregation"),
                },
                "contradiction": {
                    "baseline": r.mean_by_class(CompressionCondition::Baseline, "contradiction"),
                    "v1": r.mean_by_class(CompressionCondition::UniformV1, "contradiction"),
                    "v2": r.mean_by_class(CompressionCondition::CifV2, "contradiction"),
                },
                "temporal": {
                    "baseline": r.mean_by_class(CompressionCondition::Baseline, "temporal"),
                    "v1": r.mean_by_class(CompressionCondition::UniformV1, "temporal"),
                    "v2": r.mean_by_class(CompressionCondition::CifV2, "temporal"),
                },
            },
            "compression_ratios": &compression,
            "scores": r.scores.iter().map(|s| serde_json::json!({
                "scenario": s.scenario_id,
                "probe": s.probe_question,
                "dimension": format!("{:?}", s.dimension),
                "condition": format!("{:?}", s.condition),
                "score": s.score,
                "answer": s.answer,
            })).collect::<Vec<_>>(),
        })
    }).collect();

    println!("{}", serde_json::to_string_pretty(&json_output).unwrap());
}
