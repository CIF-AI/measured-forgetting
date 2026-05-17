/// Benchmark harness for Measured Forgetting v1 vs v2 comparison.
///
/// Generates fixed synthetic conversations with known embedded facts,
/// runs three compression strategies (baseline/v1/v2), then probes
/// the compressed context to measure retention quality.
///
/// Usage: `cargo test --lib benchmark_forgetting` for unit validation,
///        `cargo run --bin bench-forgetting` for the full cross-model sweep.

use crate::Message;
use crate::forgetting::{self, ProblemClass};

// ── Scenario Definition ──────────────────────────────────────────────

/// A single benchmark scenario with embedded ground-truth facts.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub id: &'static str,
    pub problem_class: ProblemClass,
    pub messages: Vec<Message>,
    pub probes: Vec<Probe>,
}

/// A probe question with expected answers for automated scoring.
#[derive(Debug, Clone)]
pub struct Probe {
    pub question: &'static str,
    /// Keywords that MUST appear in a correct answer (score 3).
    pub required_keywords: Vec<&'static str>,
    /// Keywords that indicate partial correctness (score 1-2).
    pub partial_keywords: Vec<&'static str>,
    /// The dimension this probe tests.
    pub dimension: ProbeDimension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeDimension {
    FactRetention,
    CausalFidelity,
    ContradictionAwareness,
    TemporalOrdering,
    EntityRecall,
}

/// Result of scoring a single probe answer.
#[derive(Debug, Clone)]
pub struct ProbeScore {
    pub scenario_id: &'static str,
    pub probe_question: &'static str,
    pub dimension: ProbeDimension,
    pub score: u8, // 0-3
    pub condition: CompressionCondition,
    pub model: String,
    pub answer: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCondition {
    /// Drop oldest messages, keep newest (naive truncation).
    Baseline,
    /// Uniform summarisation of all messages (v1 algorithm).
    UniformV1,
    /// CIF-informed compression (v2 algorithm).
    CifV2,
}

// ── Scenario Generator ───────────────────────────────────────────────

fn msg(role: &str, content: &str) -> Message {
    Message { role: role.to_string(), content: content.to_string() }
}

/// Generate all 18 benchmark scenarios (6 classes × 3 complexity levels).
pub fn generate_scenarios() -> Vec<Scenario> {
    vec![
        // ── S1: Lookup (short) ───────────────────────────────────────
        scenario_lookup_short(),
        scenario_lookup_medium(),
        scenario_lookup_long(),
        // ── S2: Multi-hop ────────────────────────────────────────────
        scenario_multihop_short(),
        scenario_multihop_medium(),
        scenario_multihop_long(),
        // ── S3: Exploratory ──────────────────────────────────────────
        scenario_exploratory_short(),
        scenario_exploratory_medium(),
        scenario_exploratory_long(),
        // ── S4: Aggregation ──────────────────────────────────────────
        scenario_aggregation_short(),
        scenario_aggregation_medium(),
        scenario_aggregation_long(),
        // ── S5: Contradiction ────────────────────────────────────────
        scenario_contradiction_short(),
        scenario_contradiction_medium(),
        scenario_contradiction_long(),
        // ── S6: Temporal ─────────────────────────────────────────────
        scenario_temporal_short(),
        scenario_temporal_medium(),
        scenario_temporal_long(),
    ]
}

// ── Lookup scenarios ─────────────────────────────────────────────────

fn scenario_lookup_short() -> Scenario {
    Scenario {
        id: "lookup-short",
        problem_class: ProblemClass::Lookup,
        messages: vec![
            msg("user", "What is the current price of gold?"),
            msg("assistant", "Let me check the current gold price for you."),
            msg("user", "[System ran `check_price` and got: Gold spot price: $2,847.30 per troy ounce as of 2026-05-16 09:00 UTC. 24h change: +$12.40 (+0.44%). Weekly high: $2,861.00, weekly low: $2,819.50.]"),
            msg("assistant", "Gold is currently at $2,847.30 per troy ounce, up 0.44% in the last 24 hours."),
            msg("user", "What about silver?"),
            msg("assistant", "Let me check silver as well."),
            msg("user", "[System ran `check_price` and got: Silver spot price: $33.42 per troy ounce as of 2026-05-16 09:00 UTC. 24h change: -$0.18 (-0.54%). Weekly high: $34.10, weekly low: $32.95.]"),
            msg("assistant", "Silver is at $33.42 per troy ounce, down 0.54% today."),
            msg("user", "And platinum?"),
            msg("assistant", "Checking platinum."),
            msg("user", "[System ran `check_price` and got: Platinum spot price: $1,089.00 per troy ounce as of 2026-05-16 09:01 UTC. 24h change: +$3.20 (+0.29%). Supply deficit continuing.]"),
            msg("assistant", "Platinum is $1,089.00 per troy ounce, up 0.29%."),
        ],
        probes: vec![
            Probe {
                question: "What was the gold price?",
                required_keywords: vec!["2847", "2,847"],
                partial_keywords: vec!["gold", "troy", "ounce"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "What was the weekly low for gold?",
                required_keywords: vec!["2819", "2,819"],
                partial_keywords: vec!["low", "weekly"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_lookup_medium() -> Scenario {
    let mut messages = vec![
        msg("user", "Find the GDP per capita of Kenya"),
        msg("assistant", "Let me look that up."),
        msg("user", "[System ran `web_search` and got: Kenya GDP per capita (2025): $2,197 (nominal), $6,254 (PPP). Growth rate: 5.2%. Population: 56.4 million. Total GDP: $124.0 billion nominal.]"),
        msg("assistant", "Kenya's GDP per capita is $2,197 nominal or $6,254 in PPP terms for 2025."),
    ];
    // Add filler messages to push the key fact back
    for i in 0..8 {
        messages.push(msg("user", &format!("What about indicator {}?", i + 1)));
        messages.push(msg("assistant", &format!("Let me check indicator {}.", i + 1)));
        messages.push(msg("user", &format!("[System ran `query` and got: Indicator {} value is {}. This is a routine data point.]", i + 1, 100 + i * 17)));
        messages.push(msg("assistant", &format!("Indicator {} is {}.", i + 1, 100 + i * 17)));
    }

    Scenario {
        id: "lookup-medium",
        problem_class: ProblemClass::Lookup,
        messages,
        probes: vec![
            Probe {
                question: "What was Kenya's GDP per capita in nominal terms?",
                required_keywords: vec!["2197", "2,197"],
                partial_keywords: vec!["Kenya", "GDP", "capita"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "What was Kenya's total GDP?",
                required_keywords: vec!["124"],
                partial_keywords: vec!["billion", "GDP"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_lookup_long() -> Scenario {
    let mut messages = vec![
        msg("user", "What is the population of Nairobi?"),
        msg("assistant", "Let me find the latest population data for Nairobi."),
        msg("user", "[System ran `query` and got: Nairobi population (2026 est.): 5,325,000. Metropolitan area: 10,200,000. Annual growth: 3.8%. Density: 6,247/km2. Area: 696 km2. Counties in metro: Nairobi, Kiambu, Machakos, Kajiado.]"),
        msg("assistant", "Nairobi's population is approximately 5.325 million, with the metropolitan area reaching 10.2 million."),
    ];
    // 20 filler exchanges
    for i in 0..20 {
        messages.push(msg("user", &format!("Tell me about district {}", i + 1)));
        messages.push(msg("assistant", &format!("[System ran `query` and got: District {} has population {} and area {} km2. Healthcare facilities: {}. Schools: {}.]", i + 1, 50000 + i * 3200, 45 + i * 7, 3 + i % 5, 12 + i * 2)));
        messages.push(msg("assistant", &format!("District {} has a population of {} spread across {} km2.", i + 1, 50000 + i * 3200, 45 + i * 7)));
    }

    Scenario {
        id: "lookup-long",
        problem_class: ProblemClass::Lookup,
        messages,
        probes: vec![
            Probe {
                question: "What was Nairobi's metropolitan area population?",
                required_keywords: vec!["10,200,000", "10.2 million", "10200000"],
                partial_keywords: vec!["metro", "Nairobi", "million"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "What counties are in the Nairobi metropolitan area?",
                required_keywords: vec!["Kiambu", "Machakos", "Kajiado"],
                partial_keywords: vec!["counties", "metro"],
                dimension: ProbeDimension::EntityRecall,
            },
        ],
    }
}

// ── Multi-hop scenarios ──────────────────────────────────────────────

fn scenario_multihop_short() -> Scenario {
    Scenario {
        id: "multihop-short",
        problem_class: ProblemClass::MultiHop,
        messages: vec![
            msg("user", "Find which CHW has the highest submission rate in Busia"),
            msg("assistant", "I'll query the database to find the top CHW by submission rate in Busia county."),
            msg("user", "[System ran `graph_query` and got: Top 5 CHWs by submission rate in Busia: 1. Mary Wanjiku (CHW-4821) - 98.3%, 2. John Odhiambo (CHW-3392) - 96.7%, 3. Grace Achieng (CHW-5510) - 95.1%]"),
            msg("assistant", "Mary Wanjiku (CHW-4821) has the highest rate at 98.3%. Let me check her details."),
            msg("user", "[System ran `graph_query` and got: CHW-4821 details: Name: Mary Wanjiku, Sub-county: Teso North, CHP: Malaba, Active since: 2023-01-15, Households: 47, Last submission: 2026-05-15, Supervisor: David Barasa]"),
            msg("assistant", "Mary Wanjiku is based in Teso North sub-county, Malaba CHP, supervising 47 households."),
            msg("user", "[System ran `graph_query` and got: Submission breakdown for CHW-4821: Monthly avg: 141 submissions. April 2026: 145 (98.3% = 145/147.5 expected). Zero-report days: 2. Device: Samsung A14.]"),
            msg("assistant", "Her 98.3% rate comes from 145 actual vs 147.5 expected submissions in April. Only 2 zero-report days. She's on a Samsung A14."),
        ],
        probes: vec![
            Probe {
                question: "Who is the top CHW and what is their rate?",
                required_keywords: vec!["Mary Wanjiku", "98.3"],
                partial_keywords: vec!["CHW", "Busia", "highest"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "How was the 98.3% submission rate calculated?",
                required_keywords: vec!["145", "147.5"],
                partial_keywords: vec!["expected", "actual", "submissions"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "Who is Mary Wanjiku's supervisor?",
                required_keywords: vec!["David Barasa"],
                partial_keywords: vec!["supervisor"],
                dimension: ProbeDimension::EntityRecall,
            },
        ],
    }
}

fn scenario_multihop_medium() -> Scenario {
    Scenario {
        id: "multihop-medium",
        problem_class: ProblemClass::MultiHop,
        messages: vec![
            msg("user", "Why is the sync rate dropping in Kakamega county?"),
            msg("assistant", "Let me investigate the sync rate trend for Kakamega."),
            msg("user", "[System ran `graph_query` and got: Kakamega sync rates (monthly): Jan=87.2%, Feb=85.1%, Mar=82.4%, Apr=78.9%, May(1-15)=74.3%. Trend: declining 2.8pp/month.]"),
            msg("assistant", "Clear downward trend: 87.2% in January to 74.3% in May. Let me look at what's driving this."),
            msg("user", "[System ran `graph_query` and got: Sync failure breakdown Kakamega Apr 2026: Network timeout=45%, App crash=28%, Storage full=18%, Auth expired=9%. Top sub-county failures: Lugari (42%), Likuyani (31%), Malava (27%).]"),
            msg("assistant", "Network timeouts are the leading cause at 45%. Lugari sub-county is worst affected. Let me check if there's an infrastructure correlation."),
            msg("user", "[System ran `web_search` and got: Safaricom network outage reports Kakamega region: Major fiber cut on A1 highway near Lugari (March 15, 2026). Partial restoration April 2, full restoration pending. Affects 3G/4G towers in Lugari, Likuyani corridor.]"),
            msg("assistant", "Found it. A Safaricom fiber cut on the A1 highway near Lugari on March 15th explains the timing perfectly — the decline started in March and Lugari is the worst-affected sub-county."),
            msg("user", "[System ran `graph_query` and got: Pre-fiber-cut sync rates (Feb 2026): Lugari=91.2%, Likuyani=89.8%, Malava=88.4%. Post-fiber-cut (Apr 2026): Lugari=62.1%, Likuyani=71.3%, Malava=79.8%. Delta: Lugari=-29.1pp, Likuyani=-18.5pp, Malava=-8.6pp.]"),
            msg("assistant", "The causal chain is clear: fiber cut (March 15) → network timeouts (45% of failures) → sync rate decline. Lugari lost 29.1 percentage points. The gradient follows proximity to the A1 highway."),
            // Filler to push it further
            msg("user", "What about Bungoma county?"),
            msg("assistant", "Let me check Bungoma for comparison."),
            msg("user", "[System ran `graph_query` and got: Bungoma sync rates: Jan=88.1%, Feb=87.9%, Mar=87.5%, Apr=87.2%, May=86.8%. Stable, minimal decline (-0.3pp/month). No infrastructure issues flagged.]"),
            msg("assistant", "Bungoma is stable — confirms the Kakamega drop is localised, not systemic."),
            msg("user", "Can we fix it?"),
            msg("assistant", "The fix is infrastructure-dependent — Safaricom needs to complete the fiber restoration. In the meantime, we could recommend offline-first submission batching for Lugari and Likuyani CHWs."),
        ],
        probes: vec![
            Probe {
                question: "What caused the sync rate decline in Kakamega?",
                required_keywords: vec!["fiber cut", "Safaricom"],
                partial_keywords: vec!["network", "Lugari", "March", "A1"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "What was Lugari's sync rate before and after the fiber cut?",
                required_keywords: vec!["91.2", "62.1"],
                partial_keywords: vec!["Lugari", "before", "after"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "What was the top failure type and its percentage?",
                required_keywords: vec!["Network timeout", "45"],
                partial_keywords: vec!["timeout", "failure"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_multihop_long() -> Scenario {
    let mut messages = vec![
        msg("user", "Trace the root cause of the KPI discrepancy in the Busia dashboard"),
        msg("assistant", "Let me trace this through the pipeline layers."),
        msg("user", "[System ran `graph_query` and got: Dashboard shows 1,247 active CHWs in Busia. Source DB shows 1,089. Discrepancy: +158 (14.5% inflation).]"),
        msg("assistant", "14.5% inflation — 158 extra CHWs showing in the dashboard. Let me check the intermediate layers."),
        msg("user", "[System ran `graph_query` and got: Pipeline layers - CouchDB: 1,089 | ClickHouse raw: 1,089 | ClickHouse int_: 1,183 | bi_metrics: 1,247. Gap appears at int_ layer (+94) and amplifies at bi_metrics (+64).]"),
        msg("assistant", "Two gaps: +94 at the integration layer and +64 at bi_metrics. Let me dig into the int_ inflation."),
        msg("user", "[System ran `graph_query` and got: int_ layer analysis: 94 extra records are CHWs who were deactivated but have status='active' in int_chw because the deactivation event arrived AFTER the nightly snapshot. Timestamp: deactivations processed at 03:00, snapshot runs at 02:00.]"),
        msg("assistant", "Race condition: snapshot at 02:00 captures pre-deactivation state, deactivations process at 03:00. That's the +94. Now for the +64 at bi_metrics."),
        msg("user", "[System ran `graph_query` and got: bi_metrics inflation: 64 records are double-counted CHWs who transferred between sub-counties. Transfer creates new record in destination without closing source record in the same month. JOIN produces duplicates.]"),
        msg("assistant", "Two independent root causes: (1) Race condition in snapshot timing (+94), (2) Transfer double-counting in JOIN logic (+64). Total: 158. Both are fixable."),
    ];
    // Add more investigation/filler
    for i in 0..10 {
        messages.push(msg("user", &format!("[System ran `graph_query` and got: Additional validation check {}: metric_{} shows value {} in dashboard vs {} in source. Delta: {}.]", i+1, i+1, 200+i*30, 195+i*28, 5+i*2)));
        messages.push(msg("assistant", &format!("Check {} confirms the pattern — {}% discrepancy consistent with the two root causes.", i+1, (5+i*2) as f64 / (195.0+i as f64*28.0) * 100.0)));
    }

    Scenario {
        id: "multihop-long",
        problem_class: ProblemClass::MultiHop,
        messages,
        probes: vec![
            Probe {
                question: "What were the two root causes of the KPI discrepancy?",
                required_keywords: vec!["race condition", "snapshot", "transfer", "double-count"],
                partial_keywords: vec!["timing", "JOIN", "deactivat"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "At which pipeline layer did the first gap appear?",
                required_keywords: vec!["int_"],
                partial_keywords: vec!["integration", "layer", "94"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "What was the total discrepancy and percentage?",
                required_keywords: vec!["158", "14.5"],
                partial_keywords: vec!["inflation", "extra"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

// ── Exploratory scenarios ────────────────────────────────────────────

fn scenario_exploratory_short() -> Scenario {
    Scenario {
        id: "exploratory-short",
        problem_class: ProblemClass::Exploratory,
        messages: vec![
            msg("user", "Fix the deployment error in the cif-graph service"),
            msg("assistant", "Let me check the deployment status."),
            msg("user", "<tool_error>kubectl: connection refused - cluster unreachable</tool_error>"),
            msg("assistant", "Cluster connection failed. Let me try the SSH tunnel."),
            msg("user", "<tool_error>ssh: Connection timed out to 178.105.40.36</tool_error>"),
            msg("assistant", "SSH also failing. Let me check if the server is responding at all."),
            msg("user", "<tool_error>ping: Request timeout for icmp_seq 0-4</tool_error>"),
            msg("assistant", "Server is completely unreachable. This is likely a Hetzner-level issue. Let me check their status page."),
            msg("user", "[System ran `web_search` and got: Hetzner Status Page (2026-05-16): Scheduled maintenance on FSN1-DC14 rack 42. Started: 08:00 UTC. ETA completion: 10:00 UTC. Affected: CPX32 instances in rack 42.]"),
            msg("assistant", "Found it. Hetzner scheduled maintenance on FSN1-DC14 rack 42, started at 08:00 UTC, ETA 10:00 UTC. Our CPX32 is in that rack."),
            msg("user", "<tool_error>Still unreachable at 09:45 UTC</tool_error>"),
            msg("assistant", "Still down 15 minutes before their ETA. Let me verify our instance details."),
            msg("user", "[System ran `hcloud server describe cif-k3s` and got: ID: 48291037, Type: cpx32, DC: fsn1-dc14, Status: running (but unreachable), IPv4: 178.105.40.36]"),
            msg("assistant", "Confirmed: our server cif-k3s (ID 48291037) is in fsn1-dc14. The fix is to wait for Hetzner maintenance to complete. No action needed from us — server will auto-recover."),
        ],
        probes: vec![
            Probe {
                question: "What was the root cause of the deployment failure?",
                required_keywords: vec!["Hetzner", "maintenance", "FSN1-DC14"],
                partial_keywords: vec!["rack 42", "scheduled", "CPX32"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "What was the resolution?",
                required_keywords: vec!["wait"],
                partial_keywords: vec!["auto-recover", "10:00", "ETA"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_exploratory_medium() -> Scenario {
    Scenario {
        id: "exploratory-medium",
        problem_class: ProblemClass::Exploratory,
        messages: vec![
            msg("user", "The forager is returning empty results — debug it"),
            msg("assistant", "Let me check the forager status."),
            msg("user", "<tool_error>forager_status: TypeError - 'NoneType' object is not iterable at line 47</tool_error>"),
            msg("assistant", "NoneType error at line 47. Let me check what's being iterated."),
            msg("user", "<tool_error>graph_query: Connection pool exhausted (max 10 connections, 10 in use, 0 available)</tool_error>"),
            msg("assistant", "Connection pool exhausted — something is leaking connections. Let me check active queries."),
            msg("user", "<tool_error>SHOW TRANSACTIONS: requires admin privileges</tool_error>"),
            msg("assistant", "Can't show transactions without admin. Let me try the health endpoint."),
            msg("user", "[System ran `web_fetch https://origin.thexi.dev/health` and got: {\"status\": \"degraded\", \"db_pool\": {\"max\": 10, \"active\": 10, \"waiting\": 47}, \"uptime_hours\": 336}]"),
            msg("assistant", "47 queries waiting — the pool has been saturated for a while (336 hours uptime = 14 days without restart). Classic connection leak. Let me find the leaking endpoint."),
            msg("user", "[System ran `graph_query` and got: Recent slow queries: 1. 'MATCH (s:Session)-[:PRECEDED_BY*1..50]->(prev) RETURN s,prev' - running 8 hours (recursive unbounded path query). Started by digest_forager at 01:47 UTC.]"),
            msg("assistant", "Found the leak: the digest forager is running an unbounded recursive path query (PRECEDED_BY*1..50) that's been running for 8 hours. Fix: add a depth limit and kill the hung query."),
            msg("user", "<tool_error>Cannot kill query without admin role</tool_error>"),
            msg("assistant", "Can't kill it remotely. Need to restart the pod to clear the hung connections."),
            msg("user", "[System ran `kubectl delete pod memgraph-0 -n cif-graph` and got: pod \"memgraph-0\" deleted. New pod starting.]"),
            msg("assistant", "Pod restarted. The fix has two parts: (1) immediate — pod restart clears hung queries, (2) permanent — patch the digest forager query to use *1..5 instead of *1..50."),
        ],
        probes: vec![
            Probe {
                question: "What caused the forager to return empty results?",
                required_keywords: vec!["connection pool", "exhausted"],
                partial_keywords: vec!["leak", "saturated", "recursive"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "What specific query was leaking connections?",
                required_keywords: vec!["PRECEDED_BY", "digest_forager"],
                partial_keywords: vec!["recursive", "unbounded", "50"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "What was the permanent fix?",
                required_keywords: vec!["*1..5"],
                partial_keywords: vec!["depth limit", "patch", "instead of"],
                dimension: ProbeDimension::CausalFidelity,
            },
        ],
    }
}

fn scenario_exploratory_long() -> Scenario {
    let mut messages = vec![
        msg("user", "Debug why the training loss is NaN after epoch 3"),
        msg("assistant", "Let me investigate the training run."),
    ];
    // 8 failed attempts
    for i in 0..8 {
        let errors = [
            "gradient norm: inf at layer 12",
            "CUDA out of memory (tried to allocate 2.4GB, 7.8GB used of 8.0GB)",
            "RuntimeError: expected scalar type Float but found Half",
            "loss: nan (first appeared at step 4,217)",
            "checkpoint corrupted — can't resume from epoch 3",
            "learning rate scheduler returned negative lr: -0.00003",
            "batch 1847: all-zero attention weights in head 7 of layer 22",
            "mixed precision overflow: FP16 range exceeded at scale 65504.0",
        ];
        messages.push(msg("user", &format!("<tool_error>{}</tool_error>", errors[i])));
        messages.push(msg("assistant", &format!("Error {}/8: {}. Investigating further.", i+1, errors[i])));
    }
    // The actual fix
    messages.push(msg("user", "[System ran `analyze_training_log` and got: Root cause identified: loss scaler overflow. At step 4,217, gradient scale reached 65504.0 (FP16 max). Next step attempted scale * 2 = overflow → NaN propagated to all parameters. Fix: set initial_scale=2**14 (not 2**16) and max_scale=2**15 in GradScaler config.]"));
    messages.push(msg("assistant", "Root cause: GradScaler overflow. The default initial_scale of 2^16 is too aggressive for this model's gradient magnitudes. Fix: initial_scale=2^14, max_scale=2^15 in the GradScaler config."));

    Scenario {
        id: "exploratory-long",
        problem_class: ProblemClass::Exploratory,
        messages,
        probes: vec![
            Probe {
                question: "What was the root cause of the NaN loss?",
                required_keywords: vec!["GradScaler", "overflow", "65504"],
                partial_keywords: vec!["FP16", "scale", "loss scaler"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "What was the fix?",
                required_keywords: vec!["initial_scale", "2^14"],
                partial_keywords: vec!["max_scale", "2^15", "GradScaler"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "At what step did the NaN first appear?",
                required_keywords: vec!["4217", "4,217"],
                partial_keywords: vec!["step"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

// ── Aggregation scenarios ────────────────────────────────────────────

fn scenario_aggregation_short() -> Scenario {
    Scenario {
        id: "aggregation-short",
        problem_class: ProblemClass::Aggregation,
        messages: vec![
            msg("user", "How many active CHWs are there per county?"),
            msg("assistant", "Let me query the active CHW counts by county."),
            msg("user", "[System ran `graph_query` and got: Active CHWs by county: Busia: 1,089. Kakamega: 1,456. Vihiga: 623. Total across 3 counties: 3,168.]"),
            msg("assistant", "Here's the breakdown: Busia has 1,089, Kakamega has 1,456, and Vihiga has 623 active CHWs. Total: 3,168 across all three counties."),
            msg("user", "Break it down by sub-county for Busia"),
            msg("assistant", "Let me get the sub-county breakdown for Busia."),
            msg("user", "[System ran `graph_query` and got: Busia sub-county breakdown: Teso North: 234, Teso South: 198, Nambale: 167, Matayos: 156, Butula: 178, Budalangi: 89, Samia: 67. Total: 1,089.]"),
            msg("assistant", "Busia's 1,089 CHWs are distributed: Teso North (234), Teso South (198), Butula (178), Nambale (167), Matayos (156), Budalangi (89), Samia (67)."),
        ],
        probes: vec![
            Probe {
                question: "What is the total number of active CHWs across all counties?",
                required_keywords: vec!["3,168", "3168"],
                partial_keywords: vec!["total", "active"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "How many active CHWs does Kakamega have?",
                required_keywords: vec!["1,456", "1456"],
                partial_keywords: vec!["Kakamega"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "Which sub-county in Busia has the most CHWs?",
                required_keywords: vec!["Teso North", "234"],
                partial_keywords: vec!["highest", "most"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_aggregation_medium() -> Scenario {
    let mut messages = vec![
        msg("user", "Summarize all open episodes and their total potential value"),
        msg("assistant", "Let me query the open episodes from the brain."),
        msg("user", "[System ran `graph_query` and got: Open episodes (12 total): EP-F01 GBPUSD long ($450 potential), EP-F02 XAUUSD short ($320), EP-F03 USDJPY long ($180), EP-F04 EURUSD short ($275), EP-P01 Ruto succession ($0 - political), EP-P02 KE elections ($0 - political), EP-S01 NBA playoffs ($150), EP-S02 Champions League ($90), EP-S03 Premier League ($210), EP-V01 Strand product ($5000 est), EP-V02 Tarjumi ($2000 est), EP-V03 B.app ($3000 est). Total financial potential: $11,675.]"),
        msg("assistant", "There are 12 open episodes. Financial breakdown: 4 forex ($1,225), 2 political ($0 direct), 3 sports ($450), 3 ventures ($10,000 est). Grand total potential: $11,675."),
    ];
    // Additional detail queries
    for ep in ["EP-F01", "EP-F02", "EP-V01", "EP-S01"] {
        messages.push(msg("user", &format!("Details on {}?", ep)));
        messages.push(msg("assistant", &format!("Querying {} details.", ep)));
        messages.push(msg("user", &format!("[System ran `check_episode` and got: {} opened 2026-04-{:02}, current Ξ={:.2}, confidence={:.0}%, last updated 2026-05-15]", ep, 1 + ep.len() % 28, 0.1 + ep.len() as f64 * 0.05, 60.0 + ep.len() as f64 * 3.0)));
        messages.push(msg("assistant", &format!("{} is tracking well with moderate confidence.", ep)));
    }

    Scenario {
        id: "aggregation-medium",
        problem_class: ProblemClass::Aggregation,
        messages,
        probes: vec![
            Probe {
                question: "How many open episodes are there in total?",
                required_keywords: vec!["12"],
                partial_keywords: vec!["episodes", "open"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "What is the total financial potential across all episodes?",
                required_keywords: vec!["11,675", "11675"],
                partial_keywords: vec!["total", "potential"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "How many forex episodes are open and what's their combined value?",
                required_keywords: vec!["4", "1,225", "1225"],
                partial_keywords: vec!["forex", "episodes"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_aggregation_long() -> Scenario {
    let mut messages = vec![
        msg("user", "List all sessions from this week with their costs"),
    ];
    // Generate 15 session entries
    for i in 0..15 {
        let cost = 0.12 + i as f64 * 0.08;
        let tokens = 2400 + i * 1800;
        messages.push(msg("assistant", &format!("Fetching session {}.", i+1)));
        messages.push(msg("user", &format!("[System ran `graph_query` and got: Session {} (2026-05-{}): topic='{}', input_tokens={}, output_tokens={}, cost=${:.2}, duration={}min]",
            2080 + i, 12 + i % 5,
            ["forex analysis", "code review", "paper writing", "debugging", "architecture"][i % 5],
            tokens, tokens / 3, cost, 15 + i * 5
        )));
    }
    messages.push(msg("assistant", "All 15 sessions fetched. Total week cost: $13.80. Average cost per session: $0.92."));
    messages.push(msg("user", "[System ran `graph_query` and got: Weekly summary: 15 sessions, total input tokens: 89,400, total output tokens: 29,800, total cost: $13.80, total duration: 7.2 hours. Most expensive: session 2094 ($1.24, paper writing). Cheapest: session 2080 ($0.12, forex analysis).]"));
    messages.push(msg("assistant", "Weekly summary complete. 15 sessions, $13.80 total, 7.2 hours. Most expensive was session 2094 (paper writing, $1.24)."));

    Scenario {
        id: "aggregation-long",
        problem_class: ProblemClass::Aggregation,
        messages,
        probes: vec![
            Probe {
                question: "What was the total cost for the week?",
                required_keywords: vec!["13.80"],
                partial_keywords: vec!["total", "cost", "week"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "How many sessions were there and what was the total duration?",
                required_keywords: vec!["15", "7.2"],
                partial_keywords: vec!["sessions", "hours"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "Which session was most expensive and what was it about?",
                required_keywords: vec!["2094", "paper writing", "1.24"],
                partial_keywords: vec!["most expensive"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

// ── Contradiction scenarios ──────────────────────────────────────────

fn scenario_contradiction_short() -> Scenario {
    Scenario {
        id: "contradiction-short",
        problem_class: ProblemClass::Contradiction,
        messages: vec![
            msg("user", "The dashboard and source database disagree on active CHW count — investigate"),
            msg("assistant", "Let me check both sources."),
            msg("user", "[System ran `graph_query` and got: Source database (CouchDB): 1,089 active CHWs in Busia as of 2026-05-16.]"),
            msg("assistant", "Source says 1,089. Let me check the dashboard."),
            msg("user", "[System ran `graph_query` and got: Dashboard (Power BI): 1,247 active CHWs in Busia as of 2026-05-16. Data refreshed at 06:00 UTC.]"),
            msg("assistant", "Dashboard shows 1,247. That's a discrepancy of +158 (14.5% inflation in the dashboard). The dashboard is overcounting."),
            msg("user", "Which one is correct?"),
            msg("assistant", "The source database (CouchDB) at 1,089 is the ground truth — it's the system of record. The dashboard's 1,247 is inflated due to the pipeline processing (likely stale records or double-counting from transfers)."),
        ],
        probes: vec![
            Probe {
                question: "What were the two conflicting numbers?",
                required_keywords: vec!["1,089", "1,247"],
                partial_keywords: vec!["CouchDB", "Power BI", "dashboard"],
                dimension: ProbeDimension::ContradictionAwareness,
            },
            Probe {
                question: "Which source is the ground truth?",
                required_keywords: vec!["CouchDB", "1,089"],
                partial_keywords: vec!["source", "ground truth", "system of record"],
                dimension: ProbeDimension::CausalFidelity,
            },
        ],
    }
}

fn scenario_contradiction_medium() -> Scenario {
    Scenario {
        id: "contradiction-medium",
        problem_class: ProblemClass::Contradiction,
        messages: vec![
            msg("user", "Two different sources give different Ruto approval ratings — which is right?"),
            msg("assistant", "Let me check both sources."),
            msg("user", "[System ran `web_search` and got: Infotrak poll (April 2026): Ruto approval rating 31%. Sample: 2,000 respondents, margin of error ±2.2%. Methodology: stratified random sampling, phone interviews.]"),
            msg("assistant", "Infotrak says 31% approval. Let me check the other source."),
            msg("user", "[System ran `web_search` and got: TIFA Research poll (April 2026): Ruto approval rating 42%. Sample: 1,500 respondents, margin of error ±2.5%. Methodology: face-to-face interviews in urban and peri-urban areas.]"),
            msg("assistant", "Significant discrepancy: Infotrak 31% vs TIFA 42%. That's an 11 percentage point gap, well outside both margins of error. Let me analyze why."),
            msg("user", "[System ran `web_search` and got: Methodological comparison: Infotrak uses phone interviews (captures rural respondents, lower income bias toward airtime cost). TIFA uses face-to-face in urban/peri-urban only (excludes rural, higher income respondents who benefit more from current policies). Kenya's rural population: 68%. Urban: 32%.]"),
            msg("assistant", "The gap is methodological: Infotrak captures rural respondents (68% of Kenya) via phone, while TIFA only samples urban/peri-urban face-to-face. Since current policies favor urban development, TIFA's sample is biased toward beneficiaries. Neither is wrong — they measure different populations."),
            msg("user", "So what's the actual approval?"),
            msg("assistant", "Best estimate using population weighting: 0.68 × 31% (rural proxy) + 0.32 × 42% (urban proxy) = 34.5%. But this is approximate — the Infotrak phone sample doesn't perfectly represent rural, and TIFA's urban sample isn't purely urban beneficiaries."),
        ],
        probes: vec![
            Probe {
                question: "What were the two different approval ratings and from which sources?",
                required_keywords: vec!["31", "42", "Infotrak", "TIFA"],
                partial_keywords: vec!["approval", "poll"],
                dimension: ProbeDimension::ContradictionAwareness,
            },
            Probe {
                question: "Why did the two polls disagree?",
                required_keywords: vec!["methodology", "rural", "urban"],
                partial_keywords: vec!["phone", "face-to-face", "sample"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "What was the estimated true approval using population weighting?",
                required_keywords: vec!["34.5"],
                partial_keywords: vec!["weighted", "estimate"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_contradiction_long() -> Scenario {
    let mut messages = vec![
        msg("user", "The model performance numbers are inconsistent between the paper and the benchmark results"),
        msg("assistant", "Let me check both sources of model performance claims."),
        msg("user", "[System ran `read_file` and got: Paper (Table 2): Qwen3-8B accuracy on ARC-C: 68.4%, HellaSwag: 81.2%, MMLU: 72.3%, TruthfulQA: 53.1%.]"),
        msg("assistant", "Paper claims noted. Let me check the actual benchmark run."),
        msg("user", "[System ran `read_file` and got: Benchmark results (our run, 2026-05-10): Qwen3-8B accuracy on ARC-C: 64.7%, HellaSwag: 79.8%, MMLU: 68.9%, TruthfulQA: 51.2%.]"),
        msg("assistant", "Discrepancy across all four benchmarks. Paper claims are 2-4pp higher than our measurements. Let me investigate."),
        msg("user", "[System ran `read_file` and got: Paper fine print: Results use 5-shot prompting with custom prompt template. Our run: 0-shot with default lm-eval-harness prompts.]"),
        msg("assistant", "Shot count difference: paper used 5-shot, we used 0-shot. That typically accounts for 2-5pp improvement. This likely explains the gap."),
    ];
    // Additional verification
    for bench in ["ARC-C", "HellaSwag", "MMLU", "TruthfulQA"] {
        messages.push(msg("user", &format!("[System ran `benchmark` and got: {} 5-shot result: {}%. Matches paper within margin of error (±0.3pp).]", bench, match bench {
            "ARC-C" => "68.1",
            "HellaSwag" => "81.0",
            "MMLU" => "72.0",
            "TruthfulQA" => "52.8",
            _ => "0",
        })));
        messages.push(msg("assistant", &format!("{} 5-shot matches paper. Discrepancy resolved for this benchmark.", bench)));
    }
    messages.push(msg("assistant", "All discrepancies resolved: the paper used 5-shot evaluation while our initial run was 0-shot. Re-running with 5-shot confirms paper claims within margin of error."));

    Scenario {
        id: "contradiction-long",
        problem_class: ProblemClass::Contradiction,
        messages,
        probes: vec![
            Probe {
                question: "What were the paper's claims vs our measurements for ARC-C?",
                required_keywords: vec!["68.4", "64.7"],
                partial_keywords: vec!["ARC", "paper", "our"],
                dimension: ProbeDimension::ContradictionAwareness,
            },
            Probe {
                question: "What caused the discrepancy between paper and our results?",
                required_keywords: vec!["5-shot", "0-shot"],
                partial_keywords: vec!["shot", "prompting", "prompt"],
                dimension: ProbeDimension::CausalFidelity,
            },
        ],
    }
}

// ── Temporal scenarios ───────────────────────────────────────────────

fn scenario_temporal_short() -> Scenario {
    Scenario {
        id: "temporal-short",
        problem_class: ProblemClass::Temporal,
        messages: vec![
            msg("user", "Show me how the GBP/USD rate changed over the last week"),
            msg("assistant", "Let me pull the daily GBP/USD rates for the past week."),
            msg("user", "[System ran `check_candles` and got: GBP/USD daily close: Mon 2026-05-12: 1.3247, Tue 2026-05-13: 1.3289 (+0.32%), Wed 2026-05-14: 1.3312 (+0.17%), Thu 2026-05-15: 1.3198 (-0.86%), Fri 2026-05-16: 1.3156 (-0.32%). Weekly change: -0.69%.]"),
            msg("assistant", "GBP/USD declined overall this week (-0.69%). It rose Monday through Wednesday (peaking at 1.3312) then dropped sharply Thursday-Friday. The Thursday drop of -0.86% was the biggest single-day move."),
            msg("user", "What happened Thursday to cause the drop?"),
            msg("assistant", "Let me check for news events on Thursday."),
            msg("user", "[System ran `web_search` and got: 2026-05-15: UK GDP data released below expectations (0.1% vs 0.3% expected). BoE Governor Bailey speech at 14:00 signaled potential rate cut in June. GBP sold off broadly.]"),
            msg("assistant", "Thursday's drop was driven by two events: (1) UK GDP miss (0.1% actual vs 0.3% expected), and (2) BoE Governor Bailey signaling a potential June rate cut."),
        ],
        probes: vec![
            Probe {
                question: "What was the GBP/USD rate on Wednesday (the weekly peak)?",
                required_keywords: vec!["1.3312"],
                partial_keywords: vec!["Wednesday", "peak"],
                dimension: ProbeDimension::TemporalOrdering,
            },
            Probe {
                question: "What caused the Thursday drop?",
                required_keywords: vec!["GDP", "Bailey", "rate cut"],
                partial_keywords: vec!["below expectations", "BoE"],
                dimension: ProbeDimension::CausalFidelity,
            },
            Probe {
                question: "What was the overall weekly change?",
                required_keywords: vec!["-0.69"],
                partial_keywords: vec!["decline", "weekly"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_temporal_medium() -> Scenario {
    Scenario {
        id: "temporal-medium",
        problem_class: ProblemClass::Temporal,
        messages: vec![
            msg("user", "Trace the history of CIF framework versions over time"),
            msg("assistant", "Let me retrieve the version history from the brain."),
            msg("user", "[System ran `brain_recall` and got: CIF version timeline: v0.1 (2024-11): Basic Ξ equation, 3 dimensions (D1-D3). v0.2 (2025-01): Added D4-D6, formalized convergence. v0.3 (2025-03): Added D7-D8, temporal parameter. v1.0 (2025-06): Influence Equation added, Φ(B) super-linear. v1.1 (2025-08): Trace Topology formalized (η_enc, B_enc, θ). v1.2 (2025-10): Problem Taxonomy, 17 classes. v2.0 (2026-01): Strand (Asidi kernel), recursive convergence. v2.1 (2026-03): Klein bottle topology at D7-D8 boundary. v2.2 (2026-05): Measured Forgetting v2 (CIF-informed context management).]"),
            msg("assistant", "Here's the full timeline, 9 versions over 18 months. The framework grew from 3 dimensions to a full recursive system."),
            msg("user", "When was Strand introduced and what version?"),
            msg("assistant", "Strand was introduced in v2.0, January 2026. It formalized the Asidi kernel as the atomic unit of convergence verification."),
            msg("user", "And the Influence Equation?"),
            msg("assistant", "Influence Equation was v1.0, June 2025. That's when Φ(B) = B^γ super-linear multiplier was added."),
            msg("user", "What about trace topology specifically?"),
            msg("assistant", "Trace Topology was formalized in v1.1, August 2025. Introduced η_enc (encoding efficiency), B_enc (encoding breadth), and θ (binding angle)."),
            // Filler
            msg("user", "How many total papers have we written?"),
            msg("assistant", "Based on the brain, we have 5 papers in various stages: Influence Equation (submitted), Trace Topology v4 (draft), Measured Forgetting (45% ready), Strand (outline), and Newcomb's Paradox (early draft)."),
            msg("user", "What's the most recent addition to the framework?"),
            msg("assistant", "The most recent is v2.2 (May 2026) — Measured Forgetting v2, which applies the Influence Equation, Trace Topology, and Problem Taxonomy to context management."),
        ],
        probes: vec![
            Probe {
                question: "When was the Influence Equation introduced and in which version?",
                required_keywords: vec!["v1.0", "June 2025"],
                partial_keywords: vec!["Influence", "Phi"],
                dimension: ProbeDimension::TemporalOrdering,
            },
            Probe {
                question: "What version introduced Strand and when?",
                required_keywords: vec!["v2.0", "January 2026"],
                partial_keywords: vec!["Strand", "Asidi kernel"],
                dimension: ProbeDimension::TemporalOrdering,
            },
            Probe {
                question: "What were the first three dimensions in v0.1?",
                required_keywords: vec!["D1", "D2", "D3"],
                partial_keywords: vec!["v0.1", "2024", "basic"],
                dimension: ProbeDimension::FactRetention,
            },
        ],
    }
}

fn scenario_temporal_long() -> Scenario {
    let mut messages = vec![
        msg("user", "Show me the deployment history of cif-graph over the past month"),
        msg("assistant", "Let me query the deployment log."),
    ];
    // Generate 12 deployment events
    let deployments = [
        ("2026-04-18", "v3.41", "digest forager bugfix", "success", "12s"),
        ("2026-04-20", "v3.42", "GNN retrain with cleaned data", "success", "45s"),
        ("2026-04-22", "v3.43", "R2 health endpoint", "success", "11s"),
        ("2026-04-25", "v3.44", "episode generator forager", "failed", "0s"),
        ("2026-04-25", "v3.44.1", "episode generator fix (import error)", "success", "14s"),
        ("2026-04-28", "v3.45", "honesty audit patches (27 gaps)", "success", "18s"),
        ("2026-05-01", "v3.46", "NLLB model integration", "success", "120s"),
        ("2026-05-03", "v3.47", "worker-mcp GKE_PROXY_URL fix", "success", "8s"),
        ("2026-05-07", "v3.48", "forager template system", "success", "15s"),
        ("2026-05-10", "v3.49", "Strand v2 specification", "success", "13s"),
        ("2026-05-13", "v3.50", "session cost tracking", "failed", "0s"),
        ("2026-05-13", "v3.50.1", "session cost fix (float parsing)", "success", "11s"),
    ];
    for (date, ver, desc, status, dur) in deployments {
        messages.push(msg("user", &format!("[System ran `graph_query` and got: Deploy {} on {}: {} — status: {}, duration: {}]", ver, date, desc, status, dur)));
        messages.push(msg("assistant", &format!("{} ({}) — {}. {}.", date, ver, desc, if status == "success" { "Deployed successfully" } else { "FAILED — fixed in patch" })));
    }
    messages.push(msg("assistant", "12 deployments in the past month. 2 failures (both fixed with immediate patches). Longest deploy was v3.46 (NLLB model, 120s due to model download)."));

    Scenario {
        id: "temporal-long",
        problem_class: ProblemClass::Temporal,
        messages,
        probes: vec![
            Probe {
                question: "How many deployments failed and on what dates?",
                required_keywords: vec!["2", "April 25", "May 13"],
                partial_keywords: vec!["failed", "patch"],
                dimension: ProbeDimension::TemporalOrdering,
            },
            Probe {
                question: "What was the longest deployment and why?",
                required_keywords: vec!["v3.46", "120", "NLLB"],
                partial_keywords: vec!["model", "longest"],
                dimension: ProbeDimension::FactRetention,
            },
            Probe {
                question: "What was deployed on May 10?",
                required_keywords: vec!["v3.49", "Strand"],
                partial_keywords: vec!["specification"],
                dimension: ProbeDimension::TemporalOrdering,
            },
        ],
    }
}

// ── Compression Strategies ───────────────────────────────────────────

/// Apply baseline compression: just truncate oldest messages.
pub fn compress_baseline(messages: &[Message], keep_recent: usize) -> Vec<Message> {
    let keep = keep_recent.min(messages.len());
    // Keep first message (question) + last N messages
    let mut result = vec![messages[0].clone()];
    result.push(Message {
        role: "user".to_string(),
        content: "[Prior context was truncated due to length]".to_string(),
    });
    result.extend_from_slice(&messages[messages.len() - keep..]);
    result
}

/// Apply v1 compression: uniform summarisation (all messages treated equally).
/// Returns the messages + a summariser prompt for uniform compression.
pub fn compress_v1(messages: &[Message], keep_recent: usize) -> (Vec<Message>, String, String) {
    let keep = keep_recent.min(messages.len().saturating_sub(1));
    let recent_start = messages.len() - keep;

    // Build uniform summary input — all messages equally
    let mut summary_input = String::from("Summarize the following conversation into key findings. Bullet points only. Maximum 5 bullets. Preserve all numbers.\n\n");
    for m in &messages[1..recent_start] {
        let content: String = m.content.chars().take(200).collect();
        summary_input.push_str(&format!("[{}]: {}\n", m.role, content.trim()));
    }

    let system = "Output only bullet points of key findings. No preamble.".to_string();

    (messages[recent_start..].to_vec(), system, summary_input)
}

/// Apply v2 compression: CIF-informed (uses the measured_forgetting module).
pub fn compress_v2(messages: &[Message], keep_recent: usize) -> crate::forgetting::CompactionAnalysis {
    crate::forgetting::analyze(messages, 0, keep_recent)
}

// ── Scoring Engine ───────────────────────────────────────────────────

/// Score a model's answer against a probe's expected keywords.
pub fn score_answer(answer: &str, probe: &Probe) -> u8 {
    let answer_lower = answer.to_lowercase();

    // Check required keywords (any match = score 3)
    let has_required = probe.required_keywords.iter().any(|kw| {
        answer_lower.contains(&kw.to_lowercase())
    });

    if has_required {
        return 3;
    }

    // Check partial keywords
    let partial_matches = probe.partial_keywords.iter()
        .filter(|kw| answer_lower.contains(&kw.to_lowercase()))
        .count();

    match partial_matches {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

// ── Results Reporting ────────────────────────────────────────────────

/// Aggregate results into a summary table.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub model: String,
    pub scores: Vec<ProbeScore>,
}

impl BenchmarkResults {
    pub fn mean_score(&self, condition: CompressionCondition) -> f64 {
        let filtered: Vec<&ProbeScore> = self.scores.iter()
            .filter(|s| s.condition == condition)
            .collect();
        if filtered.is_empty() { return 0.0; }
        filtered.iter().map(|s| s.score as f64).sum::<f64>() / filtered.len() as f64
    }

    pub fn mean_by_dimension(&self, condition: CompressionCondition, dim: ProbeDimension) -> f64 {
        let filtered: Vec<&ProbeScore> = self.scores.iter()
            .filter(|s| s.condition == condition && s.dimension == dim)
            .collect();
        if filtered.is_empty() { return 0.0; }
        filtered.iter().map(|s| s.score as f64).sum::<f64>() / filtered.len() as f64
    }

    pub fn mean_by_class(&self, condition: CompressionCondition, class_prefix: &str) -> f64 {
        let filtered: Vec<&ProbeScore> = self.scores.iter()
            .filter(|s| s.condition == condition && s.scenario_id.starts_with(class_prefix))
            .collect();
        if filtered.is_empty() { return 0.0; }
        filtered.iter().map(|s| s.score as f64).sum::<f64>() / filtered.len() as f64
    }

    /// Print a formatted results table (for paper).
    pub fn print_table(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Model: {}", self.model);
        println!("{}", "=".repeat(60));
        println!("{:<12} {:>10} {:>10} {:>10}", "Condition", "Baseline", "V1", "V2");
        println!("{:-<12} {:-<10} {:-<10} {:-<10}", "", "", "", "");
        println!("{:<12} {:>10.2} {:>10.2} {:>10.2}", "Overall",
            self.mean_score(CompressionCondition::Baseline),
            self.mean_score(CompressionCondition::UniformV1),
            self.mean_score(CompressionCondition::CifV2),
        );
        println!();
        println!("By dimension:");
        for dim in [ProbeDimension::FactRetention, ProbeDimension::CausalFidelity,
                    ProbeDimension::ContradictionAwareness, ProbeDimension::TemporalOrdering,
                    ProbeDimension::EntityRecall] {
            let label = match dim {
                ProbeDimension::FactRetention => "Fact",
                ProbeDimension::CausalFidelity => "Causal",
                ProbeDimension::ContradictionAwareness => "Contradict",
                ProbeDimension::TemporalOrdering => "Temporal",
                ProbeDimension::EntityRecall => "Entity",
            };
            println!("  {:<10} {:>10.2} {:>10.2} {:>10.2}", label,
                self.mean_by_dimension(CompressionCondition::Baseline, dim),
                self.mean_by_dimension(CompressionCondition::UniformV1, dim),
                self.mean_by_dimension(CompressionCondition::CifV2, dim),
            );
        }
        println!();
        println!("By problem class:");
        for class in ["lookup", "multihop", "exploratory", "aggregation", "contradiction", "temporal"] {
            println!("  {:<12} {:>10.2} {:>10.2} {:>10.2}", class,
                self.mean_by_class(CompressionCondition::Baseline, class),
                self.mean_by_class(CompressionCondition::UniformV1, class),
                self.mean_by_class(CompressionCondition::CifV2, class),
            );
        }
        println!("{}\n", "=".repeat(60));
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_generation() {
        let scenarios = generate_scenarios();
        assert_eq!(scenarios.len(), 18, "should generate 18 scenarios");

        // Each scenario should have at least 2 probes
        for s in &scenarios {
            assert!(!s.probes.is_empty(), "scenario {} has no probes", s.id);
            assert!(s.messages.len() >= 4, "scenario {} too short", s.id);
        }
    }

    #[test]
    fn test_scoring_exact_match() {
        let probe = Probe {
            question: "What was the gold price?",
            required_keywords: vec!["2847", "2,847"],
            partial_keywords: vec!["gold", "troy"],
            dimension: ProbeDimension::FactRetention,
        };
        assert_eq!(score_answer("The gold price was $2,847.30", &probe), 3);
        assert_eq!(score_answer("Gold was around $2847 per ounce", &probe), 3);
    }

    #[test]
    fn test_scoring_partial_match() {
        let probe = Probe {
            question: "What was the gold price?",
            required_keywords: vec!["2847", "2,847"],
            partial_keywords: vec!["gold", "troy", "ounce"],
            dimension: ProbeDimension::FactRetention,
        };
        // Has partial keywords but not the exact number
        assert_eq!(score_answer("Gold was trading at a high level per troy ounce", &probe), 2);
        assert_eq!(score_answer("The gold market was active", &probe), 1);
        assert_eq!(score_answer("I don't have that information", &probe), 0);
    }

    #[test]
    fn test_baseline_compression() {
        let messages = vec![
            msg("user", "What is X?"),
            msg("assistant", "Let me check."),
            msg("user", "[Result: X=42]"),
            msg("assistant", "X is 42."),
            msg("user", "What about Y?"),
            msg("assistant", "Y is 7."),
        ];
        let compressed = compress_baseline(&messages, 2);
        // Should keep: original question + truncation notice + last 2
        assert_eq!(compressed.len(), 4);
        assert_eq!(compressed[0].content, "What is X?");
        assert!(compressed[1].content.contains("truncated"));
    }

    #[test]
    fn test_v2_preserves_high_influence() {
        let messages = vec![
            msg("user", "How many CHWs are active in Busia county?"),
            msg("assistant", "Let me query."),
            msg("user", "[System ran `query` and got: 1,089 active CHWs in Busia. Breakdown by sub-county: Teso North 234, Teso South 198. The increase is because 47 new CHWs were onboarded in March 2026.]"),
            msg("assistant", "There are 1,089 active CHWs."),
            msg("user", "<tool_error>connection timeout</tool_error>"),
            msg("assistant", "<tool_error>retry failed</tool_error>"),
            msg("user", "[System ran `query` and got: Confirmed 1,089. Status breakdown: active=1,089, inactive=234, suspended=12.]"),
            msg("assistant", "Confirmed: 1,089 active."),
            msg("user", "thanks"),
            msg("assistant", "You're welcome."),
        ];

        let analysis = compress_v2(&messages, 2);
        // The high-influence message (index 2 with data+causal+structural) should be preserved
        assert!(analysis.stats.preserved_count > 0);
        // Errors should be heavily compressed
        assert!(analysis.stats.heavy_compressed > 0);
        // The preserved text should contain the key data
        assert!(analysis.preserved_text.contains("1,089") || analysis.preserved_text.contains("1089"));
    }

    #[test]
    fn test_all_problem_classes_represented() {
        let scenarios = generate_scenarios();
        let classes: std::collections::HashSet<_> = scenarios.iter()
            .map(|s| std::mem::discriminant(&s.problem_class))
            .collect();
        assert_eq!(classes.len(), 6, "all 6 problem classes should be represented");
    }

    #[test]
    fn test_probe_dimensions_covered() {
        let scenarios = generate_scenarios();
        let dims: std::collections::HashSet<_> = scenarios.iter()
            .flat_map(|s| s.probes.iter().map(|p| std::mem::discriminant(&p.dimension)))
            .collect();
        assert_eq!(dims.len(), 5, "all 5 probe dimensions should be covered");
    }
}
