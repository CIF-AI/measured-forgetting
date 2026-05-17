/// Measured Forgetting v2 — CIF-Informed Context Management
///
/// Three instruments from CIF theory applied to context compression:
///
/// 1. **Influence Equation**: I(m, task, t) = Φ(B_m) × Σ_d [w_d × C_d]
///    Messages with multi-dimensional convergence are exponentially more
///    worth preserving (super-linear Φ multiplier).
///
/// 2. **Trace Topology**: Causal chains (A → B → C) are binding units.
///    Compress them together, preserving the trace shape. Independent
///    lookups can be compressed independently.
///
/// 3. **Problem Taxonomy**: Task class determines what the "sufficient
///    statistic" is. Lookup → answer. Multi-hop → chain. Exploration → successes.
///
/// The algorithm:
///   1. Classify task into problem class
///   2. Score each message by influence (multi-dimensional, with Φ)
///   3. Detect trace topology (causal chains vs independent)
///   4. Apply κ_d saturation (sw-cap repeated dimensions)
///   5. Compress dimension-aware: structural dims preserved, patterned dims dropped
///   6. Target ~5 maximally-independent summary bullets (Φ_net peak)

use crate::Message;

// ── Problem Classes ──────────────────────────────────────────────────

/// Task classification for adaptive compression strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// Single fact needed — compress everything except the answer.
    Lookup,
    /// A → B → C reasoning — preserve chain topology.
    MultiHop,
    /// Try many approaches, find few — preserve successes, compress failures.
    Exploratory,
    /// Collect many data points — preserve summaries, compress individuals.
    Aggregation,
    /// Two sources disagree — preserve BOTH paths (they are the kernel).
    Contradiction,
    /// State changes over time — preserve chronological order.
    Temporal,
}

impl ProblemClass {
    /// Classify from the original user question and tool history so far.
    pub fn classify(user_question: &str, messages: &[Message]) -> Self {
        let q = user_question.to_lowercase();
        let msg_count = messages.len();

        // Count tool errors and successes in the history
        let error_count = messages.iter()
            .filter(|m| m.content.contains("<tool_error>") || m.content.contains("[graph_query error"))
            .count();
        let tool_result_count = messages.iter()
            .filter(|m| m.content.contains("<tool_result>") || m.content.contains("[System ran"))
            .count();

        // Contradiction: user asking about disagreement, or two conflicting results
        if q.contains("disagree") || q.contains("conflict") || q.contains("different")
            || q.contains("mismatch") || q.contains("inconsistent") {
            return ProblemClass::Contradiction;
        }

        // Temporal: user asking about change over time
        if q.contains("over time") || q.contains("trend") || q.contains("history")
            || q.contains("changed") || q.contains("when did") || q.contains("timeline") {
            return ProblemClass::Temporal;
        }

        // Aggregation: user asking for summaries/counts/lists
        if q.contains("how many") || q.contains("list all") || q.contains("summarize")
            || q.contains("total") || q.contains("count") || q.contains("average") {
            return ProblemClass::Aggregation;
        }

        // Exploratory: many errors relative to successes (trying approaches)
        if msg_count > 6 && error_count > tool_result_count {
            return ProblemClass::Exploratory;
        }

        // Multi-hop: multiple successful tool calls building on each other
        if tool_result_count >= 3 && error_count <= 1 {
            return ProblemClass::MultiHop;
        }

        // Lookup: short question, few tool calls expected
        if q.split_whitespace().count() < 12 && msg_count < 8 {
            return ProblemClass::Lookup;
        }

        // Default: multi-hop (safest — preserves structure)
        ProblemClass::MultiHop
    }
}

// ── Influence Scoring ────────────────────────────────────────────────

/// Dimensions a message can converge on relative to the task.
#[derive(Debug, Clone, Copy)]
struct DimensionScores {
    /// Contains numeric data (measurements, counts, IDs)
    data: f64,
    /// Contains causal explanation (because, therefore, caused by)
    causal: f64,
    /// References the original question directly
    task_reference: f64,
    /// Contains entity names/identifiers relevant to task
    entity: f64,
    /// Contains temporal information (dates, sequences, ordering)
    temporal: f64,
    /// Contains structural data (schemas, relationships, hierarchies)
    structural: f64,
}

impl DimensionScores {
    fn active_dimensions(&self) -> u32 {
        let dims = [self.data, self.causal, self.task_reference,
                    self.entity, self.temporal, self.structural];
        dims.iter().filter(|&&d| d > 0.0).count() as u32
    }

    fn weighted_sum(&self, susceptibility: &Susceptibility) -> f64 {
        self.data * susceptibility.data
            + self.causal * susceptibility.causal
            + self.task_reference * susceptibility.task_reference
            + self.entity * susceptibility.entity
            + self.temporal * susceptibility.temporal
            + self.structural * susceptibility.structural
    }
}

/// Per-dimension susceptibility — how much the task still NEEDS this dimension.
/// Decays as later messages provide the same type of information.
#[derive(Debug, Clone)]
struct Susceptibility {
    data: f64,
    causal: f64,
    task_reference: f64,
    entity: f64,
    temporal: f64,
    structural: f64,
}

impl Susceptibility {
    fn new() -> Self {
        Self {
            data: 1.0,
            causal: 1.0,
            task_reference: 1.0,
            entity: 1.0,
            temporal: 1.0,
            structural: 1.0,
        }
    }

    /// Decay susceptibility on dimensions that have been saturated by later messages.
    /// Messages are processed newest-first, so earlier messages decay more.
    fn decay(&mut self, scores: &DimensionScores, lambda: f64) {
        if scores.data > 0.0 { self.data *= lambda; }
        if scores.causal > 0.0 { self.causal *= lambda; }
        if scores.task_reference > 0.0 { self.task_reference *= lambda; }
        if scores.entity > 0.0 { self.entity *= lambda; }
        if scores.temporal > 0.0 { self.temporal *= lambda; }
        if scores.structural > 0.0 { self.structural *= lambda; }
    }
}

/// Resonance multiplier Φ(B) = B^γ where γ = 1.5
/// The super-linear multiplier that makes multi-dimensional messages
/// exponentially more valuable than single-dimensional ones.
fn phi(b: u32) -> f64 {
    if b == 0 { return 0.0; }
    (b as f64).powf(1.5)
}

/// Score a single message's dimensions relative to the task.
fn score_dimensions(msg: &Message, user_question: &str) -> DimensionScores {
    let content = &msg.content;
    let content_lower = content.to_lowercase();
    let q_lower = user_question.to_lowercase();

    // Data dimension: numbers, measurements, counts
    let digit_chars = content.chars().filter(|c| c.is_ascii_digit()).count();
    let data = if digit_chars > 5 {
        (digit_chars as f64 / content.len().max(1) as f64 * 10.0).min(1.0)
    } else {
        0.0
    };

    // Causal dimension: explanatory connectors
    let causal_markers = ["because", "therefore", "caused by", "results in",
                          "leads to", "due to", "so that", "which means", "→"];
    let causal = if causal_markers.iter().any(|m| content_lower.contains(m)) {
        0.8
    } else {
        0.0
    };

    // Task reference: words from the original question appearing in the message
    let q_words: Vec<&str> = q_lower.split_whitespace()
        .filter(|w| w.len() > 3) // skip short words
        .collect();
    let matching_words = q_words.iter()
        .filter(|w| content_lower.contains(*w))
        .count();
    let task_reference = if q_words.is_empty() {
        0.0
    } else {
        (matching_words as f64 / q_words.len() as f64).min(1.0)
    };

    // Entity dimension: proper nouns, identifiers, quoted strings
    let has_entities = content.contains('"') || content.contains('\'')
        || content.chars().filter(|c| c.is_uppercase()).count() > content.len() / 10;
    let entity = if has_entities { 0.6 } else { 0.0 };

    // Temporal dimension: dates, timestamps, ordinal markers
    let temporal_markers = ["2024", "2025", "2026", "yesterday", "today", "last week",
                           "first", "then", "after", "before", "step 1", "step 2"];
    let temporal = if temporal_markers.iter().any(|m| content_lower.contains(m)) {
        0.7
    } else {
        0.0
    };

    // Structural dimension: schemas, hierarchies, relationships
    let structural_markers = ["schema", "table", "column", "field", "parent",
                             "child", "hierarchy", "relationship", "foreign key",
                             "index", "constraint"];
    let structural = if structural_markers.iter().any(|m| content_lower.contains(m)) {
        0.7
    } else {
        0.0
    };

    DimensionScores { data, causal, task_reference, entity, temporal, structural }
}

/// Scored message with influence value and metadata for compression decisions.
#[derive(Debug, Clone)]
pub struct ScoredMessage {
    pub index: usize,
    pub influence: f64,
    pub active_dims: u32,
    pub is_error: bool,
    pub is_tool_result: bool,
    /// Index of message this one causally depends on (trace binding).
    pub depends_on: Option<usize>,
}

/// Score all messages in the compressible zone using the Influence Equation.
///
/// Returns messages scored by influence, with susceptibility decay applied
/// (later messages reduce the value of earlier messages on the same dimension).
pub fn score_messages(
    messages: &[Message],
    compressible_range: std::ops::Range<usize>,
    user_question: &str,
) -> Vec<ScoredMessage> {
    let compressible = &messages[compressible_range.clone()];
    if compressible.is_empty() {
        return Vec::new();
    }

    // First pass: score dimensions for each message
    let dim_scores: Vec<DimensionScores> = compressible.iter()
        .map(|m| score_dimensions(m, user_question))
        .collect();

    // Second pass: compute susceptibility decay (newest messages decay older ones)
    // Process from newest to oldest — newer information supersedes older
    let lambda = 0.7; // decay rate per superseding message
    let mut susceptibilities: Vec<Susceptibility> = vec![Susceptibility::new(); compressible.len()];

    for i in (0..compressible.len()).rev() {
        // For each message, decay the susceptibility of all older messages
        // on the same dimensions
        if i > 0 {
            for j in (0..i).rev() {
                susceptibilities[j].decay(&dim_scores[i], lambda);
            }
        }
    }

    // Third pass: compute influence = Φ(B) × Σ_d [w_d × C_d]
    let mut scored: Vec<ScoredMessage> = Vec::with_capacity(compressible.len());

    for (local_idx, msg) in compressible.iter().enumerate() {
        let global_idx = compressible_range.start + local_idx;
        let dims = &dim_scores[local_idx];
        let susc = &susceptibilities[local_idx];

        let b = dims.active_dimensions();
        let weighted_sum = dims.weighted_sum(susc);
        let influence = phi(b) * weighted_sum;

        let is_error = msg.content.contains("<tool_error>")
            || msg.content.contains("[graph_query error");
        let is_tool_result = msg.content.contains("<tool_result>")
            || msg.content.contains("[System ran");

        // Detect causal dependency: if this message references data from the previous
        // tool result, it depends on it (trace binding)
        let depends_on = if local_idx > 0 && msg.role == "assistant" {
            // Assistant message after a tool result likely references it
            let prev = &compressible[local_idx - 1];
            if prev.content.contains("<tool_result>") || prev.content.contains("[System ran") {
                Some(compressible_range.start + local_idx - 1)
            } else {
                None
            }
        } else {
            None
        };

        scored.push(ScoredMessage {
            index: global_idx,
            influence,
            active_dims: b,
            is_error,
            is_tool_result,
            depends_on,
        });
    }

    scored
}

// ── Trace Topology Detection ─────────────────────────────────────────

/// A causal chain: messages that form a trace (A → B → C).
/// Must be compressed together to preserve binding.
#[derive(Debug, Clone)]
pub struct CausalChain {
    /// Indices into the messages array (global indices).
    pub indices: Vec<usize>,
    /// Combined influence of the chain.
    pub combined_influence: f64,
}

/// Detect causal chains in the scored messages.
/// A chain exists when message B depends on A, and C depends on B.
pub fn detect_chains(scored: &[ScoredMessage]) -> Vec<CausalChain> {
    if scored.is_empty() {
        return Vec::new();
    }

    // Build dependency graph
    let mut chains: Vec<CausalChain> = Vec::new();
    let mut in_chain: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Walk forward, building chains from dependencies
    for sm in scored.iter() {
        if in_chain.contains(&sm.index) {
            continue;
        }

        if let Some(dep) = sm.depends_on {
            // Find or create a chain that ends with `dep`
            let existing = chains.iter_mut().find(|c| {
                c.indices.last() == Some(&dep)
            });

            if let Some(chain) = existing {
                chain.indices.push(sm.index);
                chain.combined_influence += sm.influence;
                in_chain.insert(sm.index);
            } else if !in_chain.contains(&dep) {
                // Start a new chain
                let dep_influence = scored.iter()
                    .find(|s| s.index == dep)
                    .map(|s| s.influence)
                    .unwrap_or(0.0);
                chains.push(CausalChain {
                    indices: vec![dep, sm.index],
                    combined_influence: dep_influence + sm.influence,
                });
                in_chain.insert(dep);
                in_chain.insert(sm.index);
            }
        }
    }

    chains
}

// ── κ_d Saturation Detection ─────────────────────────────────────────

/// Detect saturated dimensions (sw-cap): when 3+ messages all provide
/// the same type of information, cap them in the summary.
pub struct SaturationReport {
    /// Number of error messages (cap to "N errors, type: X")
    pub error_count: usize,
    /// Dominant error type if saturated
    pub dominant_error: Option<String>,
    /// Number of messages with high data dimension (may be redundant)
    pub data_saturated: bool,
}

pub fn detect_saturation(scored: &[ScoredMessage], messages: &[Message]) -> SaturationReport {
    let error_count = scored.iter().filter(|s| s.is_error).count();

    let dominant_error = if error_count >= 3 {
        // Find the most common error pattern
        let errors: Vec<&str> = scored.iter()
            .filter(|s| s.is_error)
            .filter_map(|s| messages.get(s.index))
            .map(|m| {
                m.content.lines()
                    .find(|l| l.to_lowercase().contains("error"))
                    .unwrap_or("unknown error")
            })
            .collect();
        errors.first().map(|e| e.to_string())
    } else {
        None
    };

    // Data saturation: if 5+ messages all have high data scores,
    // later ones may supersede earlier ones
    let data_rich = scored.iter().filter(|s| s.active_dims >= 1 && s.is_tool_result).count();
    let data_saturated = data_rich >= 5;

    SaturationReport { error_count, dominant_error, data_saturated }
}

// ── Compression Strategy ─────────────────────────────────────────────

/// Instructions for the summariser, adapted to problem class and influence scores.
pub struct CompressionPlan {
    /// Messages to preserve verbatim (high-influence, multi-dimensional)
    pub preserve_verbatim: Vec<usize>,
    /// Messages to compress lightly (medium influence, in causal chains)
    pub compress_light: Vec<usize>,
    /// Messages to compress aggressively (low influence, errors, saturated)
    pub compress_heavy: Vec<usize>,
    /// The summariser system prompt (adapted to problem class)
    pub summariser_system: String,
    /// The summariser user prompt template
    pub summariser_instruction: String,
    /// Target bullet count for the summary (Φ_net peak ≈ 5)
    pub target_bullets: usize,
}

/// Build a compression plan from scored messages, chains, and saturation.
pub fn plan_compression(
    scored: &[ScoredMessage],
    chains: &[CausalChain],
    saturation: &SaturationReport,
    problem_class: ProblemClass,
    messages: &[Message],
) -> CompressionPlan {
    if scored.is_empty() {
        return CompressionPlan {
            preserve_verbatim: Vec::new(),
            compress_light: Vec::new(),
            compress_heavy: Vec::new(),
            summariser_system: String::new(),
            summariser_instruction: String::new(),
            target_bullets: 5,
        };
    }

    // Compute influence threshold: median influence
    let mut influences: Vec<f64> = scored.iter().map(|s| s.influence).collect();
    influences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = influences[influences.len() / 2];

    // Indices of messages in causal chains (preserve as units)
    let chain_indices: std::collections::HashSet<usize> = chains.iter()
        .flat_map(|c| c.indices.iter().copied())
        .collect();

    let mut preserve_verbatim: Vec<usize> = Vec::new();
    let mut compress_light: Vec<usize> = Vec::new();
    let mut compress_heavy: Vec<usize> = Vec::new();

    for sm in scored {
        if sm.is_error {
            // Errors always compress heavily (unless in a chain that matters)
            if chain_indices.contains(&sm.index) && sm.influence > median {
                compress_light.push(sm.index);
            } else {
                compress_heavy.push(sm.index);
            }
        } else if sm.influence > median && sm.active_dims >= 3 {
            // High influence + multi-dimensional → preserve
            preserve_verbatim.push(sm.index);
        } else if chain_indices.contains(&sm.index) {
            // Part of a causal chain → compress lightly (preserve binding)
            compress_light.push(sm.index);
        } else if sm.influence > median * 0.5 {
            // Moderate influence → light compression
            compress_light.push(sm.index);
        } else {
            // Low influence → heavy compression
            compress_heavy.push(sm.index);
        }
    }

    // Problem-class-specific adjustments
    match problem_class {
        ProblemClass::Contradiction => {
            // Never heavily compress contradicting data — move to light
            let moved: Vec<usize> = compress_heavy.iter()
                .filter(|&&idx| {
                    messages.get(idx)
                        .map(|m| m.content.contains("<tool_result>") || m.content.contains("[System ran"))
                        .unwrap_or(false)
                })
                .copied()
                .collect();
            compress_heavy.retain(|idx| !moved.contains(idx));
            compress_light.extend(moved);
        }
        ProblemClass::Exploratory => {
            // Errors are expected in exploration — compress even more aggressively
            // (already handled by default heavy compression of errors)
        }
        ProblemClass::Lookup => {
            // Keep only the message that most likely contains the answer
            // (highest influence tool result)
            if let Some(best) = scored.iter()
                .filter(|s| s.is_tool_result)
                .max_by(|a, b| a.influence.partial_cmp(&b.influence).unwrap_or(std::cmp::Ordering::Equal))
            {
                if !preserve_verbatim.contains(&best.index) {
                    preserve_verbatim.push(best.index);
                    compress_light.retain(|&idx| idx != best.index);
                    compress_heavy.retain(|&idx| idx != best.index);
                }
            }
        }
        _ => {}
    }

    // Build summariser prompts
    let (summariser_system, summariser_instruction) = build_summariser_prompts(
        problem_class, saturation, &preserve_verbatim, messages,
    );

    // Target bullets: Φ_net peaks at B≈5, so aim for 5 independent summary bullets
    let target_bullets = 5;

    CompressionPlan {
        preserve_verbatim,
        compress_light,
        compress_heavy,
        summariser_system,
        summariser_instruction,
        target_bullets,
    }
}

/// Build problem-class-aware summariser prompts.
fn build_summariser_prompts(
    problem_class: ProblemClass,
    saturation: &SaturationReport,
    _preserve_indices: &[usize],
    _messages: &[Message],
) -> (String, String) {
    let system = "Output only bullet points of key findings. No preamble. No tool calls. No thinking tags.".to_string();

    let mut instruction = String::from(
        "Summarize these exchanges into key findings.\n"
    );

    // Problem-class-specific instructions
    match problem_class {
        ProblemClass::Lookup => {
            instruction.push_str("Focus on THE ANSWER. Drop everything that isn't the direct answer to the question.\n");
        }
        ProblemClass::MultiHop => {
            instruction.push_str("Preserve the CHAIN of reasoning: what led to what. Use arrows (→) for dependencies.\n");
        }
        ProblemClass::Exploratory => {
            instruction.push_str("Focus on what WORKED. Failed approaches → one line each (type only). Successes → full detail.\n");
        }
        ProblemClass::Aggregation => {
            instruction.push_str("Preserve COUNTS and TOTALS. Individual items → summarize as categories with counts.\n");
        }
        ProblemClass::Contradiction => {
            instruction.push_str("Preserve BOTH conflicting data points. Do NOT resolve the contradiction — state both sides.\n");
        }
        ProblemClass::Temporal => {
            instruction.push_str("Preserve CHRONOLOGICAL ORDER. Include all dates/timestamps. Sequence matters.\n");
        }
    }

    // Saturation instructions
    if saturation.error_count >= 3 {
        if let Some(ref err) = saturation.dominant_error {
            instruction.push_str(&format!(
                "\nNote: {} errors occurred (type: {}). Summarize as one line, not individually.\n",
                saturation.error_count, err
            ));
        }
    }

    instruction.push_str("\nRules:\n- Bullet points only\n- Preserve ALL numbers and data\n- Maximum 5 bullets\n- Drop verbose formatting\n\n");

    (system, instruction)
}

// ── Main Entry Point ─────────────────────────────────────────────────

/// Build the text to feed to the summariser, applying the compression plan.
///
/// Returns (text_for_summariser, preserved_messages_text).
/// - `text_for_summariser`: the content to compress (light + heavy sections)
/// - `preserved_messages_text`: high-influence content to keep verbatim in the summary
pub fn build_compaction_input(
    plan: &CompressionPlan,
    messages: &[Message],
) -> (String, String) {
    let mut preserved = String::new();
    let mut to_summarise = String::new();

    // Preserved verbatim: these stay in the final summary as-is
    for &idx in &plan.preserve_verbatim {
        if let Some(msg) = messages.get(idx) {
            // Truncate preserved messages to 400 chars max (key data, not raw output)
            let content: String = msg.content.chars().take(400).collect();
            preserved.push_str(&format!("- {}\n", content.trim()));
        }
    }

    // Light compression: include full content but truncated
    to_summarise.push_str("[Messages to compress — preserve causal connections]\n");
    for &idx in &plan.compress_light {
        if let Some(msg) = messages.get(idx) {
            let content: String = msg.content.chars().take(300).collect();
            to_summarise.push_str(&format!("[{}]: {}\n", msg.role, content.trim()));
        }
    }

    // Heavy compression: include only the message type and one-line summary
    if !plan.compress_heavy.is_empty() {
        to_summarise.push_str("\n[Messages to compress aggressively — type/outcome only]\n");
        for &idx in &plan.compress_heavy {
            if let Some(msg) = messages.get(idx) {
                if msg.content.contains("<tool_error>") || msg.content.contains("[graph_query error") {
                    let short = msg.content.lines()
                        .find(|l| l.to_lowercase().contains("error"))
                        .unwrap_or("(tool error)");
                    to_summarise.push_str(&format!("[error]: {}\n", short.trim()));
                } else {
                    // One-line extract
                    let first_line = msg.content.lines()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("(empty)");
                    let short: String = first_line.chars().take(80).collect();
                    to_summarise.push_str(&format!("[{}]: {}\n", msg.role, short));
                }
            }
        }
    }

    (to_summarise, preserved)
}

/// Assemble the final summary message to insert into the message array.
/// Combines the model-generated summary with preserved high-influence content.
pub fn assemble_summary(
    model_summary: &str,
    preserved: &str,
    problem_class: ProblemClass,
) -> String {
    let class_label = match problem_class {
        ProblemClass::Lookup => "lookup",
        ProblemClass::MultiHop => "multi-hop",
        ProblemClass::Exploratory => "exploration",
        ProblemClass::Aggregation => "aggregation",
        ProblemClass::Contradiction => "contradiction",
        ProblemClass::Temporal => "temporal",
    };

    let mut summary = format!("[Prior investigation summary ({})]", class_label);

    if !preserved.is_empty() {
        summary.push_str("\n\nKey findings (preserved):\n");
        summary.push_str(preserved);
    }

    if !model_summary.trim().is_empty() {
        summary.push_str("\n\nCompressed context:\n");
        summary.push_str(model_summary.trim());
    }

    summary.push_str("\n\nContinue from here. Use what you learned above.");
    summary
}

// ── Orchestrator: the full v2 pipeline ───────────────────────────────

/// Complete measured forgetting v2 analysis.
/// Call this when context threshold is breached.
///
/// Returns:
/// - The summariser request text (to feed to the model)
/// - The preserved content (to include verbatim)
/// - The compression plan metadata
/// - The problem class detected
pub struct CompactionAnalysis {
    pub text_for_summariser: String,
    pub preserved_text: String,
    pub summariser_system: String,
    pub summariser_instruction: String,
    pub problem_class: ProblemClass,
    pub stats: CompactionStats,
}

#[derive(Debug, Clone)]
pub struct CompactionStats {
    pub total_messages: usize,
    pub preserved_count: usize,
    pub light_compressed: usize,
    pub heavy_compressed: usize,
    pub error_count: usize,
    pub chain_count: usize,
    pub median_influence: f64,
    pub max_influence: f64,
}

/// Run the full v2 measured forgetting analysis.
pub fn analyze(
    messages: &[Message],
    original_question_idx: usize,
    recent_count: usize,
) -> CompactionAnalysis {
    // Determine the compressible range
    let keep_recent = recent_count.min(messages.len().saturating_sub(original_question_idx + 1));
    let recent_start = messages.len() - keep_recent;
    let compressible_range = (original_question_idx + 1)..recent_start;

    // Get the user's original question
    let user_question = messages.get(original_question_idx)
        .map(|m| m.content.as_str())
        .unwrap_or("");

    // 1. Classify problem
    let problem_class = ProblemClass::classify(user_question, messages);

    // 2. Score messages by influence
    let scored = score_messages(messages, compressible_range.clone(), user_question);

    // 3. Detect causal chains (trace topology)
    let chains = detect_chains(&scored);

    // 4. Detect κ_d saturation
    let saturation = detect_saturation(&scored, messages);

    // 5. Build compression plan
    let plan = plan_compression(&scored, &chains, &saturation, problem_class, messages);

    // 6. Build the compaction input
    let (text_for_summariser, preserved_text) = build_compaction_input(&plan, messages);

    // Stats for logging
    let mut influences: Vec<f64> = scored.iter().map(|s| s.influence).collect();
    influences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_influence = influences.get(influences.len() / 2).copied().unwrap_or(0.0);
    let max_influence = influences.last().copied().unwrap_or(0.0);

    let stats = CompactionStats {
        total_messages: scored.len(),
        preserved_count: plan.preserve_verbatim.len(),
        light_compressed: plan.compress_light.len(),
        heavy_compressed: plan.compress_heavy.len(),
        error_count: saturation.error_count,
        chain_count: chains.len(),
        median_influence,
        max_influence,
    };

    CompactionAnalysis {
        text_for_summariser,
        preserved_text,
        summariser_system: plan.summariser_system,
        summariser_instruction: plan.summariser_instruction,
        problem_class,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Message {
        Message { role: role.to_string(), content: content.to_string() }
    }

    #[test]
    fn test_phi_super_linear() {
        assert_eq!(phi(0), 0.0);
        assert_eq!(phi(1), 1.0);
        assert!((phi(2) - 2.83).abs() < 0.01);
        assert!((phi(3) - 5.20).abs() < 0.01);
        assert!((phi(5) - 11.18).abs() < 0.01);
    }

    #[test]
    fn test_problem_classification_lookup() {
        let messages = vec![
            msg("user", "What is the price of gold?"),
        ];
        assert_eq!(ProblemClass::classify("What is the price of gold?", &messages), ProblemClass::Lookup);
    }

    #[test]
    fn test_problem_classification_temporal() {
        let messages = vec![
            msg("user", "How has the GBP/USD rate changed over time?"),
        ];
        assert_eq!(
            ProblemClass::classify("How has the GBP/USD rate changed over time?", &messages),
            ProblemClass::Temporal
        );
    }

    #[test]
    fn test_problem_classification_exploratory() {
        let mut messages = vec![msg("user", "Find the bug in the pipeline")];
        // Add many errors
        for _ in 0..5 {
            messages.push(msg("assistant", "<tool_error>connection refused</tool_error>"));
        }
        messages.push(msg("assistant", "<tool_result>found it</tool_result>"));
        assert_eq!(
            ProblemClass::classify("Find the bug in the pipeline", &messages),
            ProblemClass::Exploratory
        );
    }

    #[test]
    fn test_influence_scoring_multi_dim() {
        let m = msg("assistant", "<tool_result>The query returned 47 rows because the filter on reported_date in 2026 matched. Schema: dim_chw has columns contact_id, geo_4_id.</tool_result>");
        let scores = score_dimensions(&m, "how many CHWs reported in 2026?");
        // Should have: data (47, 2026), causal (because), task_reference (CHWs, 2026), structural (schema, columns)
        assert!(scores.data > 0.0, "data dimension should fire");
        assert!(scores.causal > 0.0, "causal dimension should fire");
        assert!(scores.structural > 0.0, "structural dimension should fire");
        assert!(scores.active_dimensions() >= 3, "should have 3+ active dims");
    }

    #[test]
    fn test_influence_scoring_error_low() {
        let m = msg("assistant", "<tool_error>connection timeout</tool_error>");
        let scores = score_dimensions(&m, "how many CHWs?");
        // Errors have low dimensional engagement
        assert!(scores.active_dimensions() <= 1);
    }

    #[test]
    fn test_susceptibility_decay() {
        let mut susc = Susceptibility::new();
        let scores = DimensionScores {
            data: 0.8, causal: 0.0, task_reference: 0.0,
            entity: 0.0, temporal: 0.0, structural: 0.0,
        };
        susc.decay(&scores, 0.7);
        assert!((susc.data - 0.7).abs() < 0.01, "data susceptibility should decay");
        assert_eq!(susc.causal, 1.0, "causal should not decay (dimension inactive)");
    }

    #[test]
    fn test_full_analysis() {
        let messages = vec![
            msg("user", "How many CHWs are active in Busia county?"),
            msg("assistant", "Let me query the database for active CHWs in Busia."),
            msg("user", "[System ran `query` and got: 247 active CHWs in Busia county as of 2026-05-01. Breakdown: 180 active, 67 inactive. The drop is because 15 were deactivated in April due to missed submissions.]"),
            msg("assistant", "Based on the data, there are 247 CHWs registered in Busia, with 180 currently active."),
            msg("user", "<tool_error>timeout querying historical data</tool_error>"),
            msg("assistant", "<tool_error>retry failed</tool_error>"),
            msg("user", "[System ran `query_v2` and got: Historical trend: Jan=195, Feb=200, Mar=210, Apr=180. The April drop correlates with the policy change.]"),
            msg("assistant", "The historical data shows a growth trend with an April correction."),
        ];

        let analysis = analyze(&messages, 0, 2);

        assert_eq!(analysis.problem_class, ProblemClass::Aggregation);
        assert!(analysis.stats.error_count >= 2);
        assert!(analysis.stats.preserved_count > 0, "should preserve high-influence messages");
        assert!(!analysis.text_for_summariser.is_empty());
    }
}
