pub mod forgetting;
pub mod benchmark;

/// Minimal message type for the benchmark.
/// In production (B.app), this is extended with image support.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}
