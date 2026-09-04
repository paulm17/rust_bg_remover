//! ONNX Runtime boundary reserved for M3.
//!
//! M1 intentionally exposes metadata only and never downloads or executes a
//! checkpoint. Keeping this crate separate prevents model/runtime assumptions
//! from leaking into the typed core.

/// Provider-independent runtime status for inspection and future adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeStatus {
    DeferredUntilM3,
}

pub const RUNTIME_STATUS: RuntimeStatus = RuntimeStatus::DeferredUntilM3;
