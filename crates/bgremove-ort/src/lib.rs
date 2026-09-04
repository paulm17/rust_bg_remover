//! Verified ONNX Runtime adapter. Hash and graph validation happen before a
//! session is created; model/runtime downloads are deliberately disabled.
use anyhow::{bail, ensure, Context, Result};
use bgremove_models::{
    Activation, DimensionSpec, ModelManifest, OutputNormalization, TensorElementType,
};
use ort::{session::Session, tensor::TensorElementType as OrtType, value::Tensor};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    sync::{Arc, Condvar, Mutex, OnceLock},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RequestedProvider {
    Cpu,
    Coreml,
    Cuda,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderReport {
    pub requested: RequestedProvider,
    pub attempted_chain: Vec<String>,
    pub active: String,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub fallback_reason: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorOutput {
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInput {
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInspection {
    pub manifest_id: String,
    pub model_path: String,
    pub opset: u32,
    pub input_name: String,
    pub input_type: String,
    pub input_shape: Vec<i64>,
    pub input_symbols: Vec<String>,
    pub output_name: String,
    pub output_type: String,
    pub output_shape: Vec<i64>,
    pub output_symbols: Vec<String>,
    pub provider: ProviderReport,
}

static INIT: OnceLock<Result<(), String>> = OnceLock::new();
pub fn initialize(runtime: &Path) -> Result<()> {
    INIT.get_or_init(|| {
        ort::init_from(runtime)
            .map_err(|e| e.to_string())
            .and_then(|b| {
                if b.commit() {
                    Ok(())
                } else {
                    Err("ONNX Runtime environment was not committed".into())
                }
            })
    })
    .clone()
    .map_err(anyhow::Error::msg)
}
pub fn provider_report(
    requested: RequestedProvider,
    fallback_allowed: bool,
) -> Result<ProviderReport> {
    let name = match requested {
        RequestedProvider::Cpu => "CPUExecutionProvider",
        RequestedProvider::Coreml => "CoreMLExecutionProvider",
        RequestedProvider::Cuda => "CUDAExecutionProvider",
    };
    if requested == RequestedProvider::Cpu {
        return Ok(ProviderReport {
            requested,
            attempted_chain: vec![name.into()],
            active: name.into(),
            fallback_allowed,
            fallback_used: false,
            fallback_reason: None,
        });
    }
    let available = match requested {
        RequestedProvider::Coreml => cfg!(all(feature = "coreml", target_vendor = "apple")),
        RequestedProvider::Cuda => cfg!(feature = "cuda"),
        RequestedProvider::Cpu => true,
    };
    if available {
        bail!(
            "requested provider {name} is feature-enabled; use VerifiedSession::open to perform the real provider attempt"
        );
    }
    let reason = "provider is unavailable or feature-disabled and was not attempted";
    if !fallback_allowed {
        bail!("requested provider {name} unavailable: {reason}; strict mode forbids CPU fallback")
    }
    eprintln!("warning: {name} unavailable/unconfirmed; falling back to CPU: {reason}");
    Ok(ProviderReport {
        requested,
        attempted_chain: vec!["CPUExecutionProvider".into()],
        active: "CPUExecutionProvider".into(),
        fallback_allowed,
        fallback_used: true,
        fallback_reason: Some(reason.into()),
    })
}
fn dim_matches(spec: &DimensionSpec, actual: i64) -> bool {
    match spec {
        DimensionSpec::Static(n) => u64::try_from(actual).ok() == Some(*n),
        DimensionSpec::Dynamic(_) => actual == -1 || actual > 0,
    }
}
fn provider_name(provider: RequestedProvider) -> &'static str {
    match provider {
        RequestedProvider::Cpu => "CPUExecutionProvider",
        RequestedProvider::Coreml => "CoreMLExecutionProvider",
        RequestedProvider::Cuda => "CUDAExecutionProvider",
    }
}
fn provider_feature_enabled(provider: RequestedProvider) -> bool {
    match provider {
        RequestedProvider::Cpu => true,
        RequestedProvider::Coreml => cfg!(feature = "coreml"),
        RequestedProvider::Cuda => cfg!(feature = "cuda"),
    }
}
fn cpu_session(path: &Path) -> Result<Session> {
    Ok(Session::builder()?.commit_from_file(path)?)
}
fn requested_session(path: &Path, provider: RequestedProvider) -> Result<Session> {
    let builder = Session::builder()?.with_config_entry("session.disable_cpu_ep_fallback", "1")?;
    match provider {
        RequestedProvider::Cpu => Ok(builder.commit_from_file(path)?),
        RequestedProvider::Coreml => {
            #[cfg(feature = "coreml")]
            {
                Ok(builder
                    .with_execution_providers([ort::ep::CoreML::default()
                        .build()
                        .error_on_failure()])?
                    .commit_from_file(path)?)
            }
            #[cfg(not(feature = "coreml"))]
            {
                let _ = builder;
                bail!("CoreMLExecutionProvider feature is disabled; provider was not attempted")
            }
        }
        RequestedProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(builder
                    .with_execution_providers([ort::ep::CUDA::default()
                        .build()
                        .error_on_failure()])?
                    .commit_from_file(path)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = builder;
                bail!("CUDAExecutionProvider feature is disabled; provider was not attempted")
            }
        }
    }
}
fn ort_type(t: TensorElementType) -> OrtType {
    match t {
        TensorElementType::F32 => OrtType::Float32,
        TensorElementType::F16 => OrtType::Float16,
        TensorElementType::I32 => OrtType::Int32,
        TensorElementType::I64 => OrtType::Int64,
    }
}
fn checked_numel(shape: &[i64], label: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, dim| {
        ensure!(
            *dim > 0,
            "{label} shape contains non-positive dimension: {shape:?}"
        );
        let dimension = usize::try_from(*dim).map_err(|_| {
            anyhow::anyhow!("{label} shape dimension does not fit usize: {shape:?}")
        })?;
        count.checked_mul(dimension).ok_or_else(|| {
            anyhow::anyhow!("{label} shape element count overflows usize: {shape:?}")
        })
    })
}
pub fn validate_output_contract(
    model_id: &str,
    output_name: &str,
    expected: &[i64],
    actual: &[i64],
    values: usize,
) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "model {model_id} output {output_name} rank mismatch: expected {}, actual {}",
        expected.len(),
        actual.len()
    );
    for (want, got) in expected.iter().zip(actual) {
        ensure!(*got > 0 && (*want < 0 || want == got), "model {model_id} output {output_name} shape mismatch: expected {expected:?}, actual {actual:?}");
    }
    let expected_values = checked_numel(actual, "output")?;
    ensure!(values == expected_values, "model {model_id} output {output_name} shape/value mismatch: shape {actual:?} implies {expected_values}, actual {values}");
    Ok(())
}
fn read_varint(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    while *pos < bytes.len() && shift < 64 {
        let b = bytes[*pos];
        *pos += 1;
        value |= ((b & 127) as u64) << shift;
        if b & 128 == 0 {
            return Some(value);
        }
        shift += 7;
    }
    None
}
fn declared_opset(bytes: &[u8]) -> Option<u32> {
    let mut p = 0;
    let mut default = None;
    while p < bytes.len() {
        let tag = read_varint(bytes, &mut p)?;
        let wire = tag & 7;
        if wire == 2 {
            let len = read_varint(bytes, &mut p)? as usize;
            let end = p.checked_add(len)?;
            if end > bytes.len() {
                return None;
            }
            if tag >> 3 == 8 {
                let mut q = p;
                let mut domain = String::new();
                let mut version = None;
                while q < end {
                    let t = read_varint(bytes, &mut q)?;
                    if t >> 3 == 1 && t & 7 == 2 {
                        let l = read_varint(bytes, &mut q)? as usize;
                        let finish = q.checked_add(l)?;
                        domain = std::str::from_utf8(bytes.get(q..finish)?).ok()?.to_owned();
                        q = finish;
                    } else if t >> 3 == 2 && t & 7 == 0 {
                        version = Some(read_varint(bytes, &mut q)? as u32);
                    } else if t & 7 == 2 {
                        let l = read_varint(bytes, &mut q)? as usize;
                        q = q.checked_add(l)?;
                        if q > end {
                            return None;
                        }
                    } else if t & 7 == 0 {
                        let _ = read_varint(bytes, &mut q)?;
                    } else if t & 7 == 1 {
                        q = q.checked_add(8)?;
                    } else if t & 7 == 5 {
                        q = q.checked_add(4)?;
                    } else {
                        return None;
                    }
                }
                let version = version?;
                if domain.is_empty() || domain == "ai.onnx" {
                    default = Some(version);
                }
            }
            p = end;
        } else if wire == 0 {
            let _ = read_varint(bytes, &mut p)?;
        } else {
            return None;
        }
    }
    default
}
fn inspect_session(
    session: &Session,
    m: &ModelManifest,
    path: &Path,
    provider: ProviderReport,
) -> Result<ModelInspection> {
    let actual_inputs = session
        .inputs()
        .iter()
        .map(|x| x.name())
        .collect::<Vec<_>>();
    let input = session
        .inputs()
        .iter()
        .find(|x| x.name() == m.input_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "model {} input name mismatch: expected {}, actual {:?}",
                m.id,
                m.input_name,
                actual_inputs
            )
        })?;
    let actual_outputs = session
        .outputs()
        .iter()
        .map(|x| x.name())
        .collect::<Vec<_>>();
    let output = session
        .outputs()
        .iter()
        .find(|x| x.name() == m.output_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "model {} output name mismatch: expected {}, actual {:?}",
                m.id,
                m.output_name,
                actual_outputs
            )
        })?;
    let (itype, ishape, isymbols) = match input.dtype() {
        ort::value::ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => (
            *ty,
            shape.to_vec(),
            dimension_symbols.iter().cloned().collect(),
        ),
        other => bail!(
            "model {} input {} expected tensor, actual {}",
            m.id,
            m.input_name,
            other
        ),
    };
    let (otype, oshape, osymbols) = match output.dtype() {
        ort::value::ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => (
            *ty,
            shape.to_vec(),
            dimension_symbols.iter().cloned().collect(),
        ),
        other => bail!(
            "model {} output {} expected tensor, actual {}",
            m.id,
            m.output_name,
            other
        ),
    };
    if let Some(t) = m.input_type {
        ensure!(
            itype == ort_type(t),
            "model {} input {} element type mismatch: expected {}, actual {}",
            m.id,
            m.input_name,
            ort_type(t),
            itype
        );
    }
    if let Some(t) = m.output_type {
        ensure!(
            otype == ort_type(t),
            "model {} output {} element type mismatch: expected {}, actual {}",
            m.id,
            m.output_name,
            ort_type(t),
            otype
        );
    }
    if !m.input_shape.is_empty() {
        ensure!(
            m.input_shape.len() == ishape.len(),
            "model {} input {} rank mismatch: expected {}, actual {}",
            m.id,
            m.input_name,
            m.input_shape.len(),
            ishape.len()
        );
        for (s, a) in m.input_shape.iter().zip(&ishape) {
            ensure!(
                dim_matches(s, *a),
                "model {} input {} shape mismatch: expected {:?}, actual {:?}",
                m.id,
                m.input_name,
                m.input_shape,
                ishape
            );
        }
    }
    if !m.output_shape.is_empty() {
        ensure!(
            m.output_shape.len() == oshape.len(),
            "model {} output {} rank mismatch: expected {}, actual {}",
            m.id,
            m.output_name,
            m.output_shape.len(),
            oshape.len()
        );
        for (s, a) in m.output_shape.iter().zip(&oshape) {
            ensure!(
                dim_matches(s, *a),
                "model {} output {} shape mismatch: expected {:?}, actual {:?}",
                m.id,
                m.output_name,
                m.output_shape,
                oshape
            );
        }
    }
    Ok(ModelInspection {
        manifest_id: m.id.clone(),
        model_path: path.display().to_string(),
        opset: m.opset,
        input_name: m.input_name.clone(),
        input_type: itype.to_string(),
        input_shape: ishape,
        input_symbols: isymbols,
        output_name: m.output_name.clone(),
        output_type: otype.to_string(),
        output_shape: oshape,
        output_symbols: osymbols,
        provider,
    })
}

/// Convert canonical encoded RGB into a manifest-shaped f32 tensor. Scale is
/// applied before mean/std; layout and channel order are explicit.
pub fn preprocess_rgb(
    image: &bgremove_core::CanonicalImage,
    m: &ModelManifest,
) -> Result<TensorInput> {
    let (w, h) = image.dimensions();
    let channels = 3i64;
    let mut values = Vec::with_capacity((w * h * 3) as usize);
    let mut channel_data = (0..3)
        .map(|_| Vec::with_capacity((w * h) as usize))
        .collect::<Vec<_>>();
    for px in image.rgb().data() {
        let rgb = match m.channel_order {
            bgremove_models::ChannelOrder::Rgb => *px,
            bgremove_models::ChannelOrder::Bgr => [px[2], px[1], px[0]],
        };
        for c in 0..3 {
            let v = (rgb[c] * m.scale - m.mean[c]) / m.std[c];
            ensure!(
                v.is_finite(),
                "model {} preprocessing produced NaN/Inf",
                m.id
            );
            channel_data[c].push(v);
        }
    }
    let tensor = match m.layout {
        bgremove_models::ModelLayout::Nchw => {
            values.extend(channel_data.into_iter().flatten());
            TensorInput {
                shape: vec![1, channels, h as i64, w as i64],
                values,
            }
        }
        bgremove_models::ModelLayout::Nhwc => {
            for (i, _) in channel_data[0].iter().enumerate().take((w * h) as usize) {
                for channel in channel_data.iter().take(3) {
                    values.push(channel[i]);
                }
            }
            TensorInput {
                shape: vec![1, h as i64, w as i64, channels],
                values,
            }
        }
    };
    if !m.input_shape.is_empty() {
        ensure!(
            m.input_shape.len() == tensor.shape.len(),
            "model {} preprocessing rank mismatch: expected {}, actual {}",
            m.id,
            m.input_shape.len(),
            tensor.shape.len()
        );
        for (expected, actual) in m.input_shape.iter().zip(&tensor.shape) {
            ensure!(
                dim_matches(expected, *actual),
                "model {} preprocessing shape mismatch: expected {:?}, actual {:?}",
                m.id,
                m.input_shape,
                tensor.shape
            );
        }
    }
    Ok(tensor)
}

impl TensorOutput {
    /// Map common single-channel output layouts to the core AlphaMask.
    pub fn to_alpha_mask(&self) -> Result<bgremove_core::AlphaMask> {
        ensure!(
            self.values.iter().all(|x| x.is_finite()),
            "output tensor contains NaN/Inf"
        );
        let (w, h) = match self.shape.as_slice() {
            [1, 1, h, w] => (*w, *h),
            [1, h, w, 1] => (*w, *h),
            [h, w] => (*w, *h),
            [1, h, w] => (*w, *h),
            other => bail!("output shape {:?} cannot map to AlphaMask; expected [1,1,H,W], [1,H,W,1], [H,W], or [1,H,W]", other),
        };
        let width =
            u32::try_from(w).map_err(|_| anyhow::anyhow!("output width {w} does not fit u32"))?;
        let height =
            u32::try_from(h).map_err(|_| anyhow::anyhow!("output height {h} does not fit u32"))?;
        let expected = checked_numel(&[h, w], "alpha output")?;
        ensure!(
            self.values.len() == expected,
            "output shape {:?} has {} values, expected {}",
            self.shape,
            self.values.len(),
            expected
        );
        bgremove_core::AlphaMask::new(
            width,
            height,
            self.values.iter().map(|v| v.clamp(0.0, 1.0)).collect(),
        )
    }
}

pub struct VerifiedSession {
    session: Session,
    pub inspection: ModelInspection,
    manifest: ModelManifest,
    run_count: u64,
}
impl VerifiedSession {
    pub fn open(
        m: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        let path = m
            .verify_model_hash(manifest_path)
            .context("verify model hash before session creation")?;
        let model_bytes = std::fs::read(&path)?;
        let actual_opset = declared_opset(&model_bytes)
            .ok_or_else(|| anyhow::anyhow!("model {} opset metadata is missing", m.id))?;
        ensure!(
            actual_opset == m.opset,
            "model {} opset mismatch: expected {}, actual {}",
            m.id,
            m.opset,
            actual_opset
        );
        initialize(runtime)?;
        let (session, provider) = if requested == RequestedProvider::Cpu {
            (
                cpu_session(&path)
                    .with_context(|| format!("create CPU ONNX session for {}", m.id))?,
                ProviderReport {
                    requested,
                    attempted_chain: vec![provider_name(requested).into()],
                    active: provider_name(requested).into(),
                    fallback_allowed,
                    fallback_used: false,
                    fallback_reason: None,
                },
            )
        } else {
            if !provider_feature_enabled(requested) {
                let reason = format!(
                    "{} feature is disabled; provider was not attempted",
                    provider_name(requested)
                );
                if !fallback_allowed {
                    bail!(
                        "requested provider {} unavailable: {}; strict mode forbids CPU fallback",
                        provider_name(requested),
                        reason
                    );
                }
                eprintln!("warning: {reason}; falling back to CPU");
                let session = cpu_session(&path)
                    .with_context(|| format!("create CPU fallback session for {}", m.id))?;
                (
                    session,
                    ProviderReport {
                        requested,
                        attempted_chain: vec!["CPUExecutionProvider".into()],
                        active: "CPUExecutionProvider".into(),
                        fallback_allowed,
                        fallback_used: true,
                        fallback_reason: Some(reason),
                    },
                )
            } else {
                match requested_session(&path, requested) {
                Ok(session) => (session, ProviderReport { requested, attempted_chain: vec![provider_name(requested).into()], active: provider_name(requested).into(), fallback_allowed, fallback_used: false, fallback_reason: None }),
                Err(error) if !fallback_allowed => bail!("requested provider {} registration/session creation failed: {}; strict mode forbids CPU fallback", provider_name(requested), error),
                Err(error) => {
                    let reason = format!("{} registration/session creation failed: {error}", provider_name(requested));
                    eprintln!("warning: {reason}; falling back to CPU");
                    let session = cpu_session(&path).with_context(|| format!("create CPU fallback session for {}", m.id))?;
                    (session, ProviderReport { requested, attempted_chain: vec![provider_name(requested).into(), "CPUExecutionProvider".into()], active: "CPUExecutionProvider".into(), fallback_allowed, fallback_used: true, fallback_reason: Some(reason) })
                }
            }
            }
        };
        let inspection = inspect_session(&session, m, &path, provider)?;
        if let Some(index) = m.output_index {
            ensure!(
                session.outputs().get(index).map(|x| x.name()) == Some(m.output_name.as_str()),
                "model {} output index mismatch: expected index {} to name {}, actual {:?}",
                m.id,
                index,
                m.output_name,
                session
                    .outputs()
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<_>>()
            );
        }
        Ok(Self {
            session,
            inspection,
            manifest: m.clone(),
            run_count: 0,
        })
    }
    pub fn run(&mut self, shape: &[i64], values: &[f32]) -> Result<TensorOutput> {
        ensure!(
            shape.len() == self.inspection.input_shape.len(),
            "model {} input rank mismatch: expected {}, actual {}",
            self.manifest.id,
            self.inspection.input_shape.len(),
            shape.len()
        );
        for (actual, expected) in shape.iter().zip(&self.inspection.input_shape) {
            ensure!(
                *actual > 0 && (*expected < 0 || actual == expected),
                "model {} runtime input shape mismatch: expected {:?}, actual {:?}",
                self.manifest.id,
                self.inspection.input_shape,
                shape
            );
        }
        let expected_values = checked_numel(shape, "input")?;
        ensure!(
            values.len() == expected_values,
            "model {} input value count mismatch: expected {}, actual {}",
            self.manifest.id,
            expected_values,
            values.len()
        );
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "model {} input contains NaN/Inf",
            self.manifest.id
        );
        let input = Tensor::from_array((shape.to_vec(), values.to_vec()))?;
        let outputs = self
            .session
            .run(ort::inputs! { self.manifest.input_name.as_str() => input })?;
        self.run_count += 1;
        let out = outputs.get(&self.manifest.output_name).ok_or_else(|| {
            anyhow::anyhow!(
                "model {} declared output {} not returned",
                self.manifest.id,
                self.manifest.output_name
            )
        })?;
        let (shape, data) = out.try_extract_tensor::<f32>().with_context(|| {
            format!(
                "model {} output {} must be f32",
                self.manifest.id, self.manifest.output_name
            )
        })?;
        ensure!(
            data.iter().all(|x| x.is_finite()),
            "model {} output {} contains NaN/Inf",
            self.manifest.id,
            self.manifest.output_name
        );
        validate_output_contract(
            &self.manifest.id,
            &self.manifest.output_name,
            &self.inspection.output_shape,
            shape,
            data.len(),
        )?;
        Ok(TensorOutput {
            shape: shape.to_vec(),
            values: data.to_vec(),
        })
    }
    pub fn run_count(&self) -> u64 {
        self.run_count
    }
}
struct PoolState {
    available: Vec<VerifiedSession>,
}
pub struct SessionPool {
    state: Arc<(Mutex<PoolState>, Condvar)>,
    size: usize,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}
pub struct SessionLease {
    state: Arc<(Mutex<PoolState>, Condvar)>,
    session: Option<VerifiedSession>,
    active: Arc<AtomicUsize>,
}
impl SessionPool {
    pub fn new(
        m: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        ensure!(workers > 0, "workers must be greater than zero");
        let mut available = Vec::with_capacity(workers);
        for _ in 0..workers {
            available.push(VerifiedSession::open(
                m,
                manifest_path,
                runtime,
                requested,
                fallback_allowed,
            )?);
        }
        Ok(Self {
            state: Arc::new((Mutex::new(PoolState { available }), Condvar::new())),
            size: workers,
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
        })
    }
    pub fn checkout(&self) -> SessionLease {
        let (lock, cv) = &*self.state;
        let mut state = lock.lock().unwrap();
        while state.available.is_empty() {
            state = cv.wait(state).unwrap();
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        SessionLease {
            state: Arc::clone(&self.state),
            session: Some(state.available.pop().unwrap()),
            active: Arc::clone(&self.active),
        }
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }
}
impl SessionLease {
    pub fn session_mut(&mut self) -> &mut VerifiedSession {
        self.session.as_mut().unwrap()
    }
}
impl Drop for SessionLease {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
        if let Some(s) = self.session.take() {
            let (lock, cv) = &*self.state;
            lock.lock().unwrap().available.push(s);
            cv.notify_one();
        }
    }
}

pub fn apply_output_transform(
    mut t: TensorOutput,
    activation: Activation,
    normalization: OutputNormalization,
) -> Result<TensorOutput> {
    ensure!(
        t.values.iter().all(|x| x.is_finite()),
        "output tensor contains NaN/Inf"
    );
    if activation == Activation::Sigmoid {
        for x in &mut t.values {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
    }
    match normalization {
        OutputNormalization::None => {}
        OutputNormalization::Clamp => t.values.iter_mut().for_each(|x| *x = x.clamp(0.0, 1.0)),
        OutputNormalization::MinMax => {
            let (min, max) = t
                .values
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| {
                    (a.min(*x), b.max(*x))
                });
            if max == min {
                t.values.fill(0.0);
            } else {
                for x in &mut t.values {
                    *x = ((*x - min) / (max - min)).clamp(0.0, 1.0);
                }
            }
        }
    }
    ensure!(
        t.values.iter().all(|x| x.is_finite()),
        "transformed output tensor contains NaN/Inf"
    );
    Ok(t)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_minmax_is_finite() {
        let t = apply_output_transform(
            TensorOutput {
                shape: vec![1, 1],
                values: vec![2.0, 2.0],
            },
            Activation::None,
            OutputNormalization::MinMax,
        )
        .unwrap();
        assert_eq!(t.values, vec![0.0, 0.0]);
    }

    #[test]
    fn hash_failure_precedes_runtime_initialization() {
        let path = Path::new("../../models/m3_identity.toml");
        let text = std::fs::read_to_string(path).unwrap();
        let mut manifest = bgremove_models::parse_toml(&text).unwrap();
        manifest.sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        let result = VerifiedSession::open(
            &manifest,
            path,
            Path::new("/does/not/exist"),
            RequestedProvider::Cpu,
            false,
        );
        let error = match result {
            Ok(_) => panic!("hash mismatch must fail before runtime"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("hash mismatch"));
    }

    #[test]
    fn preprocessing_honours_rgb_bgr_and_layout() {
        let mut manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m3_identity.toml").unwrap(),
        )
        .unwrap();
        let image =
            bgremove_core::CanonicalImage::new(2, 1, vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
                .unwrap();
        let nchw = preprocess_rgb(&image, &manifest).unwrap();
        assert_eq!(nchw.shape, vec![1, 3, 1, 2]);
        assert_eq!(nchw.values, vec![0.1, 0.4, 0.2, 0.5, 0.3, 0.6]);
        manifest.channel_order = bgremove_models::ChannelOrder::Bgr;
        manifest.layout = bgremove_models::ModelLayout::Nhwc;
        let nhwc = preprocess_rgb(&image, &manifest).unwrap();
        assert_eq!(nhwc.shape, vec![1, 1, 2, 3]);
        assert_eq!(nhwc.values, vec![0.3, 0.2, 0.1, 0.6, 0.5, 0.4]);
        manifest.channel_order = bgremove_models::ChannelOrder::Rgb;
        manifest.layout = bgremove_models::ModelLayout::Nchw;
        manifest.scale = 0.5;
        manifest.mean = [0.1, 0.2, 0.3];
        manifest.std = [0.5, 0.5, 0.5];
        let normalized = preprocess_rgb(&image, &manifest).unwrap();
        let expected = [-0.1, 0.2, -0.2, 0.1, -0.3, 0.0];
        assert!(normalized
            .values
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-6));
    }

    #[test]
    fn transforms_cover_sigmoid_clamp_and_nonfinite() {
        let sigmoid = apply_output_transform(
            TensorOutput {
                shape: vec![1],
                values: vec![0.0],
            },
            Activation::Sigmoid,
            OutputNormalization::None,
        )
        .unwrap();
        assert!((sigmoid.values[0] - 0.5).abs() < f32::EPSILON);
        let clamped = apply_output_transform(
            TensorOutput {
                shape: vec![2],
                values: vec![-2.0, 2.0],
            },
            Activation::None,
            OutputNormalization::Clamp,
        )
        .unwrap();
        assert_eq!(clamped.values, vec![0.0, 1.0]);
        assert!(apply_output_transform(
            TensorOutput {
                shape: vec![1],
                values: vec![f32::NAN]
            },
            Activation::None,
            OutputNormalization::None
        )
        .is_err());
    }

    #[test]
    fn alpha_mask_layouts_are_checked() {
        for shape in [
            vec![1, 1, 2, 2],
            vec![1, 2, 2, 1],
            vec![2, 2],
            vec![1, 2, 2],
        ] {
            assert!(TensorOutput {
                shape,
                values: vec![0.0; 4]
            }
            .to_alpha_mask()
            .is_ok());
        }
        assert!(TensorOutput {
            shape: vec![1, 3, 2, 2],
            values: vec![0.0; 12]
        }
        .to_alpha_mask()
        .is_err());
        assert!(TensorOutput {
            shape: vec![2, 2],
            values: vec![0.0; 3]
        }
        .to_alpha_mask()
        .is_err());
    }

    #[test]
    fn output_contract_rejects_rank_and_static_dimension_mismatch() {
        assert!(validate_output_contract("m", "mask", &[1, 1, -1, 4], &[1, 1, 2], 2).is_err());
        assert!(validate_output_contract("m", "mask", &[1, 1, -1, 4], &[1, 1, 3, 4], 12).is_ok());
        assert!(validate_output_contract("m", "mask", &[1, 1, 2, 4], &[1, 1, 3, 4], 12).is_err());
    }

    #[test]
    fn provider_reports_are_truthful_and_strict() {
        let cpu = provider_report(RequestedProvider::Cpu, false).unwrap();
        assert_eq!(cpu.attempted_chain, vec!["CPUExecutionProvider"]);
        assert!(!cpu.fallback_used);
        #[cfg(not(feature = "cuda"))]
        {
            let fallback = provider_report(RequestedProvider::Cuda, true).unwrap();
            assert_eq!(fallback.active, "CPUExecutionProvider");
            assert!(fallback.fallback_used);
            assert_eq!(fallback.attempted_chain, vec!["CPUExecutionProvider"]);
            assert!(provider_report(RequestedProvider::Cuda, false).is_err());
        }
        #[cfg(feature = "cuda")]
        assert!(provider_report(RequestedProvider::Cuda, true).is_err());
    }

    #[test]
    fn zero_worker_pool_is_rejected_before_runtime() {
        let path = Path::new("../../models/m3_identity.toml");
        let m = bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        let result = SessionPool::new(
            &m,
            path,
            Path::new("/missing"),
            0,
            RequestedProvider::Cpu,
            false,
        );
        let error = match result {
            Ok(_) => panic!("zero workers must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("workers"));
    }

    #[test]
    #[ignore = "requires ORT_DYLIB and runs the checked-in fixture"]
    fn real_pool_reuses_sessions_and_bounds_concurrency() {
        let path = Path::new("../../models/m3_identity.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        let runtime_os = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let runtime = Path::new(&runtime_os);
        let pool = Arc::new(
            SessionPool::new(&manifest, path, runtime, 2, RequestedProvider::Cpu, false).unwrap(),
        );
        {
            let mut lease = pool.checkout();
            let output = lease
                .session_mut()
                .run(&[1, 1, 1, 2], &[0.25, 0.75])
                .unwrap();
            assert_eq!(output.values, vec![0.25, 0.75]);
        }
        {
            let lease = pool.checkout();
            assert!(lease.session.as_ref().unwrap().run_count() >= 1);
        }
        let mut joins = Vec::new();
        for _ in 0..4 {
            let shared = Arc::clone(&pool);
            joins.push(std::thread::spawn(move || {
                let mut lease = shared.checkout();
                lease
                    .session_mut()
                    .run(&[1, 1, 1, 2], &[0.25, 0.75])
                    .unwrap();
                std::thread::sleep(std::time::Duration::from_millis(10));
            }));
        }
        for join in joins {
            join.join().unwrap();
        }
        assert_eq!(pool.size(), 2);
        assert!(pool.max_active() <= 2);
        assert!(pool.max_active() >= 2);
    }

    #[test]
    #[ignore = "requires ORT_DYLIB and runs the checked-in fixture"]
    fn real_fixture_runs_two_dynamic_geometries_and_selects_declared_output() {
        let path = Path::new("../../models/m3_identity.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        let runtime_os = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let mut session = VerifiedSession::open(
            &manifest,
            path,
            Path::new(&runtime_os),
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        for shape in [[1, 3, 2, 3], [1, 3, 3, 5]] {
            let count = checked_numel(&shape, "test input").unwrap();
            let values = (0..count)
                .map(|i| i as f32 / count as f32)
                .collect::<Vec<_>>();
            let output = session.run(&shape, &values).unwrap();
            assert_eq!(output.shape, shape);
            assert_eq!(output.values, values);
        }
        assert_eq!(session.run_count(), 2);
    }

    #[test]
    fn traversal_and_intended_use_are_rejected() {
        let path = Path::new("../../models/m3_identity.toml");
        let mut m = bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        m.file = "../Cargo.toml".into();
        assert!(m.verify_model_hash(path).is_err());
        let mut approved =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        approved.intended_use_approved = false;
        assert!(approved.verify_model_hash(path).is_err());
    }

    #[test]
    fn malformed_opset_bytes_fail_closed() {
        assert_eq!(declared_opset(&[]), None);
        assert_eq!(declared_opset(&[0x42, 0x80]), None);
        assert_eq!(declared_opset(&[0x42, 0x01, 0x80]), None);
    }

    #[test]
    fn opset_reader_prefers_default_domain_over_vendor_order() {
        // OperatorSetIdProto: domain (field 1), version (field 2).
        let vendor = [
            0x42, 0x0a, 0x0a, 0x06, b'v', b'e', b'n', b'd', b'o', b'r', 0x10, 0x07,
        ];
        let default = [0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d];
        let mut model = vendor.to_vec();
        model.extend(default);
        assert_eq!(declared_opset(&model), Some(13));
        assert_eq!(declared_opset(&vendor), None);
    }
}
