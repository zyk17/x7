//! Engine 生命周期的 UCI option。
//!
//! `go` 时拍快照：算法 → `SearchParams`，当前 worker 配置 → `SearchConfig`，停止条件/`searchmoves` → `SearchLimits`。

use crate::neural::cache::{DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO, MAX_NN_CACHE_SIZE_POWER_OF_TWO};
use crate::search::{DecisionRule, SearchConfig, SearchParams};

/// 正式 UCI 的最小 option 集。
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub weights_file: String,
    pub show_wdl: bool,
    pub show_eps: bool,
    /// UCI `MultiPV`：每次 info 最多输出的根变化数。
    pub multi_pv: usize,
    /// UCI `NnBatchSize`：单次 NN 调用最多合并的局面数；0 使用 backend 建议值。
    pub nn_batch_size: usize,
    /// UCI `NnCacheSizePowerOfTwo`：NN cache 固定保存 `2^N` 个直映槽。
    pub nn_cache_size_power_of_two: u8,
    /// UCI `CPuct`：所有节点共用的 PUCT 初始探索系数。
    pub cpuct: f32,
    /// UCI `CPuctBase`/`CPuctFactor`：PUCT 随访问数增长的形状。
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    /// UCI `FpuReduction`：未访问边相对 parent Q 的 FPU 降幅。
    pub fpu_reduction: f32,
    /// UCI `VarianceBonusScale`：已观察 edge 的 `scale * SE` 复核 bonus。
    pub variance_bonus_scale: f32,
    /// Eval claim 上限相对 batch 的倍率；控制 pending work 的新鲜度与 NN 供给。
    pub nn_window: f32,
    /// reservation 临时写入的 FPU 缩放；仅在 in-flight 时影响 action Q。
    pub virtual_mean_fpu_scale: f32,
    /// UCI 根决策 `Lcb` / `Ucb` 的 SE 半径倍数。
    pub decision_lcb_stdevs: f32,
    pub decision_ucb_stdevs: f32,
    pub decision_rule: DecisionRule,
    pub decision_mix_n_weight: f32,
    /// UCI `Threads` 只分配 Gather + Eval；Backprop 和 NN 各固定一条线程。
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        let search = SearchParams::default();
        Self {
            weights_file: String::new(),
            show_wdl: false,
            show_eps: false,
            multi_pv: 1,
            nn_batch_size: 0,
            nn_cache_size_power_of_two: DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO,
            cpuct: search.cpuct,
            cpuct_base: search.cpuct_base,
            cpuct_factor: search.cpuct_factor,
            fpu_reduction: search.fpu_reduction,
            variance_bonus_scale: search.variance_bonus_scale,
            nn_window: SearchConfig::default().nn_window,
            virtual_mean_fpu_scale: search.virtual_mean_fpu_scale,
            decision_lcb_stdevs: search.decision_lcb_stdevs,
            decision_ucb_stdevs: search.decision_ucb_stdevs,
            decision_rule: search.decision_rule,
            decision_mix_n_weight: search.decision_mix_n_weight,
            threads: 8,
        }
    }
}

impl Options {
    pub fn list_options_uci(&self) -> Vec<String> {
        vec![
            format!("option name Threads type spin default {} min 2 max 128", self.threads),
            format!(
                "option name NnCacheSizePowerOfTwo type spin default {} min 0 max {}",
                self.nn_cache_size_power_of_two, MAX_NN_CACHE_SIZE_POWER_OF_TWO
            ),
            format!("option name MultiPV type spin default {} min 1 max 500", self.multi_pv),
            format!(
                "option name NnBatchSize type spin default {} min 0 max 1024",
                self.nn_batch_size
            ),
            format!("option name CPuct type string default {}", self.cpuct),
            format!("option name CPuctBase type string default {}", self.cpuct_base),
            format!("option name CPuctFactor type string default {}", self.cpuct_factor),
            format!("option name FpuReduction type string default {}", self.fpu_reduction),
            format!(
                "option name VarianceBonusScale type string default {}",
                self.variance_bonus_scale
            ),
            format!("option name NnWindow type string default {}", self.nn_window),
            format!(
                "option name VirtualMeanFpuScale type string default {}",
                self.virtual_mean_fpu_scale
            ),
            format!(
                "option name DecisionRule type combo default {} var Auto var MaxQ var MaxN var Lcb var Ucb var MixNQ",
                self.decision_rule.uci_name()
            ),
            format!(
                "option name DecisionLcbStdevs type string default {}",
                self.decision_lcb_stdevs
            ),
            format!(
                "option name DecisionUcbStdevs type string default {}",
                self.decision_ucb_stdevs
            ),
            format!(
                "option name DecisionMixNWeight type string default {}",
                self.decision_mix_n_weight
            ),
            format!("option name UCI_ShowWDL type check default {}", self.show_wdl),
            format!("option name UCI_ShowEPS type check default {}", self.show_eps),
            format!("option name WeightsFile type string default {}", self.weights_file),
        ]
    }

    pub fn set_uci_option(&mut self, name: &str, value: &str) -> Result<(), crate::EnginError> {
        let flag = |name| match value {
            value if value.eq_ignore_ascii_case("true") => Ok(true),
            value if value.eq_ignore_ascii_case("false") => Ok(false),
            _ => Err(crate::EnginError::Uci(format!(
                "Flag '{name}' must be either true or false"
            ))),
        };
        let option_name = name.to_ascii_lowercase();
        match option_name.as_str() {
            "uci_showwdl" => self.show_wdl = flag(name)?,
            "uci_showeps" => self.show_eps = flag(name)?,
            "multipv" => {
                let value = value
                    .parse::<usize>()
                    .map_err(|_| crate::EnginError::Uci("MultiPV must be an integer within [1, 500]".into()))?;
                if !(1..=500).contains(&value) {
                    return Err(crate::EnginError::Uci("MultiPV must be within [1, 500]".into()));
                }
                self.multi_pv = value;
            }
            "weightsfile" => self.weights_file = value.to_owned(),
            "nnbatchsize" => {
                let value = value
                    .parse::<usize>()
                    .map_err(|_| crate::EnginError::Uci("NnBatchSize must be an integer within [0, 1024]".into()))?;
                if value > 1024 {
                    return Err(crate::EnginError::Uci("NnBatchSize must be within [0, 1024]".into()));
                }
                self.nn_batch_size = value;
            }
            "nncachesizepoweroftwo" => {
                let value = value.parse::<u8>().map_err(|_| {
                    crate::EnginError::Uci(format!(
                        "NnCacheSizePowerOfTwo must be an integer within [0, {MAX_NN_CACHE_SIZE_POWER_OF_TWO}]"
                    ))
                })?;
                if value > MAX_NN_CACHE_SIZE_POWER_OF_TWO {
                    return Err(crate::EnginError::Uci(format!(
                        "NnCacheSizePowerOfTwo must be within [0, {MAX_NN_CACHE_SIZE_POWER_OF_TWO}]"
                    )));
                }
                self.nn_cache_size_power_of_two = value;
            }
            "cpuctbase" => self.cpuct_base = parse_positive_float("CPuctBase", value)?,
            "cpuctfactor" => self.cpuct_factor = parse_non_negative_float("CPuctFactor", value)?,
            "cpuct" => self.cpuct = parse_non_negative_float("CPuct", value)?,
            "fpureduction" => self.fpu_reduction = parse_non_negative_float("FpuReduction", value)?,
            "variancebonusscale" => self.variance_bonus_scale = parse_non_negative_float("VarianceBonusScale", value)?,
            "nnwindow" => self.nn_window = parse_positive_float("NnWindow", value)?,
            "virtualmeanfpuscale" => {
                self.virtual_mean_fpu_scale = parse_non_negative_float("VirtualMeanFpuScale", value)?
            }
            "decisionlcbstdevs" => self.decision_lcb_stdevs = parse_non_negative_float("DecisionLcbStdevs", value)?,
            "decisionucbstdevs" => self.decision_ucb_stdevs = parse_non_negative_float("DecisionUcbStdevs", value)?,
            "decisionrule" => {
                self.decision_rule = DecisionRule::parse_uci(value).ok_or_else(|| {
                    crate::EnginError::Uci("DecisionRule must be one of Auto, MaxQ, MaxN, Lcb, Ucb, MixNQ".into())
                })?
            }
            "decisionmixnweight" => self.decision_mix_n_weight = parse_non_negative_float("DecisionMixNWeight", value)?,
            "threads" => self.threads = parse_thread_count(value)?,
            _ => return Err(crate::EnginError::Uci(format!("Unknown option: {name}"))),
        }
        Ok(())
    }
}

fn parse_non_negative_float(name: &str, value: &str) -> Result<f32, crate::EnginError> {
    let value = value
        .parse::<f32>()
        .map_err(|_| crate::EnginError::Uci(format!("{name} must be a finite non-negative number")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(crate::EnginError::Uci(format!(
            "{name} must be a finite non-negative number"
        )));
    }
    Ok(value)
}

fn parse_positive_float(name: &str, value: &str) -> Result<f32, crate::EnginError> {
    let value = parse_non_negative_float(name, value)?;
    if value == 0.0 {
        return Err(crate::EnginError::Uci(format!("{name} must be finite and positive")));
    }
    Ok(value)
}

fn parse_thread_count(value: &str) -> Result<usize, crate::EnginError> {
    let value = value
        .parse::<usize>()
        .map_err(|_| crate::EnginError::Uci("Threads must be an integer no greater than 128".into()))?;
    if value > 128 {
        return Err(crate::EnginError::Uci("Threads must be no greater than 128".into()));
    }
    Ok(value.max(2))
}
