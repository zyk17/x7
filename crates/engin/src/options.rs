//! Engine 生命周期的 UCI option。
//!
//! `Engine` 持有 option，搜索在启动 job 时读取快照。option 名称与常见 UCI/引擎
//! 习惯对齐；不是 px0 `OptionsDict` 的翻译层。

use crate::neural::cache::{DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO, MAX_NN_CACHE_SIZE_POWER_OF_TWO};
use crate::search::SearchParams;

/// 正式 UCI 的最小 option 集。
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub weights_file: String,
    pub show_wdl: bool,
    pub show_eps: bool,
    /// UCI `MultiPV`：每次 info 最多输出的根变化数。
    pub multi_pv: usize,
    /// UCI `MiniBatchSize`：单次 NN 调用最多合并的局面数；0 使用 backend 建议值。
    pub mini_batch_size: usize,
    /// UCI `NNCacheSizePowerOfTwo`：NN cache 固定保存 `2^N` 个直映槽。
    pub nn_cache_size_power_of_two: u8,
    /// UCI `CPuct`：所有节点共用的 PUCT 初始探索系数。
    pub cpuct: f32,
    /// UCI `CPuctBase`/`CPuctFactor`：PUCT 随访问数增长的形状。
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    /// UCI `FpuReduction`：未访问边相对 parent Q 的 FPU 降幅。
    pub fpu_reduction: f32,
    /// 根最终 Decision 的 LCB 参数；不参与 PUCT。
    pub lcb_stdevs: f32,
    pub lcb_min_visit_fraction: f32,
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
            mini_batch_size: 0,
            nn_cache_size_power_of_two: DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO,
            cpuct: search.cpuct,
            cpuct_base: search.cpuct_base,
            cpuct_factor: search.cpuct_factor,
            fpu_reduction: search.fpu_reduction,
            lcb_stdevs: search.lcb_stdevs,
            lcb_min_visit_fraction: search.lcb_min_visit_fraction,
            threads: 8,
        }
    }
}

impl Options {
    pub fn list_options_uci(&self) -> Vec<String> {
        vec![
            format!("option name UCI_ShowWDL type check default {}", self.show_wdl),
            format!("option name UCI_ShowEPS type check default {}", self.show_eps),
            format!("option name MultiPV type spin default {} min 1 max 500", self.multi_pv),
            format!(
                "option name MiniBatchSize type spin default {} min 0 max 1024",
                self.mini_batch_size
            ),
            format!(
                "option name NNCacheSizePowerOfTwo type spin default {} min 0 max {}",
                self.nn_cache_size_power_of_two, MAX_NN_CACHE_SIZE_POWER_OF_TWO
            ),
            format!("option name CPuct type string default {}", self.cpuct),
            format!("option name CPuctBase type string default {}", self.cpuct_base),
            format!("option name CPuctFactor type string default {}", self.cpuct_factor),
            format!("option name FpuReduction type string default {}", self.fpu_reduction),
            format!("option name LcbStdevs type string default {}", self.lcb_stdevs),
            format!(
                "option name LcbMinVisitFraction type string default {}",
                self.lcb_min_visit_fraction
            ),
            format!("option name Threads type spin default {} min 2 max 128", self.threads),
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
            "minibatchsize" => {
                let value = value
                    .parse::<usize>()
                    .map_err(|_| crate::EnginError::Uci("MiniBatchSize must be an integer within [0, 1024]".into()))?;
                if value > 1024 {
                    return Err(crate::EnginError::Uci("MiniBatchSize must be within [0, 1024]".into()));
                }
                self.mini_batch_size = value;
            }
            "nncachesizepoweroftwo" => {
                let value = value.parse::<u8>().map_err(|_| {
                    crate::EnginError::Uci(format!(
                        "NNCacheSizePowerOfTwo must be an integer within [0, {MAX_NN_CACHE_SIZE_POWER_OF_TWO}]"
                    ))
                })?;
                if value > MAX_NN_CACHE_SIZE_POWER_OF_TWO {
                    return Err(crate::EnginError::Uci(format!(
                        "NNCacheSizePowerOfTwo must be within [0, {MAX_NN_CACHE_SIZE_POWER_OF_TWO}]"
                    )));
                }
                self.nn_cache_size_power_of_two = value;
            }
            "cpuctbase" => self.cpuct_base = parse_positive_float("CPuctBase", value)?,
            "cpuctfactor" => self.cpuct_factor = parse_non_negative_float("CPuctFactor", value)?,
            "cpuct" => self.cpuct = parse_non_negative_float("CPuct", value)?,
            "fpureduction" => self.fpu_reduction = parse_non_negative_float("FpuReduction", value)?,
            "lcbstdevs" => self.lcb_stdevs = parse_non_negative_float("LcbStdevs", value)?,
            "lcbminvisitfraction" => {
                self.lcb_min_visit_fraction = parse_unit_interval_float("LcbMinVisitFraction", value)?
            }
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

fn parse_unit_interval_float(name: &str, value: &str) -> Result<f32, crate::EnginError> {
    let value = parse_non_negative_float(name, value)?;
    if value > 1.0 {
        return Err(crate::EnginError::Uci(format!("{name} must be within [0, 1]")));
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
        .map_err(|_| crate::EnginError::Uci("Threads must be an integer within [2, 128]".into()))?;
    if !(2..=128).contains(&value) {
        return Err(crate::EnginError::Uci("Threads must be within [2, 128]".into()));
    }
    Ok(value)
}
