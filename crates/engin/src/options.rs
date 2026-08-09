//! Engine 生命周期的 UCI option。
//!
//! 对照 px0 `src/engine.h` 的 `OptionsDict`：`Engine` 持有 option，搜索在
//! 启动 job 时读取需要的快照。

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
    /// UCI `CPuct`：PUCT 的初始探索系数。
    pub cpuct: f32,
    /// UCI `CPuctBase`/`CPuctFactor`：PUCT 随访问数增长的形状。
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    /// UCI `FpuReduction`：未访问边相对 parent Q 的 FPU 降幅。
    pub fpu_reduction: f32,
    /// 三类 stream worker 的常驻线程数。
    pub gather_workers: usize,
    pub eval_workers: usize,
    pub backprop_workers: usize,
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
            gather_workers: 4,
            eval_workers: 4,
            backprop_workers: 1,
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
            format!(
                "option name GatherWorkers type spin default {} min 1 max 64",
                self.gather_workers
            ),
            format!(
                "option name EvalWorkers type spin default {} min 1 max 64",
                self.eval_workers
            ),
            format!(
                "option name BackpropWorkers type spin default {} min 1 max 64",
                self.backprop_workers
            ),
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
            "gatherworkers" => self.gather_workers = parse_worker_count(name, value)?,
            "evalworkers" => self.eval_workers = parse_worker_count(name, value)?,
            "backpropworkers" => self.backprop_workers = parse_worker_count(name, value)?,
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

fn parse_worker_count(name: &str, value: &str) -> Result<usize, crate::EnginError> {
    let value = value
        .parse::<usize>()
        .map_err(|_| crate::EnginError::Uci(format!("{name} must be an integer within [1, 64]")))?;
    if !(1..=64).contains(&value) {
        return Err(crate::EnginError::Uci(format!("{name} must be within [1, 64]")));
    }
    Ok(value)
}
