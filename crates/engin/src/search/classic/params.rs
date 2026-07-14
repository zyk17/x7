//! px0 `src/search/classic/params.h:37-260`、`params.cc:543-640` 默认值子集。

/// px0 `ScoreType` choices (`src/search/classic/params.cc:587-595`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ScoreType {
    Centipawn,
    CentipawnWithDrawscore,
    Centipawn2019,
    Centipawn2018,
    WinPercentage,
    Q,
    WinLoss,
    #[default]
    WdlMu,
}

impl ScoreType {
    pub const fn as_uci(self) -> &'static str {
        match self {
            Self::Centipawn => "centipawn",
            Self::CentipawnWithDrawscore => "centipawn_with_drawscore",
            Self::Centipawn2019 => "centipawn_2019",
            Self::Centipawn2018 => "centipawn_2018",
            Self::WinPercentage => "win_percentage",
            Self::Q => "Q",
            Self::WinLoss => "W-L",
            Self::WdlMu => "WDL_mu",
        }
    }

    pub fn parse_uci(value: &str) -> Option<Self> {
        Some(match value {
            "centipawn" => Self::Centipawn,
            "centipawn_with_drawscore" => Self::CentipawnWithDrawscore,
            "centipawn_2019" => Self::Centipawn2019,
            "centipawn_2018" => Self::Centipawn2018,
            "win_percentage" => Self::WinPercentage,
            "Q" => Self::Q,
            "W-L" => Self::WinLoss,
            "WDL_mu" => Self::WdlMu,
            _ => return None,
        })
    }
}

/// px0 `ContemptMode` (`src/search/classic/params.h:37`). `Play` is resolved
/// once per `StartSearch`; workers only receive White, Black, or None.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ContemptMode {
    #[default]
    Play,
    White,
    Black,
    None,
}

/// px0 `BaseSearchParams::WDLRescaleParams` (`params.h:44-53`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WdlRescaleParams {
    pub ratio: f32,
    pub diff: f32,
}

/// px0 `AccurateWDLRescaleParams` (`params.cc:92-115`).
pub fn accurate_wdl_rescale_params(
    contempt: f32,
    mut draw_rate_target: f32,
    draw_rate_reference: f32,
    book_exit_bias: f32,
    contempt_max: f32,
    contempt_attenuation: f32,
) -> WdlRescaleParams {
    if draw_rate_target > 0.0 && draw_rate_target < 0.001 {
        draw_rate_target = 0.001;
    }
    let scale_reference = 1.0 / ((1.0 + draw_rate_reference) / (1.0 - draw_rate_reference)).ln();
    let scale_target = if draw_rate_target == 0.0 {
        scale_reference
    } else {
        1.0 / ((1.0 + draw_rate_target) / (1.0 - draw_rate_target)).ln()
    };
    let ratio = scale_target / scale_reference;
    let sech2_left = 1.0 / (0.5 * (1.0 - book_exit_bias) / scale_target).cosh().powi(2);
    let sech2_right = 1.0 / (0.5 * (1.0 + book_exit_bias) / scale_target).cosh().powi(2);
    let diff = scale_target / (scale_reference * scale_reference) / (sech2_left + sech2_right) * 10.0_f32.ln() / 200.0
        * contempt.clamp(-contempt_max, contempt_max)
        * contempt_attenuation;
    WdlRescaleParams { ratio, diff }
}

/// px0 `ConvertRegularToGamePairElo` / `SimplifiedWDLRescaleParams`
/// (`src/search/classic/params.cc:120-174`).
pub fn simplified_wdl_rescale_params(
    contempt: f32,
    draw_rate_reference: f32,
    mut elo_active: f32,
    contempt_max: f32,
    contempt_attenuation: f32,
) -> WdlRescaleParams {
    const SCALE_ZERO: f32 = 15.0;
    const ELO_SLOPE: f32 = 425.0;
    const OFFSET: f32 = 6.75;
    let scale_reference = 1.0 / ((1.0 + draw_rate_reference) / (1.0 - draw_rate_reference)).ln();
    let mut elo_opp = elo_active - contempt.clamp(-contempt_max, contempt_max);
    let convert = |elo: f32| elo + 0.5 * 250.0 * (1.0 + ((2737.0 - elo) / 250.0).exp()).ln();
    elo_active = convert(elo_active);
    elo_opp = convert(elo_opp);
    let scale = |elo: f32| 1.0 / (1.0 / SCALE_ZERO + (elo / ELO_SLOPE - OFFSET).exp());
    let scale_active = scale(elo_active);
    let scale_opp = scale(elo_opp);
    let scale_target = ((scale_active * scale_active + scale_opp * scale_opp) / 2.0).sqrt();
    let ratio = scale_target / scale_reference;
    let mu = |elo: f32| {
        -10.0_f32.ln() / 200.0 * SCALE_ZERO * ELO_SLOPE * (1.0 + (-elo / ELO_SLOPE + OFFSET).exp() / SCALE_ZERO).ln()
    };
    let diff = (mu(elo_active) - mu(elo_opp)) / (scale_reference * scale_reference) * contempt_attenuation;
    WdlRescaleParams { ratio, diff }
}

/// px0 `BaseSearchParams` / `SearchParams` 单线程搜索所需字段。
#[derive(Clone, Debug)]
pub struct SearchParams {
    // 引擎试图打包多少个局面（positions）进行神经网络的并行计算。较大的 Batch 可能会稍微降低棋力，
    // 特别是在总模拟次数（playouts）较少的情况下。设置为 0 则使用后端推荐的默认值。
    pub minibatch_size: i32,
    // 较高的值会促进更多的探索（更宽的搜索），较低的值会促进更多的置信度（更深的搜索）。
    pub cpuct: f32,
    /// px0 `CpuctAtRoot` (`src/search/classic/params.cc:543-551`).
    pub cpuct_at_root: f32,
    // 较低的值意味着：随着节点访问次数的增加，Cpuct 的增长速度更快。
    pub cpuct_base: f32,
    /// px0 `CpuctBaseAtRoot` (`src/search/classic/params.cc:546-551`).
    pub cpuct_base_at_root: f32,
    // Cpuct 增长公式的乘数因子（Multiplier）
    pub cpuct_factor: f32,
    /// px0 `CpuctFactorAtRoot` (`src/search/classic/params.cc:548-551`).
    pub cpuct_factor_at_root: f32,
    /// px0 `RootHasOwnCpuctParams` (`src/search/classic/params.cc:551`).
    /// When disabled, all three `*AtRoot` values are ignored.
    pub root_has_own_cpuct_params: bool,
    // 决定如何评估未访问的节点。FPU 通过在查询神经网络之前使用一个占位评估值，来改变搜索行为，
    // 使引擎更早或更晚地访问未访问节点。reduction（减少）策略会从父节点评估中减去用 --fpu-value 指定的值；
    // absolute（绝对值）策略则直接使用该值。
    pub fpu_absolute: bool,
    pub fpu_value: f32,
    pub fpu_absolute_at_root: bool,
    pub fpu_value_at_root: f32,
    pub fpu_strategy_at_root_same: bool,
    /// px0 `DrawScore` (`src/search/classic/params.cc:605`); zero is neutral.
    pub draw_score: f32,
    /// px0 `TwoFoldDraws`: tree-reused two-fold terminals are reverted when
    /// the first repetition predates the current root (`search.cc:1510-1550`).
    pub two_fold_draws: bool,
    // 在搜索过程中发现终局（游戏结束）局面时，允许前一步局面的评估“粘附”到更准确的值上。
    // 例如，如果至少有一个走法会导致杀棋（Checkmate），则该局面应粘附为“被杀棋”；
    // 同理，如果所有走法都是和棋或被杀，则局面应粘附为和棋或被杀。
    pub sticky_endgames: bool,
    // 只有访问次数至少达到该数值的节点，才会被考虑进行固化，以提高缓存的局部性（Cache Locality）。
    pub solid_tree_threshold: u32,
    // （仅专业模式可见）如果设为 0，引擎会直接选择最佳步法（贪婪选择）。较大的值会增加落子时的随机性。
    pub temperature: f32,
    // 如果不为 0，则最多允许这么多个搜索工作线程同时收集 Mini-batch。
    pub max_concurrent_searchers: i32,
    /// px0 `SearchSpinBackoff` (`params.cc:525-526,632`): choose an
    /// exponential backoff instead of the default hard-spin while waiting for
    /// a `MaxConcurrentSearchers` permit.
    pub search_spin_backoff: bool,
    /// px0 `NodesPerSecondLimit` (`params.cc:473-477,621`). Zero disables
    /// post-iteration throughput throttling.
    pub nps_limit: f32,
    /// px0 `BaseSearchParams::WDLRescaleParams` / `WDLMaxS`
    /// (`params.h:44-53,125-128`). Defaults are the neutral no-contempt
    /// result of `AccurateWDLRescaleParams`.
    pub wdl_rescale_ratio: f32,
    pub wdl_rescale_diff: f32,
    pub wdl_max_s: f32,
    pub contempt_mode: ContemptMode,
    // 指在多线程并行搜索中，多个线程同时访问并等待同一个尚未被神经网络估值的节点时的最大允许访问数。
    pub max_collision_visits: i32,
    // 最大碰撞访问量开始进行比例缩放的树尺寸。
    // 即从多少个节点开始，最大碰撞访问量从 1 开始向上按比例缩放。
    pub max_collision_visits_scaling_start: i32,
    // 最大碰撞访问量达到最大缩放的树尺寸。
    // 设置为 0 可以完全禁用此缩放。
    pub max_collision_visits_scaling_end: i32,
    // 碰撞访问缩放曲线的幂指数（Power）。
    // 应用于 1 到最大值之间插值的幂指数，使其缩放轨迹呈现曲线形态。
    pub max_collision_visits_scaling_power: f32,
    // 无序评估（即时评估）。
    // 在收集一批局面发送给神经网络时，如果某个局面恰好在缓存中或是终局节点，则立即对其进行评估，而不放入 Batch 发送给 NN。
    // 关闭时，这只可能发生在 Batch 的第一个节点上；开启时，可以发生在任意节点上。
    pub out_of_order_eval: bool,
    // 无序评估的最大数量系数。
    // 在收集一个 Batch 期间，允许的最大无序评估数量通过【最大 Batch 大小 * 该系数】计算得出。
    pub max_out_of_order_evals_factor: f32,
    // 辅助任务工作线程（Task Workers）的数量。
    // 用于协助搜索工作线程。设置为 -1 将使用启发式默认值。
    pub task_workers_per_search_worker: i32,
    // 启动任务加速的最小访问量。
    // 在利用辅助任务加速处理之前，必须收集齐至少这么多数量的访问。
    pub minimum_work_size_for_processing: i32,
    // 分流至任务线程的最小分支访问量。
    // 具有超过该数值的碰撞/访问的搜索分支，可以被分流（split off）给辅助任务线程处理。
    pub minimum_work_size_for_picking: i32,
    // 分流后剩余工作的最小限制。
    // 除非分流后仍然剩下至少这么多工作要做，否则搜索分支不会被拆分给辅助任务线程。
    pub minimum_remaining_work_size_for_picking: i32,
    // 单个任务的最小处理工作量。
    // 处理工作不会被拆分成小于该值的任务块（除非其超过 MinimumProcessingWork 的一半）。
    pub minimum_work_per_task_for_processing: i32,
    /// px0 `IdlingMinimumWork` (`params.cc:498-501,628`): after this many
    /// queued NN inputs, a worker may leave gather early when another worker
    /// can keep the backend busy.
    pub idling_minimum_work: i32,
    /// px0 `ThreadIdlingThreshold` (`params.cc:502-505,629`).
    pub thread_idling_threshold: i32,
    // 最大预取（Prefetch）批量。
    // 当引擎无法收集到足够大的 Batch 供即时使用时，尝试预取最多 X 个可能很快会有用的局面，并将它们放入缓存中。
    pub max_prefetch_batch: i32,
    /// px0 `MultiPV` / `PerPVCounters` (`params.cc:360-368,585-586`).
    /// 多线分析数量（主要变线 Principal Variations）。
    // （始终可见）在 UCI 信息输出中显示的对局线（PV）数量。
    pub multi_pv: usize,
    // 在 UCI 中按主要变线（PV）显示节点计数。
    // 显示每条 PV 分布的节点数，而不是仅显示总节点数。
    pub per_pv_counters: bool,
    /// px0 `ScoreType` (`params.cc:587-595`). `Q` and `W-L` output are
    /// their internal values scaled by 10_000, not centipawns.
    pub score_type: ScoreType,
}

fn mix(high: i32, low: i32, ratio: f32) -> i32 {
    (low as f32 + (high - low) as f32 * ratio).round() as i32
}

impl Default for SearchParams {
    /// px0 `params.cc` 注册默认值。
    fn default() -> Self {
        Self {
            minibatch_size: 0,
            cpuct: 1.0,
            cpuct_at_root: 1.745,
            cpuct_base: 38_739.0,
            cpuct_base_at_root: 38_739.0,
            cpuct_factor: 3.894,
            cpuct_factor_at_root: 3.894,
            root_has_own_cpuct_params: false,
            fpu_absolute: false,
            fpu_value: 0.220,
            fpu_absolute_at_root: false,
            fpu_value_at_root: 1.0,
            fpu_strategy_at_root_same: true,
            draw_score: 0.0,
            two_fold_draws: true,
            sticky_endgames: true,
            solid_tree_threshold: 100,
            temperature: 0.0,
            max_concurrent_searchers: 1,
            search_spin_backoff: false,
            nps_limit: 0.0,
            wdl_rescale_ratio: 1.0,
            wdl_rescale_diff: 0.0,
            wdl_max_s: 1.4,
            contempt_mode: ContemptMode::Play,
            max_collision_visits: 80_000,
            max_collision_visits_scaling_start: 28,
            max_collision_visits_scaling_end: 145_000,
            max_collision_visits_scaling_power: 1.25,
            out_of_order_eval: true,
            max_out_of_order_evals_factor: 2.4,
            task_workers_per_search_worker: -1,
            minimum_work_size_for_processing: 20,
            minimum_work_size_for_picking: 1,
            minimum_remaining_work_size_for_picking: 20,
            minimum_work_per_task_for_processing: 8,
            idling_minimum_work: 0,
            thread_idling_threshold: 1,
            max_prefetch_batch: 32,
            multi_pv: 1,
            per_pv_counters: false,
            score_type: ScoreType::WdlMu,
        }
    }
}

impl SearchParams {
    pub fn cpuct(&self, at_root: bool) -> f32 {
        if at_root && self.root_has_own_cpuct_params {
            self.cpuct_at_root
        } else {
            self.cpuct
        }
    }

    /// px0 `BaseSearchParams::GetCpuctBase` (`params.h:60-62`).
    pub fn cpuct_base(&self, at_root: bool) -> f32 {
        if at_root && self.root_has_own_cpuct_params {
            self.cpuct_base_at_root
        } else {
            self.cpuct_base
        }
    }

    /// px0 `BaseSearchParams::GetCpuctFactor` (`params.h:63-65`).
    pub fn cpuct_factor(&self, at_root: bool) -> f32 {
        if at_root && self.root_has_own_cpuct_params {
            self.cpuct_factor_at_root
        } else {
            self.cpuct_factor
        }
    }

    pub fn fpu_absolute(&self, at_root: bool) -> bool {
        if at_root && !self.fpu_strategy_at_root_same {
            self.fpu_absolute_at_root
        } else {
            self.fpu_absolute
        }
    }

    pub fn fpu_value(&self, at_root: bool) -> f32 {
        if at_root && !self.fpu_strategy_at_root_same {
            self.fpu_value_at_root
        } else {
            self.fpu_value
        }
    }

    /// px0 `CalculateCollisionsLeft` (`search.cc:1251-1265`)。
    pub fn collisions_left(&self, nodes: i64) -> i32 {
        if nodes >= self.max_collision_visits_scaling_end as i64 {
            return self.max_collision_visits;
        }
        if nodes <= self.max_collision_visits_scaling_start as i64 {
            return 1;
        }
        let ratio = ((nodes - self.max_collision_visits_scaling_start as i64) as f32)
            / ((self.max_collision_visits_scaling_end - self.max_collision_visits_scaling_start) as f32);
        mix(
            self.max_collision_visits,
            1,
            ratio.powf(self.max_collision_visits_scaling_power),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{accurate_wdl_rescale_params, simplified_wdl_rescale_params, SearchParams};

    /// px0 `BaseSearchParams` keeps each root PUCT parameter equal to the
    /// ordinary value unless `RootHasOwnCpuctParams` is enabled
    /// (`src/search/classic/params.cc:644-655`).
    #[test]
    fn root_cpuct_parameters_follow_px0_gate() {
        let mut params = SearchParams {
            cpuct: 1.0,
            cpuct_at_root: 2.0,
            cpuct_base: 100.0,
            cpuct_base_at_root: 200.0,
            cpuct_factor: 3.0,
            cpuct_factor_at_root: 4.0,
            ..SearchParams::default()
        };

        assert_eq!(params.cpuct(true), 1.0);
        assert_eq!(params.cpuct_base(true), 100.0);
        assert_eq!(params.cpuct_factor(true), 3.0);

        params.root_has_own_cpuct_params = true;
        assert_eq!(params.cpuct(true), 2.0);
        assert_eq!(params.cpuct_base(true), 200.0);
        assert_eq!(params.cpuct_factor(true), 4.0);
    }

    /// px0 optimized defaults (`src/search/classic/params.cc:543-583`).
    #[test]
    fn defaults_match_px0_classic() {
        let params = SearchParams::default();
        assert_eq!(params.fpu_value_at_root, 1.0);
        assert_eq!(params.max_collision_visits_scaling_end, 145_000);
    }

    #[test]
    fn accurate_wdl_defaults_are_px0_neutral() {
        let params = accurate_wdl_rescale_params(0.0, 0.0, 0.5, 0.65, 420.0, 1.0);
        assert!((params.ratio - 1.0).abs() < f32::EPSILON);
        assert!(params.diff.abs() < f32::EPSILON);
    }

    #[test]
    fn simplified_wdl_contempt_changes_only_diff_at_equal_elo() {
        let neutral = simplified_wdl_rescale_params(0.0, 0.5, 2000.0, 420.0, 1.0);
        let contempt = simplified_wdl_rescale_params(100.0, 0.5, 2000.0, 420.0, 1.0);
        assert!(neutral.ratio.is_finite() && contempt.ratio.is_finite());
        assert!(contempt.diff.is_finite());
        assert!(contempt.diff.abs() > neutral.diff.abs());
    }
}
