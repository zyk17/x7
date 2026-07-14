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

/// px0 `BaseSearchParams` / `SearchParams` 单线程搜索所需字段。
#[derive(Clone, Debug)]
pub struct SearchParams {
    // 引擎试图打包多少个局面（positions）进行神经网络的并行计算。较大的 Batch 可能会稍微降低棋力，
    // 特别是在总模拟次数（playouts）较少的情况下。设置为 0 则使用后端推荐的默认值。
    pub minibatch_size: i32,
    // 较高的值会促进更多的探索（更宽的搜索），较低的值会促进更多的置信度（更深的搜索）。
    pub cpuct: f32,
    // （仅专业模式可见）专门应用于根节点的 cpuct_init 参数。
    pub cpuct_at_root: f32,
    // 较低的值意味着：随着节点访问次数的增加，Cpuct 的增长速度更快。
    pub cpuct_base: f32,
    
    pub cpuct_factor: f32,
    pub root_has_own_cpuct_params: bool,
    pub fpu_absolute: bool,
    pub fpu_value: f32,
    pub fpu_absolute_at_root: bool,
    pub fpu_value_at_root: f32,
    pub fpu_strategy_at_root_same: bool,
    pub draw_score: f32,
    pub two_fold_draws: bool,
    pub sticky_endgames: bool,
    pub solid_tree_threshold: u32,
    pub temperature: f32,
    pub max_concurrent_searchers: i32,
    pub max_collision_visits: i32,
    pub max_collision_visits_scaling_start: i32,
    pub max_collision_visits_scaling_end: i32,
    pub max_collision_visits_scaling_power: f32,
    pub out_of_order_eval: bool,
    pub max_out_of_order_evals_factor: f32,
    pub task_workers_per_search_worker: i32,
    pub minimum_work_size_for_processing: i32,
    pub minimum_work_size_for_picking: i32,
    pub minimum_remaining_work_size_for_picking: i32,
    pub minimum_work_per_task_for_processing: i32,
    pub max_prefetch_batch: i32,
    /// px0 `MultiPV` / `PerPVCounters` (`params.cc:360-368,585-586`).
    pub multi_pv: usize,
    pub per_pv_counters: bool,
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
            cpuct_factor: 3.894,
            root_has_own_cpuct_params: false,
            fpu_absolute: false,
            fpu_value: 0.220,
            fpu_absolute_at_root: false,
            fpu_value_at_root: 0.220,
            fpu_strategy_at_root_same: true,
            draw_score: 0.0,
            two_fold_draws: true,
            sticky_endgames: true,
            solid_tree_threshold: 100,
            temperature: 0.0,
            max_concurrent_searchers: 1,
            max_collision_visits: 80_000,
            max_collision_visits_scaling_start: 28,
            max_collision_visits_scaling_end: 100_000,
            max_collision_visits_scaling_power: 1.25,
            out_of_order_eval: true,
            max_out_of_order_evals_factor: 2.4,
            task_workers_per_search_worker: -1,
            minimum_work_size_for_processing: 20,
            minimum_work_size_for_picking: 1,
            minimum_remaining_work_size_for_picking: 20,
            minimum_work_per_task_for_processing: 8,
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

    pub fn cpuct_base(&self, _at_root: bool) -> f32 {
        self.cpuct_base
    }

    pub fn cpuct_factor(&self, _at_root: bool) -> f32 {
        self.cpuct_factor
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
