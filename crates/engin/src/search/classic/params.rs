//! px0 `src/search/classic/params.h:37-260`、`params.cc:543-640` 默认值子集。

/// px0 `BaseSearchParams` / `SearchParams` 单线程搜索所需字段。
#[derive(Clone, Debug)]
pub struct SearchParams {
    pub minibatch_size: i32,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
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
}
