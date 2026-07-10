use std::sync::Arc;

use super::{PolicyValueEval, PolicyValueOutput, PolicyValueTask};

/// px0 风格 NN batch：收集 expand 任务后一次性推理。
pub(crate) struct BackendComputation<'a, E> {
    eval: &'a mut E,
    tasks: Vec<Arc<PolicyValueTask>>,
}

impl<'a, E> BackendComputation<'a, E> {
    pub fn new(eval: &'a mut E) -> Self {
        Self {
            eval,
            tasks: Vec::new(),
        }
    }

    pub fn add_input(&mut self, task: &Arc<PolicyValueTask>) {
        self.tasks.push(Arc::clone(task));
    }

    pub fn compute_blocking(&mut self) -> Result<Vec<PolicyValueOutput>, E::Error>
    where
        E: PolicyValueEval,
    {
        self.eval.evaluate_many_arcs(&self.tasks)
    }
}
