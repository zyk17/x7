use std::sync::Arc;

use super::{PolicyValueEval, PolicyValueOutput, PolicyValueTask};

/// lc0 `BackendComputation`：gather 阶段 `AddInput`，`UsedBatchSize` 仅计待推理条目。
pub(crate) struct BackendComputation<'a, E> {
    eval: &'a mut E,
    slots: Vec<EvalSlot>,
}

struct EvalSlot {
    task: Arc<PolicyValueTask>,
    cached: Option<PolicyValueOutput>,
    prefetch: bool,
}

impl<'a, E> BackendComputation<'a, E> {
    pub fn new(eval: &'a mut E) -> Self {
        Self {
            eval,
            slots: Vec::new(),
        }
    }

    /// lc0 `UsedBatchSize`：尚未 cache hit、需要进 NN batch 的输入数（不含 prefetch-only）。
    pub fn used_batch_size(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| !slot.prefetch && slot.cached.is_none())
            .count()
    }

    /// lc0 `AddInput`：返回 `true` 表示 `FETCHED_IMMEDIATELY`（cache hit）。
    pub fn add_input(&mut self, task: &Arc<PolicyValueTask>) -> bool
    where
        E: PolicyValueEval,
    {
        let cached = self.eval.evaluate_cached(task.as_input());
        let immediate = cached.is_some();
        self.slots.push(EvalSlot {
            task: Arc::clone(task),
            cached,
            prefetch: false,
        });
        immediate
    }

    /// lc0 prefetch：`AddInput` 进 batch 暖 cache，不计入 backup 输出。
    pub fn add_prefetch_input(&mut self, task: &Arc<PolicyValueTask>) -> bool
    where
        E: PolicyValueEval,
    {
        if self.eval.evaluate_cached(task.as_input()).is_some() {
            return true;
        }
        self.slots.push(EvalSlot {
            task: Arc::clone(task),
            cached: None,
            prefetch: true,
        });
        true
    }

    pub fn eval_mut(&mut self) -> &mut E {
        self.eval
    }

    pub fn compute_blocking(&mut self) -> Result<Vec<PolicyValueOutput>, E::Error>
    where
        E: PolicyValueEval,
    {
        if self.slots.is_empty() {
            return Ok(Vec::new());
        }
        let mut outputs = Vec::new();
        let mut miss_tasks = Vec::new();
        let mut miss_slots = Vec::new();
        for (idx, slot) in self.slots.iter().enumerate() {
            if slot.cached.is_none() {
                miss_slots.push(idx);
                miss_tasks.push(PolicyValueTask {
                    position: slot.task.position.clone(),
                    history: slot.task.history.clone_for_search(),
                    legal_moves: slot.task.legal_moves.clone(),
                });
            }
        }
        let fresh = if !miss_tasks.is_empty() {
            self.eval.evaluate_many(&miss_tasks)?
        } else {
            Vec::new()
        };
        let mut fresh_iter = fresh.into_iter();
        let mut minibatch_idx = 0usize;
        for slot in &self.slots {
            if slot.prefetch {
                if slot.cached.is_none() {
                    let _ = fresh_iter.next();
                }
                continue;
            }
            let out = if let Some(cached) = &slot.cached {
                cached.clone()
            } else {
                fresh_iter
                    .next()
                    .expect("batched eval size mismatch")
            };
            outputs.push(out);
            minibatch_idx += 1;
            let _ = minibatch_idx;
        }
        Ok(outputs)
    }
}
