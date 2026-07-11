use std::sync::{Arc, Mutex};

use super::{PolicyValueEval, PolicyValueOutput, PolicyValueTask};

/// lc0 `BackendComputation`：gather 阶段 `AddInput`，`UsedBatchSize` 仅计待推理条目。
/// `eval` 用裸指针避免与 `ProcessingBackend::Local` 叠代 lifetime 冲突；`BackendComputation` 不得比 `eval` 活得更久。
pub(crate) struct BackendComputation<E> {
    eval: *mut E,
    slots: Vec<EvalSlot>,
}

struct EvalSlot {
    task: Arc<PolicyValueTask>,
    cached: Option<PolicyValueOutput>,
    prefetch: bool,
}

impl<E> BackendComputation<E> {
    pub fn new(eval: &mut E) -> Self {
        Self {
            eval: eval as *mut E,
            slots: Vec::new(),
        }
    }

    pub(crate) fn eval_mut(&mut self) -> &mut E {
        // SAFETY: `new` 绑定 `eval` 指针来源；调用方保证 `BackendComputation` 不越过 `eval` 生命周期。
        unsafe { &mut *self.eval }
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
        let cached = self.eval_mut().evaluate_cached(task.as_input());
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
        if self.eval_mut().evaluate_cached(task.as_input()).is_some() {
            return true;
        }
        self.slots.push(EvalSlot {
            task: Arc::clone(task),
            cached: None,
            prefetch: true,
        });
        true
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
        for slot in &self.slots {
            if slot.cached.is_none() {
                miss_tasks.push(PolicyValueTask {
                    position: slot.task.position.clone(),
                    history: slot.task.history.clone_for_search(),
                    legal_moves: slot.task.legal_moves.clone(),
                });
            }
        }
        let fresh = if !miss_tasks.is_empty() {
            self.eval_mut().evaluate_many(&miss_tasks)?
        } else {
            Vec::new()
        };
        let mut fresh_iter = fresh.into_iter();
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
        }
        self.slots.clear();
        Ok(outputs)
    }
}

/// task worker 并行 processing 路径：共享 `eval`（lc0 共享 `computation_`）。
pub(crate) struct SharedBackendComputation<E> {
    eval: Arc<Mutex<E>>,
    slots: Mutex<Vec<EvalSlot>>,
}

impl<E> SharedBackendComputation<E> {
    pub fn new(eval: Arc<Mutex<E>>) -> Self {
        Self {
            eval,
            slots: Mutex::new(Vec::new()),
        }
    }

    pub fn used_batch_size(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|slot| !slot.prefetch && slot.cached.is_none())
            .count()
    }

    pub fn add_input(&self, task: &Arc<PolicyValueTask>) -> bool
    where
        E: PolicyValueEval,
    {
        let cached = self
            .eval
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .evaluate_cached(task.as_input());
        let immediate = cached.is_some();
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(EvalSlot {
                task: Arc::clone(task),
                cached,
                prefetch: false,
            });
        immediate
    }

    pub fn add_prefetch_input(&self, task: &Arc<PolicyValueTask>) -> bool
    where
        E: PolicyValueEval,
    {
        if self
            .eval
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .evaluate_cached(task.as_input())
            .is_some()
        {
            return true;
        }
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(EvalSlot {
                task: Arc::clone(task),
                cached: None,
                prefetch: true,
            });
        true
    }

    pub fn with_eval_mut<R>(&self, f: impl FnOnce(&mut E) -> R) -> R {
        let mut guard = self.eval.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }

    pub fn compute_blocking(&self) -> Result<Vec<PolicyValueOutput>, E::Error>
    where
        E: PolicyValueEval,
    {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let mut outputs = Vec::new();
        let mut miss_tasks = Vec::new();
        for slot in slots.iter() {
            if slot.cached.is_none() {
                miss_tasks.push(PolicyValueTask {
                    position: slot.task.position.clone(),
                    history: slot.task.history.clone_for_search(),
                    legal_moves: slot.task.legal_moves.clone(),
                });
            }
        }
        let fresh = if !miss_tasks.is_empty() {
            self.eval
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .evaluate_many(&miss_tasks)?
        } else {
            Vec::new()
        };
        let mut fresh_iter = fresh.into_iter();
        for slot in slots.iter() {
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
        }
        slots.clear();
        Ok(outputs)
    }
}
