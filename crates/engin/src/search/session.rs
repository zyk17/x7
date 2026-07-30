//! stream 搜索的 UCI 生命周期适配层。

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use xiangqi_core::{GameState, Position, PositionHistory};

use crate::callbacks::{BestMoveInfo, SearchResponder, ThinkingInfo};
use crate::neural::backend::Backend;
use crate::{EnginError, GoParams};

use super::time::{TimeBudget, TimeManager};
use super::{SearchControl, SearchLimits, SearchState, WatchdogSnapshot};

/// Engine 持有的 LC3 风格搜索会话。
///
/// UCI 只驱动 `Engine`；由 Engine 持有会话和 worker 生命周期。
pub(crate) struct SearchSession {
    state: SearchState,
    position: Option<Position>,
    time_manager: TimeManager,
    active: Option<ActiveSearch>,
    responder: Option<Arc<dyn SearchResponder>>,
}

struct ActiveSearch {
    control: SearchControl,
    completion: Arc<Completion>,
    publish_output: Arc<Mutex<bool>>,
    search_thread: JoinHandle<()>,
    watchdog_thread: JoinHandle<()>,
    started: Instant,
    clock_budget: Option<TimeBudget>,
}

#[derive(Default)]
struct Completion {
    result: Mutex<Option<Result<super::SearchResult, EnginError>>>,
    ready: Condvar,
}

fn watchdog(
    control: SearchControl,
    snapshot: WatchdogSnapshot,
    completion: Arc<Completion>,
    publish_output: Arc<Mutex<bool>>,
    responder: Option<Arc<dyn SearchResponder>>,
    started: Instant,
) {
    // LC3 Overview 的 "WatchdogWorker"：每次搜索一个 watchdog；它只负责
    // 输出，Gather/Eval/NN/Backprop 负责搜索。
    loop {
        let mut result = completion.result.lock();
        if result.is_none() {
            completion.ready.wait_for(&mut result, Duration::from_millis(100));
        }
        if let Some(result) = result.as_ref() {
            if let Some(responder) = responder.as_ref() {
                let publishing = publish_output.lock();
                if !*publishing {
                    return;
                }
                match result {
                    Ok(result) => {
                        let mut info = snapshot.thinking_info(result.stats.clone(), started);
                        info.pv = result.principal_variation.clone();
                        responder.output_thinking_info(&[info]);
                        responder
                            .output_best_move(&BestMoveInfo::new(result.best_move.unwrap_or(xiangqi_core::Move::NULL)));
                    }
                    Err(error) => responder.output_thinking_info(&[ThinkingInfo {
                        comment: format!("stream search failed: {error}"),
                        ..ThinkingInfo::default()
                    }]),
                }
            }
            return;
        }
        drop(result);
        if let Some(responder) = responder.as_ref() {
            let publishing = publish_output.lock();
            if !*publishing {
                return;
            }
            responder.output_thinking_info(&[snapshot.thinking_info(control.stats(), started)]);
        }
    }
}

impl SearchSession {
    pub(crate) fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            state: SearchState::new(backend),
            position: None,
            time_manager: TimeManager::default(),
            active: None,
            responder: None,
        }
    }

    pub(crate) fn set_responder(&mut self, responder: Option<Arc<dyn SearchResponder>>) {
        self.responder = responder;
    }

    pub(crate) fn set_virtual_loss(&mut self, virtual_loss: f32) {
        self.state.set_virtual_loss(virtual_loss);
    }

    pub(crate) fn set_position(&mut self, state: &GameState) -> Result<(), EnginError> {
        self.abort()?;
        self.state
            .set_position(Arc::new(PositionHistory::from_positions(state.positions())))
            .map(|_| {
                self.position = Some(state.current_position());
            })
    }

    pub(crate) fn reset_clock(&mut self) {
        self.time_manager.reset();
    }

    pub(crate) fn validate_go(&self, params: &GoParams) -> Result<(), EnginError> {
        let unsupported = [
            (params.depth.is_some(), "go depth is not supported"),
            (params.mate.is_some(), "go mate is not supported"),
        ];
        if let Some((_, feature)) = unsupported.into_iter().find(|(present, _)| *present) {
            return Err(EnginError::PortIncomplete(feature));
        }
        if params.nodes.is_some_and(|nodes| nodes <= 0) {
            return Err(EnginError::Uci("go nodes must be positive".into()));
        }
        if params.movetime.is_some_and(|time| time < 0) {
            return Err(EnginError::Uci("go movetime must not be negative".into()));
        }
        let has_clock = params.wtime.is_some()
            || params.btime.is_some()
            || params.winc.is_some()
            || params.binc.is_some()
            || params.movestogo.is_some();
        if [params.wtime, params.btime, params.winc, params.binc]
            .into_iter()
            .flatten()
            .any(|value| value < 0)
            || params.movestogo.is_some_and(|value| value <= 0)
        {
            return Err(EnginError::Uci(
                "go clock values must be non-negative and movestogo positive".into(),
            ));
        }
        if has_clock {
            let position = self
                .position
                .as_ref()
                .ok_or(EnginError::Uci("position is not configured".into()))?;
            let side_time = if position.is_black_to_move() {
                params.btime
            } else {
                params.wtime
            };
            if side_time.is_none() {
                return Err(EnginError::Uci("go clock is missing side-to-move time".into()));
            }
        }
        if params.movetime.is_some() && has_clock {
            return Err(EnginError::Uci(
                "go movetime cannot be combined with clock fields".into(),
            ));
        }
        if params.infinite && (params.nodes.is_some() || params.movetime.is_some() || has_clock) {
            return Err(EnginError::Uci(
                "go infinite cannot be combined with nodes, movetime, or clock fields".into(),
            ));
        }
        if !params.infinite && params.nodes.is_none() && params.movetime.is_none() && !has_clock {
            return Err(EnginError::Uci(
                "go requires nodes, movetime, clock fields, or infinite".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn start(&mut self, params: &GoParams) -> Result<(), EnginError> {
        self.validate_go(params)?;
        self.abort()?;
        // 先验证并构造 root filter；若 `searchmoves` 非法，不能消耗时钟的首手状态
        // 或预支时钟预算。
        let running = self.state.begin_search(&params.searchmoves)?;
        let started = Instant::now();
        let clock_budget = if params.movetime.is_none() {
            let position = self.position.as_ref().expect("validated search position");
            self.time_manager.budget(params, position)
        } else {
            None
        };
        let control = running.control();
        let snapshot = running.watchdog_snapshot();
        let deadline = params
            .movetime
            .map(|ms| started + Duration::from_millis(ms.max(0) as u64))
            .or_else(|| clock_budget.as_ref().map(|budget| budget.deadline_after(started)));
        let limits = SearchLimits {
            max_playouts: params.nodes.map(|nodes| nodes.max(1) as u64),
            deadline,
        };
        let completion = Arc::new(Completion::default());
        let search_completion = Arc::clone(&completion);
        let search_thread = std::thread::spawn(move || {
            let result = running.run(limits);
            *search_completion.result.lock() = Some(result);
            search_completion.ready.notify_all();
        });
        let publish_output = Arc::new(Mutex::new(true));
        let watchdog_thread = std::thread::spawn({
            let completion = Arc::clone(&completion);
            let publish_output = Arc::clone(&publish_output);
            let responder = self.responder.clone();
            let control = control.clone();
            move || watchdog(control, snapshot, completion, publish_output, responder, started)
        });
        self.active = Some(ActiveSearch {
            control,
            completion,
            publish_output,
            search_thread,
            watchdog_thread,
            started,
            clock_budget,
        });
        Ok(())
    }

    pub(crate) fn wait(&mut self) -> Result<(), EnginError> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let ActiveSearch {
            completion,
            search_thread,
            watchdog_thread,
            started,
            clock_budget,
            ..
        } = active;
        loop {
            let mut guard = completion.result.lock();
            if guard.is_some() {
                break;
            }
            completion.ready.wait(&mut guard);
        }
        let _ = search_thread.join();
        let _ = watchdog_thread.join();
        let result = completion
            .result
            .lock()
            .take()
            .expect("completed stream search has a result");
        if let Some(clock_budget) = clock_budget {
            self.time_manager.finish(clock_budget, started.elapsed());
        }
        result.map(|_| ())
    }

    pub(crate) fn stop(&mut self) {
        if let Some(active) = &self.active {
            active.control.request_stop();
        }
    }

    pub(crate) fn abort(&mut self) -> Result<(), EnginError> {
        if let Some(active) = &self.active {
            *active.publish_output.lock() = false;
        }
        self.stop();
        self.wait()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, Once};

    use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

    use crate::callbacks::{BestMoveInfo, SearchResponder, ThinkingInfo};
    use crate::neural::backend::UniformBackend;
    use crate::GoParams;

    use super::SearchSession;

    static INIT: Once = Once::new();

    #[derive(Default)]
    struct RecordingResponder {
        bestmoves: Mutex<Vec<BestMoveInfo>>,
        infos: Mutex<Vec<ThinkingInfo>>,
    }

    impl SearchResponder for RecordingResponder {
        fn output_best_move(&self, info: &BestMoveInfo) {
            self.bestmoves.lock().expect("bestmove lock").push(info.clone());
        }

        fn output_thinking_info(&self, infos: &[ThinkingInfo]) {
            self.infos.lock().expect("info lock").extend_from_slice(infos);
        }
    }

    #[test]
    fn watchdog_reports_info_and_one_bestmove() {
        INIT.call_once(initialize_magic_bitboards);
        let responder = Arc::new(RecordingResponder::default());
        let mut search = SearchSession::new(Arc::new(UniformBackend::default()));
        search.set_responder(Some(Arc::clone(&responder) as Arc<dyn SearchResponder>));
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        search.set_position(&state).expect("position");
        search
            .start(&GoParams {
                nodes: Some(8),
                ..GoParams::default()
            })
            .expect("start");
        search.wait().expect("wait");

        let infos = responder.infos.lock().expect("info lock");
        let final_info = infos.last().expect("watchdog info");
        assert!(final_info.depth >= 1);
        assert!(final_info.seldepth >= final_info.depth);
        assert_eq!(final_info.score, Some(0));
        assert!(final_info.wdl.is_some());
        assert!(!final_info.pv.is_empty());
        assert_eq!(responder.bestmoves.lock().expect("bestmove lock").len(), 1);
    }
}
