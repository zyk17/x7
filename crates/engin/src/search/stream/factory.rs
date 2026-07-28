//! Stream implementation of the generic search construction boundary.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use xiangqi_core::{GameState, PositionHistory};

use crate::callbacks::{BestMoveInfo, SearchResponder, ThinkingInfo};
use crate::neural::backend::Backend;
use crate::search::{SearchBase, SearchFactory};
use crate::{EnginError, GoParams};

use super::{Runner, SearchControl, SearchLimits, WatchdogSnapshot};

#[derive(Clone, Copy, Debug, Default)]
pub struct Factory;

impl SearchFactory for Factory {
    fn create(&self, backend: Arc<dyn Backend>) -> Box<dyn SearchBase> {
        Box::new(StreamSearch {
            controller: Runner::new(backend),
            active: None,
            completed: None,
            responder: None,
        })
    }
}

/// `SearchBase` adapter for the LC3-style search.
pub struct StreamSearch {
    controller: Runner,
    active: Option<ActiveSearch>,
    completed: Option<super::SearchResult>,
    responder: Option<Arc<dyn SearchResponder>>,
}

struct ActiveSearch {
    control: SearchControl,
    completion: Arc<Completion>,
    publish_bestmove: Arc<AtomicBool>,
    search_thread: JoinHandle<()>,
    watchdog_thread: JoinHandle<()>,
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
    publish_bestmove: Arc<AtomicBool>,
    responder: Option<Arc<dyn SearchResponder>>,
    started: Instant,
) {
    // LC3 overview, "WatchdogWorker": this is deliberately one thread per
    // search. It owns reporting, while Gather/Eval/NN/Backprop own search.
    loop {
        let mut result = completion.result.lock();
        if result.is_none() {
            completion.ready.wait_for(&mut result, Duration::from_millis(100));
        }
        if let Some(result) = result.as_ref() {
            if let Some(responder) = responder.as_ref() {
                match result {
                    Ok(result) => {
                        let mut info = snapshot.thinking_info(result.stats, started);
                        info.pv = result.principal_variation.clone();
                        responder.output_thinking_info(&[info]);
                        if publish_bestmove.load(Ordering::Acquire) {
                            responder.output_best_move(&BestMoveInfo::new(
                                result.best_move.unwrap_or(xiangqi_core::Move::NULL),
                            ));
                        }
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
            responder.output_thinking_info(&[snapshot.thinking_info(control.stats(), started)]);
        }
    }
}

impl SearchBase for StreamSearch {
    fn set_responder(&mut self, responder: Option<Arc<dyn SearchResponder>>) {
        self.responder = responder;
    }

    fn new_game(&mut self) -> Result<(), EnginError> {
        self.abort_search()?;
        self.completed = None;
        Ok(())
    }

    fn set_position(&mut self, state: &GameState) -> Result<(), EnginError> {
        self.abort_search()?;
        self.completed = None;
        self.controller
            .set_position(Arc::new(PositionHistory::from_positions(state.positions())))
            .map(|_| ())
    }

    fn validate_go(&self, params: &GoParams) -> Result<(), EnginError> {
        let unsupported = [
            (
                params.wtime.is_some() || params.btime.is_some(),
                "go wtime/go btime time control",
            ),
            (
                params.winc.is_some() || params.binc.is_some(),
                "go winc/go binc time control",
            ),
            (params.movestogo.is_some(), "go movestogo time control"),
            (params.depth.is_some(), "go depth stopper"),
            (params.mate.is_some(), "go mate stopper"),
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
        if params.infinite && (params.nodes.is_some() || params.movetime.is_some()) {
            return Err(EnginError::Uci(
                "go infinite cannot be combined with nodes or movetime".into(),
            ));
        }
        if !params.infinite && params.nodes.is_none() && params.movetime.is_none() {
            return Err(EnginError::Uci("go requires nodes, movetime, or infinite".into()));
        }
        Ok(())
    }

    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError> {
        self.validate_go(params)?;
        self.abort_search()?;
        self.completed = None;
        let running = self.controller.begin_search(&params.searchmoves)?;
        let control = running.control();
        let snapshot = running.watchdog_snapshot();
        let limits = SearchLimits {
            max_playouts: params.nodes.map(|nodes| nodes.max(1) as u64),
            deadline: params
                .movetime
                .map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms.max(0) as u64)),
        };
        let completion = Arc::new(Completion::default());
        let search_completion = Arc::clone(&completion);
        let search_thread = std::thread::spawn(move || {
            let result = running.run(limits);
            *search_completion.result.lock() = Some(result);
            search_completion.ready.notify_all();
        });
        let publish_bestmove = Arc::new(AtomicBool::new(true));
        let watchdog_thread = std::thread::spawn({
            let completion = Arc::clone(&completion);
            let publish_bestmove = Arc::clone(&publish_bestmove);
            let responder = self.responder.clone();
            let control = control.clone();
            move || {
                watchdog(
                    control,
                    snapshot,
                    completion,
                    publish_bestmove,
                    responder,
                    Instant::now(),
                )
            }
        });
        self.active = Some(ActiveSearch {
            control,
            completion,
            publish_bestmove,
            search_thread,
            watchdog_thread,
        });
        Ok(())
    }

    fn start_clock(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn wait_search(&mut self) -> Result<(), EnginError> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        loop {
            let mut guard = active.completion.result.lock();
            if guard.is_some() {
                break;
            }
            active.completion.ready.wait(&mut guard);
        }
        let _ = active.search_thread.join();
        let _ = active.watchdog_thread.join();
        let result = active
            .completion
            .result
            .lock()
            .take()
            .expect("completed stream search has a result");
        self.completed = Some(result?);
        Ok(())
    }

    fn stop_search(&mut self) -> Result<(), EnginError> {
        if let Some(active) = &self.active {
            active.control.request_stop();
        }
        Ok(())
    }

    fn abort_search(&mut self) -> Result<(), EnginError> {
        if let Some(active) = &self.active {
            active.publish_bestmove.store(false, Ordering::Release);
        }
        self.stop_search()?;
        self.wait_search()
    }

    fn best_move(&self) -> Option<xiangqi_core::Move> {
        self.completed.as_ref().and_then(|result| result.best_move)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, Once};

    use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

    use crate::callbacks::{BestMoveInfo, SearchResponder, ThinkingInfo};
    use crate::neural::backend::UniformBackend;
    use crate::search::SearchBase;
    use crate::GoParams;

    use super::StreamSearch;

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
        let mut search = StreamSearch {
            controller: super::Runner::new(Arc::new(UniformBackend::default())),
            active: None,
            completed: None,
            responder: Some(Arc::clone(&responder) as Arc<dyn SearchResponder>),
        };
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        search.set_position(&state).expect("position");
        search
            .start_search(&GoParams {
                nodes: Some(8),
                ..GoParams::default()
            })
            .expect("start");
        search.wait_search().expect("wait");

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
