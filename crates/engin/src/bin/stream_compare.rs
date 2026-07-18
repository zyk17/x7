//! Real-ONNX equivalence probe for the LC3-style stream search.
//!
//! Reference: LC3 overview, "Workers" and "Node repository":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>.
//! This binary compares the deterministic serial baseline with the owned-event
//! worker pipeline. It deliberately does not compare exact edge visit counts:
//! queue scheduling changes selection order. The required invariants are the
//! same completed playout count, legal root expansion, and no root reservation
//! left in flight.

use std::cmp::Ordering;
use std::path::PathBuf;
use std::sync::Arc;

use engin::neural::backend::{Backend, CachingBackend};
use engin::neural::onnx::OnnxBackend;
use engin::search::classic::ClassicSearch;
use engin::search::stream::{SearchGeneration, StreamSearch, StreamStats, StreamWorkerConfig, StreamWorkerPipeline};
use engin::SearchBase;
use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

const DEFAULT_PLAYOUTS: u64 = 128;
const ROOT_TOP: usize = 8;

/// CLI parsing is intentionally local to this diagnostic binary. The search
/// contract is defined by `search::stream`, not by an additional UCI option.
fn parse_args() -> Result<(PathBuf, String, u64), String> {
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut playouts = DEFAULT_PLAYOUTS;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--onnx" => onnx = PathBuf::from(args.next().ok_or("--onnx requires a path")?),
            "--fen" => fen = args.next().ok_or("--fen requires a quoted FEN")?,
            "--playouts" => {
                playouts = args
                    .next()
                    .ok_or("--playouts requires an integer")?
                    .parse()
                    .map_err(|_| "--playouts must be an unsigned integer")?;
            }
            "--help" | "-h" => {
                return Err("usage: stream_compare [--onnx data/x7.onnx] [--playouts 128] [--fen \"...\"]".to_owned());
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    if playouts == 0 {
        return Err("--playouts must be greater than zero".to_owned());
    }
    Ok((onnx, fen, playouts))
}

/// LC3 stream roots retain completed node statistics while in-flight visits
/// live on edges. This output samples the stable root view only.
fn print_root(label: &str, stats: StreamStats, root: &engin::search::stream::StreamNode) -> bool {
    let mut edges = root.edges().to_vec();
    edges.sort_unstable_by(|left, right| {
        right
            .completed_visits()
            .cmp(&left.completed_visits())
            .then_with(|| right.prior().partial_cmp(&left.prior()).unwrap_or(Ordering::Equal))
            .then_with(|| left.mv().to_uci().cmp(&right.mv().to_uci()))
    });
    let root_settled = edges.iter().all(|edge| edge.visits() == edge.completed_visits());
    println!(
        "{label}: completed={} collisions={} network_batches={} network_evaluations={} root_N={} root_Q={:.5} root_D={:.5} root_settled={root_settled}",
        stats.completed_playouts,
        stats.collisions,
        stats.network_batches,
        stats.network_evaluations,
        root.completed_visits(),
        root.q(),
        root.draw(),
    );
    for edge in edges.iter().take(ROOT_TOP) {
        println!(
            "  {} N={}/{} Q={:.5} P={:.5}",
            edge.mv().to_uci(),
            edge.completed_visits(),
            edge.visits(),
            edge.q(),
            edge.prior(),
        );
    }
    root_settled
}

/// Each search gets an independent cache so the comparison observes its own
/// batch lifecycle rather than cross-run cache hits.
fn load_backend(path: &PathBuf) -> Result<Arc<dyn Backend>, Box<dyn std::error::Error>> {
    let backend = OnnxBackend::from_file(path)?;
    println!("provider={}", backend.provider().name());
    Ok(Arc::new(CachingBackend::new(Box::new(backend))))
}

/// Classic and stream use different scheduling and node storage, so this
/// prints their stable root-level interpretation rather than requiring exact
/// visit-by-visit equality.
fn print_classic_root(stats: &engin::search::classic::ClassicRootStats) -> bool {
    let mut edges = stats.edges.clone();
    edges.sort_unstable_by(|left, right| {
        right
            .completed_visits
            .cmp(&left.completed_visits)
            .then_with(|| right.prior.partial_cmp(&left.prior).unwrap_or(Ordering::Equal))
            .then_with(|| left.mv.to_uci().cmp(&right.mv.to_uci()))
    });
    let root_settled =
        stats.in_flight_visits == 0 && edges.iter().all(|edge| edge.started_visits == edge.completed_visits);
    println!(
        "classic: root_N={} root_Q={:.5} root_D={:.5} root_settled={root_settled}",
        stats.completed_visits, stats.q, stats.draw,
    );
    for edge in edges.iter().take(ROOT_TOP) {
        println!(
            "  {} N={}/{} Q={:.5} P={:.5}",
            edge.mv.to_uci(),
            edge.completed_visits,
            edge.started_visits,
            edge.q,
            edge.prior,
        );
    }
    root_settled
}

/// Reference execution: serial `Gather -> Eval -> Backprop` versus persistent
/// LC3-style workers. This is not a classic-vs-stream strength comparison.
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (onnx, fen, playouts) = parse_args().map_err(std::io::Error::other)?;
    let state = GameState::from_fen_moves(&fen, &[] as &[&str])?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    let legal_moves = history.last().board().generate_legal_moves();

    println!("onnx={}", onnx.display());
    println!("fen={fen}");
    println!("requested_playouts={playouts} legal_moves={}", legal_moves.len());

    let mut classic = ClassicSearch::new(Box::new(CachingBackend::new(Box::new(OnnxBackend::from_file(&onnx)?))));
    classic.set_position(&state)?;
    let (classic_best, classic_visits) = classic.run_blocking_nodes(playouts as u32);
    let classic_root = classic.root_stats_snapshot();
    println!(
        "classic_bestmove={} returned_visits={classic_visits}",
        classic_best.to_uci()
    );
    let classic_settled = print_classic_root(&classic_root);
    drop(classic);

    // Classic's fixed-node stopper can complete a final already-submitted
    // minibatch after the requested limit. Compare stream against the classic
    // root's actual completed N, while reporting the original request above.
    let comparison_playouts = u64::from(classic_root.completed_visits);
    println!("stream_comparison_playouts={comparison_playouts}");

    let serial_backend = load_backend(&onnx)?;
    let root_eval = serial_backend.evaluate(&history, &legal_moves);
    println!(
        "root_network: W={:.5} D={:.5} L={:.5} Q={:.5}",
        (root_eval.wl + 1.0 - root_eval.d) * 0.5,
        root_eval.d,
        (1.0 - root_eval.d - root_eval.wl) * 0.5,
        root_eval.wl,
    );
    let mut serial = StreamSearch::new(serial_backend, SearchGeneration(1), Arc::clone(&history), 1.0);
    let serial_stats = serial.run_playouts(comparison_playouts)?;
    let serial_root = serial.repository().get(serial.root_key()).expect("serial root exists");
    let serial_settled = print_root("serial", serial_stats, &serial_root);

    let worker_backend = load_backend(&onnx)?;
    let mut workers = StreamWorkerPipeline::new(
        worker_backend,
        SearchGeneration(1),
        history,
        StreamWorkerConfig::default(),
    );
    let worker_stats = workers.run_playouts(comparison_playouts)?;
    let worker_root = workers
        .repository()
        .get(workers.root_key())
        .expect("worker root exists");
    let worker_settled = print_root("workers", worker_stats, &worker_root);
    workers.stop_and_join();

    let stream_passed = serial_stats.completed_playouts == comparison_playouts
        && worker_stats.completed_playouts == comparison_playouts
        && classic_settled
        && serial_settled
        && worker_settled;
    println!(
        "classic_budget_delta={}",
        i64::from(classic_root.completed_visits) - playouts as i64
    );
    println!("stream_pipeline={}", if stream_passed { "PASS" } else { "FAIL" });
    if stream_passed {
        Ok(())
    } else {
        Err("stream comparison invariant failed".into())
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("stream_compare: {error}");
        std::process::exit(2);
    }
}
