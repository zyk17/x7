//! Classic ↔ stream diagnostic compare (not a UCI path).
//!
//! Compares fixed-visit root structure and classic-aligned bestmove ranking
//! (N then Q then P; terminal win > loss). Scheduling may change per-edge N/Q;
//! failures are unsettled roots, missing bestmove, or budget mismatches.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engin::neural::backend::{Backend, CachingBackend, UniformBackend};
use engin::neural::onnx::OnnxBackend;
use engin::search::classic::{ClassicRootStats, ClassicSearch};
use engin::search::stream::{
    best_move, principal_variation, root_settled, root_stats, SearchGeneration, SearchLimits,
    Stats, SearchConfig, Search,
};
use engin::SearchBase;
use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

const DEFAULT_PLAYOUTS: u64 = 128;
const ROOT_TOP: usize = 8;

struct Args {
    onnx: Option<PathBuf>,
    fen: String,
    playouts: u64,
    stream_movetime: Option<Duration>,
    skip_classic: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = None;
    let mut fen = STARTPOS_FEN.to_owned();
    let mut playouts = DEFAULT_PLAYOUTS;
    let mut stream_movetime = None;
    let mut skip_classic = false;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--onnx" => onnx = Some(PathBuf::from(args.next().ok_or("--onnx requires a path")?)),
            "--uniform" => onnx = None,
            "--fen" => fen = args.next().ok_or("--fen requires a quoted FEN")?,
            "--playouts" => {
                playouts = args
                    .next()
                    .ok_or("--playouts requires an integer")?
                    .parse()
                    .map_err(|_| "--playouts must be an unsigned integer")?;
            }
            "--movetime-ms" => {
                let millis = args
                    .next()
                    .ok_or("--movetime-ms requires an integer")?
                    .parse::<u64>()
                    .map_err(|_| "--movetime-ms must be an unsigned integer")?;
                stream_movetime = Some(Duration::from_millis(millis));
            }
            "--skip-classic" => skip_classic = true,
            "--help" | "-h" => {
                return Err(
                    "usage: stream_compare [--onnx data/x7.onnx | --uniform] [--playouts 128] [--movetime-ms N] [--skip-classic] [--fen \"...\"]"
                        .to_owned(),
                );
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    if playouts == 0 {
        return Err("--playouts must be greater than zero".to_owned());
    }
    Ok(Args {
        onnx,
        fen,
        playouts,
        stream_movetime,
        skip_classic,
    })
}

fn load_backend(onnx: &Option<PathBuf>) -> Result<Arc<dyn Backend>, Box<dyn std::error::Error>> {
    match onnx {
        Some(path) => {
            let backend = OnnxBackend::from_file(path)?;
            println!("provider={}", backend.provider().name());
            Ok(Arc::new(CachingBackend::new(Box::new(backend))))
        }
        None => {
            println!("provider=uniform");
            Ok(Arc::new(UniformBackend::default()))
        }
    }
}

fn print_classic_root(stats: &ClassicRootStats, best: Move) -> bool {
    let mut edges = stats.edges.clone();
    edges.sort_unstable_by(|left, right| {
        right
            .completed_visits
            .cmp(&left.completed_visits)
            .then_with(|| right.q.partial_cmp(&left.q).unwrap_or(Ordering::Equal))
            .then_with(|| right.prior.partial_cmp(&left.prior).unwrap_or(Ordering::Equal))
            .then_with(|| left.mv.to_uci().cmp(&right.mv.to_uci()))
    });
    let root_settled =
        stats.in_flight_visits == 0 && edges.iter().all(|edge| edge.started_visits == edge.completed_visits);
    println!(
        "classic: bestmove={} root_N={} root_Q={:.5} root_D={:.5} root_settled={root_settled}",
        best.to_uci(),
        stats.completed_visits,
        stats.q,
        stats.draw,
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

struct RootReport {
    settled: bool,
    best: Option<Move>,
    legal: BTreeSet<String>,
}

fn print_root(
    label: &str,
    stats: Stats,
    repository: &engin::search::stream::NodeRepository,
    root_key: engin::search::stream::NodeKey,
    root_is_black: bool,
) -> Result<RootReport, Box<dyn std::error::Error>> {
    let root = root_stats(repository, root_key).ok_or("missing stream root stats")?;
    let mut edges = root.edges.clone();
    edges.sort_unstable_by(|left, right| {
        right
            .completed_visits
            .cmp(&left.completed_visits)
            .then_with(|| right.q.partial_cmp(&left.q).unwrap_or(Ordering::Equal))
            .then_with(|| right.prior.partial_cmp(&left.prior).unwrap_or(Ordering::Equal))
            .then_with(|| left.mv.to_uci().cmp(&right.mv.to_uci()))
    });
    let settled = root_settled(repository, root_key);
    let best = best_move(repository, root_key, root_is_black);
    let pv = principal_variation(repository, root_key, root_is_black);
    let legal: BTreeSet<String> = edges.iter().map(|edge| edge.mv.to_uci()).collect();
    println!(
        "{label}: completed={} collisions={} network_batches={} network_evaluations={} root_N={} root_Q={:.5} root_D={:.5} root_settled={settled}",
        stats.completed_playouts,
        stats.collisions,
        stats.network_batches,
        stats.network_evaluations,
        root.completed_visits,
        root.q,
        root.draw,
    );
    println!(
        "{label}_bestmove={} pv={}",
        best.map(|mv| mv.to_uci()).unwrap_or_else(|| "-".into()),
        pv.iter().map(|mv| mv.to_uci()).collect::<Vec<_>>().join(" "),
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
    Ok(RootReport {
        settled,
        best,
        legal,
    })
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(std::io::Error::other)?;
    let state = GameState::from_fen_moves(&args.fen, &[] as &[&str])?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    let legal_moves = history.last().board().generate_legal_moves();
    let board_legal: BTreeSet<String> = legal_moves.iter().map(|mv| mv.to_uci()).collect();

    println!(
        "backend={} fen={}",
        args.onnx
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "uniform".into()),
        args.fen
    );
    println!("requested_playouts={} legal_moves={}", args.playouts, legal_moves.len());

    let mut comparison_playouts = args.playouts;
    let mut classic_best = None;
    let mut classic_legal = BTreeSet::new();
    let mut classic_settled = true;

    if !args.skip_classic {
        let classic_boxed: Box<dyn Backend> = match &args.onnx {
            Some(path) => Box::new(CachingBackend::new(Box::new(OnnxBackend::from_file(path)?))),
            None => Box::new(UniformBackend::default()),
        };
        let mut classic = ClassicSearch::new(classic_boxed);
        classic.set_position(&state)?;
        let started = Instant::now();
        let (best, classic_visits) = classic.run_blocking_nodes(args.playouts as u32);
        let classic_root = classic.root_stats_snapshot();
        println!(
            "classic_elapsed_ms={} returned_visits={classic_visits}",
            started.elapsed().as_millis()
        );
        classic_settled = print_classic_root(&classic_root, best);
        classic_best = Some(best);
        classic_legal = classic_root.edges.iter().map(|edge| edge.mv.to_uci()).collect();
        comparison_playouts = u64::from(classic_root.completed_visits);
        println!(
            "classic_budget_delta={}",
            i64::from(classic_root.completed_visits) - args.playouts as i64
        );
        drop(classic);
    }

    println!("stream_comparison_playouts={comparison_playouts}");

    let root_is_black = history.is_black_to_move();
    let stream_backend = load_backend(&args.onnx)?;
    let mut search = Search::new(
        stream_backend,
        SearchGeneration(1),
        history,
        SearchConfig::default(),
    );
    let stream_started = Instant::now();
    let (stream_stats, stream_budget_satisfied) = match args.stream_movetime {
        Some(movetime) => {
            let stats = search.run_with_limits(SearchLimits {
                max_playouts: None,
                deadline: Some(stream_started + movetime),
            })?;
            println!(
                "stream_movetime_ms={} elapsed_ms={} completed_playouts={}",
                movetime.as_millis(),
                stream_started.elapsed().as_millis(),
                stats.completed_playouts,
            );
            (stats, stats.completed_playouts > 0)
        }
        None => {
            let stats = search.run_playouts(comparison_playouts)?;
            println!("stream_elapsed_ms={}", stream_started.elapsed().as_millis());
            (stats, stats.completed_playouts == comparison_playouts)
        }
    };
    let stream_report = print_root(
        "stream",
        stream_stats,
        search.repository(),
        search.root_key(),
        root_is_black,
    )?;
    let stream_settled = stream_report.settled;
    let stream_best = stream_report.best;
    let stream_legal = stream_report.legal;
    search.stop_and_join();

    let legal_ok = stream_legal == board_legal
        && (args.skip_classic || classic_legal.len() == board_legal.len());
    let bestmove_ok = stream_best.is_some() && (args.skip_classic || classic_best.is_some());

    if let (Some(classic), Some(stream)) = (classic_best, stream_best) {
        println!(
            "bestmove_compare classic={} stream={} classic_eq_stream={}",
            classic.to_uci(),
            stream.to_uci(),
            classic == stream,
        );
    }

    let passed = stream_budget_satisfied
        && classic_settled
        && stream_settled
        && legal_ok
        && bestmove_ok;
    println!("legal_moves_match={legal_ok}");
    println!("stream_compare={}", if passed { "PASS" } else { "FAIL" });
    if passed {
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
