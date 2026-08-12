//! 只测 stream worker、NN batch、缓存与流水线吞吐。
//!
//! 搜索参数和根边分流诊断见 `search_benchmark`；这里刻意固定
//! `SearchParams::default()`，避免把两类实验混在一个命令中。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engin::neural::backend::{Backend, CachingBackend};
use engin::neural::onnx::OnnxBackend;
use engin::search::{QueueStats, Search, SearchConfig, SearchLimits, Stats, root_stats};
use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

struct Args {
    onnx: PathBuf,
    fen: String,
    moves: Vec<String>,
    positions: Option<PathBuf>,
    playouts: Option<u64>,
    movetime: Option<u64>,
    repeat: usize,
    gathers: Vec<usize>,
    evals: Vec<usize>,
    backprops: Vec<usize>,
    eval_batch: Option<usize>,
    cache: bool,
    warm_cache: bool,
    root_top: usize,
}

type BackendSetup = (Arc<dyn Backend>, &'static str, usize);

fn usage() -> &'static str {
    "usage: benchmark [--onnx data/x7.onnx] [--fen \"...\" | --positions data/benchmark_positions.txt] [--moves \"c3c4 h7h3 ...\"] [--playouts 20000 | --movetime 3000] \\
     [--repeat 1] [--gathers 4,8] [--evals 1,2] [--backprops 1,2] [--eval-batch 64] [--cache|--warm-cache] [--root-top 8]"
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut moves = Vec::new();
    let mut positions = None;
    let mut playouts = Some(20_000);
    let mut movetime = None;
    let mut repeat = 1;
    let mut gathers = vec![4];
    let mut evals = vec![4];
    let mut backprops = vec![1];
    let mut eval_batch = None;
    let mut cache = false;
    let mut warm_cache = false;
    let mut root_top = 8;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--onnx" => onnx = PathBuf::from(args.next().ok_or("--onnx requires a path")?),
            "--fen" => fen = args.next().ok_or("--fen requires a quoted FEN")?,
            "--positions" => positions = Some(PathBuf::from(args.next().ok_or("--positions requires a path")?)),
            "--moves" => {
                moves = args
                    .next()
                    .ok_or("--moves requires space-separated ICCS moves")?
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect();
            }
            "--playouts" => {
                playouts = Some(
                    args.next()
                        .ok_or("--playouts requires an integer")?
                        .parse()
                        .map_err(|_| "--playouts must be an unsigned integer")?,
                );
                movetime = None;
            }
            "--movetime" => {
                movetime = Some(
                    args.next()
                        .ok_or("--movetime requires milliseconds")?
                        .parse()
                        .map_err(|_| "--movetime must be an unsigned integer")?,
                );
                playouts = None;
            }
            "--repeat" => {
                repeat = args
                    .next()
                    .ok_or("--repeat requires an integer")?
                    .parse()
                    .map_err(|_| "--repeat must be an unsigned integer")?
            }
            "--gathers" => gathers = parse_list(&args.next().ok_or("--gathers requires list like 4,8")?)?,
            "--evals" => evals = parse_list(&args.next().ok_or("--evals requires list like 1,2")?)?,
            "--backprops" => backprops = parse_list(&args.next().ok_or("--backprops requires list like 1,2")?)?,
            "--eval-batch" => {
                eval_batch = Some(
                    args.next()
                        .ok_or("--eval-batch requires an integer")?
                        .parse()
                        .map_err(|_| "--eval-batch must be an unsigned integer")?,
                )
            }
            "--cache" => cache = true,
            "--warm-cache" => {
                cache = true;
                warm_cache = true;
            }
            "--root-top" => {
                root_top = args
                    .next()
                    .ok_or("--root-top requires an integer")?
                    .parse()
                    .map_err(|_| "--root-top must be an unsigned integer")?
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if playouts == Some(0) || movetime == Some(0) || repeat == 0 || eval_batch == Some(0) || root_top == 0 {
        return Err("playouts, movetime, repeat, eval-batch, and root-top must be positive".into());
    }
    if positions.is_some() && !moves.is_empty() {
        return Err("--positions cannot be combined with --moves".into());
    }
    for (name, values) in [
        ("--gathers", &gathers),
        ("--evals", &evals),
        ("--backprops", &backprops),
    ] {
        if values.contains(&0) {
            return Err(format!("{name} entries must be > 0"));
        }
    }
    Ok(Args {
        onnx,
        fen,
        moves,
        positions,
        playouts,
        movetime,
        repeat,
        gathers,
        evals,
        backprops,
        eval_batch,
        cache,
        warm_cache,
        root_top,
    })
}

fn parse_list(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|part| part.trim().parse().map_err(|_| format!("invalid list entry: {part}")))
        .collect()
}

/// `data/benchmark_positions.txt` 的 `名称 | FEN` 格式。
fn load_positions(args: &Args) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let Some(path) = &args.positions else {
        return Ok(vec![("input".into(), args.fen.clone())]);
    };
    let text = std::fs::read_to_string(path)?;
    let mut positions = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, fen) = line
            .split_once('|')
            .ok_or_else(|| format!("{}:{} expected `name | FEN`", path.display(), line_index + 1))?;
        if name.trim().is_empty() || fen.trim().is_empty() {
            return Err(format!("{}:{} empty name or FEN", path.display(), line_index + 1).into());
        }
        positions.push((name.trim().to_owned(), fen.trim().to_owned()));
    }
    if positions.is_empty() {
        return Err(format!("{} has no positions", path.display()).into());
    }
    Ok(positions)
}

fn make_backend(path: &PathBuf, cache: bool) -> Result<BackendSetup, Box<dyn std::error::Error>> {
    let onnx = OnnxBackend::from_file(path)?;
    let provider = onnx.provider().name();
    let recommended = onnx.attributes().recommended_batch_size;
    let backend: Arc<dyn Backend> = if cache {
        Arc::new(CachingBackend::new(Box::new(onnx)))
    } else {
        Arc::new(onnx)
    };
    Ok((backend, provider, recommended))
}

fn average_wait_us(queue: QueueStats) -> f64 {
    if queue.samples == 0 {
        0.0
    } else {
        queue.total_wait_ns as f64 / queue.samples as f64 / 1e3
    }
}

fn collision_depths(stats: &Stats) -> String {
    let depths: Vec<_> = stats
        .collisions_by_depth
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(depth, count)| format!("d{depth}:{count}"))
        .collect();
    if depths.is_empty() {
        "-".into()
    } else {
        depths.join(",")
    }
}

fn print_root_candidates(search: &Search, root_is_black: bool, top: usize) {
    let Some(root) = root_stats(search.repository(), search.root_key()) else {
        return;
    };
    let mut edges = root.edges;
    edges.sort_unstable_by(|left, right| {
        right
            .completed_visits
            .cmp(&left.completed_visits)
            .then_with(|| right.started_visits.cmp(&left.started_visits))
            .then_with(|| right.prior.total_cmp(&left.prior))
    });
    println!(
        "    root candidates top {}/{}: move       P    done flight       Q",
        edges.len().min(top),
        edges.len()
    );
    for edge in edges.into_iter().take(top) {
        let mv = if root_is_black { edge.mv.flip() } else { edge.mv };
        println!(
            "                       {:<6} {:>7.4} {:>7} {:>6} {:>7.4}",
            mv.to_uci(),
            edge.prior,
            edge.completed_visits,
            edge.started_visits.saturating_sub(edge.completed_visits),
            edge.q
        );
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if !args.onnx.is_file() {
        return Err(format!("onnx missing: {}", args.onnx.display()).into());
    }
    let (_, provider, recommended) = make_backend(&args.onnx, false)?;
    let target_batch = args.eval_batch.unwrap_or(recommended).max(1);
    let positions = load_positions(&args)?;
    println!(
        "onnx={} provider={} cache={} warm_cache={} recommended_batch={} target_batch={} budget={} repeat={} worker_matrix={} positions={}",
        args.onnx.display(),
        provider,
        args.cache,
        args.warm_cache,
        recommended,
        target_batch,
        args.playouts
            .map(|n| format!("playouts={n}"))
            .unwrap_or_else(|| format!("movetime={}ms", args.movetime.unwrap_or(0))),
        args.repeat,
        args.gathers.len() * args.evals.len() * args.backprops.len(),
        positions.len(),
    );
    println!(
        "note: each run has a fresh graph; --warm-cache alone reuses one NN cache between repeats; hit is normal cache hits; q_* is average queue delay in us"
    );
    println!("  G   E   B run       ms      nps      eps    done    hit  coll%    peak  root%    q_g    q_e    q_n");
    let mut generation = 0;
    for (name, fen) in positions {
        let state = GameState::from_fen_moves(&fen, &args.moves)?;
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        println!("position: {name}");
        for &gather_workers in &args.gathers {
            for &eval_workers in &args.evals {
                for &backprop_workers in &args.backprops {
                    let warm_backend = args.warm_cache.then(|| make_backend(&args.onnx, true)).transpose()?;
                    for run_index in 1..=args.repeat {
                        generation += 1;
                        let backend = match &warm_backend {
                            Some((backend, _, _)) => Arc::clone(backend),
                            None => make_backend(&args.onnx, args.cache)?.0,
                        };
                        let mut search = Search::new(
                            backend,
                            generation,
                            Arc::clone(&history),
                            SearchConfig {
                                eval_batch_size: target_batch,
                                gather_workers,
                                eval_workers,
                                backprop_workers,
                                ..SearchConfig::default()
                            },
                        );
                        let started = Instant::now();
                        let stats = search.run_with_limits(SearchLimits {
                            max_playouts: args.playouts,
                            deadline: args.movetime.map(|ms| Instant::now() + Duration::from_millis(ms)),
                        })?;
                        let seconds = started.elapsed().as_secs_f64();
                        let total = stats.completed_playouts + stats.collisions;
                        let collision_rate = if total == 0 {
                            0.0
                        } else {
                            stats.collisions as f64 * 100.0 / total as f64
                        };
                        let root_share = root_stats(search.repository(), search.root_key())
                            .and_then(|root| {
                                root.edges
                                    .into_iter()
                                    .map(|edge| edge.completed_visits)
                                    .max()
                                    .map(|best| best as f64 * 100.0 / root.completed_visits.max(1) as f64)
                            })
                            .unwrap_or(0.0);
                        println!(
                            "{:>3} {:>3} {:>3} {:>3} {:>8.1} {:>8.0} {:>8.0} {:>7} {:>6} {:>6.1} {:>7} {:>6.1} {:>6.1} {:>6.1} {:>6.1}",
                            gather_workers,
                            eval_workers,
                            backprop_workers,
                            run_index,
                            seconds * 1e3,
                            stats.completed_playouts as f64 / seconds,
                            stats.network_evaluations as f64 / seconds,
                            stats.completed_playouts,
                            stats.cache_hits,
                            collision_rate,
                            stats.peak_in_flight,
                            root_share,
                            average_wait_us(stats.gather_queue),
                            average_wait_us(stats.eval_queue),
                            average_wait_us(stats.nn_queue)
                        );
                        println!(
                            "    batch avg={:.2} max={} q_backprop={:.1}us submitted={} collision_depths={}",
                            if stats.network_batches == 0 {
                                0.0
                            } else {
                                stats.network_evaluations as f64 / stats.network_batches as f64
                            },
                            stats.network_batch_size_max,
                            average_wait_us(stats.backprop_queue),
                            stats.submitted_playouts,
                            collision_depths(&stats)
                        );
                        print_root_candidates(&search, root_is_black, args.root_top);
                        search.stop_and_finish();
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("benchmark: {error}");
        std::process::exit(2);
    }
}
