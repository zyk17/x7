//! 用 Gather × Eval × Backprop 组合测量吞吐与 collision。
//!
//! NN 是独立 ONNX 线程；Eval 负责终局、缓存与编码；NN 合并队列中的 tensor，
//! 最大 batch 为 `--eval-batch`。默认 worker 为 `4/4/1`。这不是 UCI 路径。
//!
//! ```text
//! cargo run -p engin --release --bin benchmark -- --cache --playouts 20000
//! cargo run -p engin --release --bin benchmark -- --movetime 3000 --repeat 3
//! cargo run -p engin --release --bin benchmark -- --gathers 4,8 --evals 1,2 --backprops 1,2 --cache
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engin::neural::backend::{Backend, CachingBackend};
use engin::neural::onnx::OnnxBackend;
use engin::search::{QueueStats, Search, SearchConfig, SearchGeneration, SearchLimits, Stats, root_stats};
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
    root_top: usize,
}

fn usage() -> &'static str {
    "usage: benchmark [--onnx data/x7.onnx] [--fen \"...\" | --positions data/benchmark_positions.txt] [--moves \"c3c4 h7h3 ...\"] [--playouts 20000 | --movetime 3000] \
     [--repeat 1] [--gathers 4,8] [--evals 1,2] [--backprops 1,2] [--eval-batch 64] [--cache] [--root-top 8]"
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
    let mut backprops = vec![1]; // 除非遇到显示的back瓶颈, 它暂时很难成为瓶颈
    let mut eval_batch = None;
    let mut cache = false;
    let mut root_top = 8;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--onnx" => onnx = PathBuf::from(args.next().ok_or("--onnx requires a path")?),
            "--fen" => fen = args.next().ok_or("--fen requires a quoted FEN")?,
            "--positions" => {
                positions = Some(PathBuf::from(args.next().ok_or("--positions requires a path")?));
            }
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
                    .map_err(|_| "--repeat must be an unsigned integer")?;
            }
            "--gathers" => {
                gathers = parse_list(&args.next().ok_or("--gathers requires list like 4,8")?)?;
            }
            "--evals" => {
                evals = parse_list(&args.next().ok_or("--evals requires list like 1,2")?)?;
            }
            "--backprops" => {
                backprops = parse_list(&args.next().ok_or("--backprops requires list like 1,2")?)?;
            }
            "--eval-batch" => {
                eval_batch = Some(
                    args.next()
                        .ok_or("--eval-batch requires an integer")?
                        .parse()
                        .map_err(|_| "--eval-batch must be an unsigned integer")?,
                );
            }
            "--cache" => cache = true,
            "--root-top" => {
                root_top = args
                    .next()
                    .ok_or("--root-top requires an integer")?
                    .parse()
                    .map_err(|_| "--root-top must be an unsigned integer")?;
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if playouts == Some(0) {
        return Err("--playouts must be > 0".into());
    }
    if movetime == Some(0) {
        return Err("--movetime must be > 0".into());
    }
    if repeat == 0 {
        return Err("--repeat must be > 0".into());
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
    if eval_batch == Some(0) {
        return Err("--eval-batch must be > 0".into());
    }
    if root_top == 0 {
        return Err("--root-top must be > 0".into());
    }
    if positions.is_some() && !moves.is_empty() {
        return Err("--positions cannot be combined with --moves".into());
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
        root_top,
    })
}

/// 读取仓库的 `名称 | FEN` benchmark 局面文件；注释和空行不参与运行。
/// 参考：`data/benchmark_positions.txt` 的文件格式。
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
        let name = name.trim();
        let fen = fen.trim();
        if name.is_empty() || fen.is_empty() {
            return Err(format!("{}:{} empty name or FEN", path.display(), line_index + 1).into());
        }
        positions.push((name.to_owned(), fen.to_owned()));
    }
    if positions.is_empty() {
        return Err(format!("{} has no positions", path.display()).into());
    }
    Ok(positions)
}

/// 格式化 benchmark 专用 stream 遥测中的一项平均队列延迟。
///
/// 参考：LC3 Overview 的 “Stats Collection”。
fn average_wait_us(queue: QueueStats) -> f64 {
    if queue.samples == 0 {
        0.0
    } else {
        queue.total_wait_ns as f64 / queue.samples as f64 / 1e3
    }
}

/// 格式化非零 collision bucket，且不加宽结果表。
///
/// 参考：LC3 Overview 的 “Stats Collection”。
fn collision_depths(stats: &Stats) -> String {
    let buckets: Vec<String> = stats
        .collisions_by_depth
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(depth, count)| format!("d{depth}:{count}"))
        .collect();
    if buckets.is_empty() {
        "-".into()
    } else {
        buckets.join(",")
    }
}

/// 打印 root 候选，诊断关键应手是否因低 P、低 completed N 或错误 Q 被忽略。
/// `RootEdgeStats` 的 Q 是走该 root 着一方视角；started - completed 是 edge-local
/// reservation/in-flight。两者来自 stream 的只读 root snapshot，不影响正式 UCI。
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
    let shown = edges.len().min(top);
    println!(
        "    root candidates top {shown}/{}: move       P    done flight       Q",
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
            edge.q,
        );
    }
}

fn parse_list(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid list entry: {part}"))
        })
        .collect()
}

type BackendSetup = (Arc<dyn Backend>, &'static str, usize);

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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if !args.onnx.is_file() {
        return Err(format!("onnx missing: {}", args.onnx.display()).into());
    }
    let (_, provider, recommended) = make_backend(&args.onnx, false)?;
    let target_batch = args.eval_batch.unwrap_or(recommended).max(1);
    let cells = args.gathers.len() * args.evals.len() * args.backprops.len();
    let positions = load_positions(&args)?;
    println!(
        "onnx={} provider={} cache={} recommended_batch={} target_batch={} budget={} repeat={} matrix={} positions={}",
        args.onnx.display(),
        provider,
        args.cache,
        recommended,
        target_batch,
        args.playouts
            .map(|n| format!("playouts={n}"))
            .unwrap_or_else(|| format!("movetime={}ms", args.movetime.unwrap_or(0))),
        args.repeat,
        cells,
        positions.len(),
    );
    println!("note: fresh backend/cache per run; hit is normal cache hits; q_* is average queue delay in us");
    println!(
        "{:>3} {:>3} {:>3} {:>3} {:>8} {:>8} {:>8} {:>7} {:>6} {:>6} {:>7} {:>6} {:>6} {:>6} {:>6}",
        "G", "E", "B", "run", "ms", "nps", "eps", "done", "hit", "coll%", "peak", "root%", "q_g", "q_e", "q_n"
    );

    if !args.moves.is_empty() {
        println!("history: startpos/fen + {} moves", args.moves.len());
    }
    let mut generation = 0u64;

    for (position_name, fen) in positions {
        let state = GameState::from_fen_moves(&fen, &args.moves)?;
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        println!("position: {position_name}");
        for &gather_workers in &args.gathers {
            for &eval_workers in &args.evals {
                for &backprop_workers in &args.backprops {
                    for run_index in 1..=args.repeat {
                        generation += 1;
                        let (backend, _, _) = make_backend(&args.onnx, args.cache)?;
                        let mut search = Search::new(
                            backend,
                            SearchGeneration(generation),
                            Arc::clone(&history),
                            SearchConfig {
                                eval_batch_size: target_batch,
                                gather_workers,
                                eval_workers,
                                backprop_workers,
                                benchmark_telemetry: true,
                                ..SearchConfig::default()
                            },
                        );
                        let t0 = Instant::now();
                        let stats = search.run_with_limits(SearchLimits {
                            max_playouts: args.playouts,
                            deadline: args.movetime.map(|ms| Instant::now() + Duration::from_millis(ms)),
                        })?;
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        let seconds = ms / 1e3;
                        let nps = if seconds > 0.0 {
                            stats.completed_playouts as f64 / seconds
                        } else {
                            0.0
                        };
                        let eps = if seconds > 0.0 {
                            stats.network_evaluations as f64 / seconds
                        } else {
                            0.0
                        };
                        let total = stats.completed_playouts + stats.collisions;
                        let collision_rate = if total > 0 {
                            stats.collisions as f64 * 100.0 / total as f64
                        } else {
                            0.0
                        };
                        let root_share = root_stats(search.repository(), search.root_key())
                            .and_then(|root| {
                                root.edges
                                    .into_iter()
                                    .map(|edge| edge.completed_visits)
                                    .max()
                                    .map(|best| (best, root.completed_visits))
                            })
                            .map(|(best, total)| best as f64 * 100.0 / total.max(1) as f64)
                            .unwrap_or(0.0);
                        println!(
                            "{:>3} {:>3} {:>3} {:>3} {:>8.1} {:>8.0} {:>8.0} {:>7} {:>6} {:>6.1} {:>7} {:>6.1} {:>6.1} {:>6.1} {:>6.1}",
                            gather_workers,
                            eval_workers,
                            backprop_workers,
                            run_index,
                            ms,
                            nps,
                            eps,
                            stats.completed_playouts,
                            stats.cache_hits,
                            collision_rate,
                            stats.peak_in_flight,
                            root_share,
                            average_wait_us(stats.gather_queue),
                            average_wait_us(stats.eval_queue),
                            average_wait_us(stats.nn_queue),
                        );
                        println!(
                            "    batch avg={:.2} max={} q_backprop={:.1}us submitted={} collision_depths={}",
                            if stats.network_batches > 0 {
                                stats.network_evaluations as f64 / stats.network_batches as f64
                            } else {
                                0.0
                            },
                            stats.network_batch_size_max,
                            average_wait_us(stats.backprop_queue),
                            stats.submitted_playouts,
                            collision_depths(&stats),
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
