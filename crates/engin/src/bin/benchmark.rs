//! 只测 stream worker、NN batch、缓存与流水线吞吐。
//!
//! 搜索参数和根边分流诊断见 `search_benchmark`；这里固定默认 PUCT，只允许切换
//! pending-work 实验，观察它对 batch/collision 的影响。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engin::neural::backend::{Backend, CachingBackend, EvalResult};
use engin::neural::onnx::OnnxBackend;
use engin::neural::{EncodedBatch, FillEmptyHistory, encode_position_input_planes, eval_result_from_encoded_row};
use engin::search::{
    BenchObserver, BenchStats, NodeId, QueueStats, RootEdgeStats, Search, SearchConfig, SearchLimits, SearchParams,
    Stats, root_stats,
};
use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

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
    eval_batch: Option<usize>,
    nn_window: f32,
    virtual_mean_fpu_scale: f32,
    root_top: usize,
    trace: Vec<u64>,
    show_collision_dist: bool,
    tree_depth: Option<usize>,
    tree_top: usize,
}

type BackendSetup = (Arc<dyn Backend>, &'static str, usize);

fn usage() -> &'static str {
    "usage: benchmark [--onnx data/x7.onnx] [--fen \"...\" | --positions data/benchmark_positions.txt] [--moves \"c3c4 h7h3 ...\"] [--playouts 20000 | --movetime 3000] \\
     [--repeat 1] [--gathers 2,4] [--evals 4,6] [--eval-batch 64] [--nn-window 2.25] [--virtual-mean-fpu-scale 1.0] \\
     [--root-top 8] [--trace 128,256,512] [--collision-dist] [--tree-depth 4] [--tree-top 4]"
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut moves = Vec::new();
    let mut positions = None;
    let mut playouts = Some(20_000);
    let mut movetime = None;
    let mut repeat = 1;
    let mut gathers = vec![3];
    let mut evals = vec![5];
    let mut eval_batch = None;
    let mut nn_window = 2.25f32;
    let mut virtual_mean_fpu_scale = 1.0f32;
    let mut root_top = 8;
    let mut trace = Vec::new();
    let mut show_collision_dist = false;
    let mut tree_depth = None;
    let mut tree_top = 4;
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
            "--eval-batch" => {
                eval_batch = Some(
                    args.next()
                        .ok_or("--eval-batch requires an integer")?
                        .parse()
                        .map_err(|_| "--eval-batch must be an unsigned integer")?,
                )
            }
            "--nn-window" => {
                nn_window = args
                    .next()
                    .ok_or("--nn-window requires a number")?
                    .parse()
                    .map_err(|_| "--nn-window must be a number")?;
            }
            "--virtual-mean-fpu-scale" => {
                let scale = args
                    .next()
                    .ok_or("--virtual-mean-fpu-scale needs a value")?
                    .parse::<f32>()
                    .map_err(|_| "--virtual-mean-fpu-scale must be a number")?;
                if !scale.is_finite() || scale < 0.0 {
                    return Err("--virtual-mean-fpu-scale must be finite and non-negative".into());
                }
                virtual_mean_fpu_scale = scale;
            }
            "--root-top" => {
                root_top = args
                    .next()
                    .ok_or("--root-top requires an integer")?
                    .parse()
                    .map_err(|_| "--root-top must be an unsigned integer")?
            }
            "--trace" => trace = parse_u64_list(&args.next().ok_or("--trace requires a list")?)?,
            "--collision-dist" => show_collision_dist = true,
            "--tree-depth" => {
                tree_depth = Some(
                    args.next()
                        .ok_or("--tree-depth requires an integer")?
                        .parse()
                        .map_err(|_| "--tree-depth must be an unsigned integer")?,
                );
            }
            "--tree-top" => {
                tree_top = args
                    .next()
                    .ok_or("--tree-top requires an integer")?
                    .parse()
                    .map_err(|_| "--tree-top must be an unsigned integer")?
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if playouts == Some(0) || movetime == Some(0) || repeat == 0 || eval_batch == Some(0) || root_top == 0 {
        return Err("playouts, movetime, repeat, eval-batch, and root-top must be positive".into());
    }
    if tree_top == 0 || tree_depth == Some(0) {
        return Err("--tree-top and --tree-depth must be positive when set".into());
    }
    if !nn_window.is_finite() || nn_window <= 0.0 {
        return Err("--nn-window must be finite and > 0".into());
    }
    if positions.is_some() && !moves.is_empty() {
        return Err("--positions cannot be combined with --moves".into());
    }
    if !trace.is_empty() {
        let Some(playouts) = playouts else {
            return Err("--trace requires --playouts".into());
        };
        if trace.windows(2).any(|pair| pair[0] >= pair[1]) || trace.iter().any(|&value| value == 0 || value > playouts)
        {
            return Err("--trace milestones must be strictly increasing and within --playouts".into());
        }
    }
    for (name, values) in [("--gathers", &gathers), ("--evals", &evals)] {
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
        eval_batch,
        nn_window,
        virtual_mean_fpu_scale,
        root_top,
        trace,
        show_collision_dist,
        tree_depth,
        tree_top,
    })
}

fn parse_list(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|part| part.trim().parse().map_err(|_| format!("invalid list entry: {part}")))
        .collect()
}

fn parse_u64_list(text: &str) -> Result<Vec<u64>, String> {
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

/// 预热该局面 NN，并留下 root 的初始 P / Q(V) / M 供搜后对照。
struct RootNnProbe {
    wl: f32,
    plies_left: f32,
    /// 合法着 policy，与 `legal` 对齐。
    policies: Vec<(Move, f32)>,
}

fn make_backend(path: &PathBuf) -> Result<BackendSetup, Box<dyn std::error::Error>> {
    let onnx = OnnxBackend::from_file(path)?;
    let provider = onnx.provider().name();
    let recommended = onnx.attributes().recommended_batch_size;
    let backend: Arc<dyn Backend> = Arc::new(CachingBackend::new(Box::new(onnx)));
    Ok((backend, provider, recommended))
}

fn evaluate_root(
    backend: &dyn Backend,
    history: &PositionHistory,
) -> Result<Arc<EvalResult>, Box<dyn std::error::Error>> {
    let legal = history.last().board().generate_legal_moves();
    let sample = encode_position_input_planes(history, FillEmptyHistory::FenOnly);
    let mut logits = Vec::new();
    let mut wdl = Vec::new();
    let mut moves_left = Vec::new();
    backend.infer_input_planes_into(&[sample], &mut logits, &mut wdl, &mut moves_left)?;
    let output = EncodedBatch {
        logits,
        wdl,
        moves_left,
    };
    Ok(eval_result_from_encoded_row(&output, 0, &legal)?)
}

/// 对 bench 局面做 NN 预热（ORT/TRT），并返回该局面 root 的初始 P/V/M。
fn warmup_position(
    backend: &dyn Backend,
    history: &PositionHistory,
    batch: usize,
) -> Result<RootNnProbe, Box<dyn std::error::Error>> {
    let planes = encode_position_input_planes(history, FillEmptyHistory::FenOnly);
    let samples = vec![planes; batch.max(1)];
    let mut logits = Vec::new();
    let mut wdl = Vec::new();
    let mut moves_left = Vec::new();
    let started = Instant::now();
    for _ in 0..3 {
        backend.infer_input_planes_into(&samples, &mut logits, &mut wdl, &mut moves_left)?;
    }
    if batch > 1 {
        let one = &samples[..1];
        for _ in 0..2 {
            backend.infer_input_planes_into(one, &mut logits, &mut wdl, &mut moves_left)?;
        }
    }
    let eval = evaluate_root(backend, history)?;
    backend.clear_cache();
    let legal = history.last().board().generate_legal_moves();
    let policies: Vec<(Move, f32)> = legal.into_iter().zip(eval.policies.iter().copied()).collect();
    println!(
        "nn warmup: batch={batch} rounds=3 (+batch1) Q={:.4} M={:.1} took {:.1}ms",
        eval.wl,
        eval.plies_left,
        started.elapsed().as_secs_f64() * 1e3
    );
    Ok(RootNnProbe {
        wl: eval.wl,
        plies_left: eval.plies_left,
        policies,
    })
}

fn average_wait_us(queue: QueueStats) -> f64 {
    if queue.samples == 0 {
        0.0
    } else {
        queue.total_wait_ns as f64 / queue.samples as f64 / 1e3
    }
}

fn max_wait_us(queue: QueueStats) -> f64 {
    queue.max_wait_ns as f64 / 1e3
}

fn sorted_root_edges(search: &Search<BenchObserver>) -> Option<Vec<RootEdgeStats>> {
    let root = root_stats(search.arena(), search.root_id())?;
    let mut edges = root.edges;
    edges.sort_unstable_by(|left, right| {
        right
            .completed_visits
            .cmp(&left.completed_visits)
            .then_with(|| right.started_visits.cmp(&left.started_visits))
            .then_with(|| right.prior.total_cmp(&left.prior))
    });
    Some(edges)
}

fn format_batch_dist(batches_by_size: &[u64]) -> String {
    let parts: Vec<_> = batches_by_size
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, count)| **count > 0)
        .map(|(size, count)| format!("{size}×{count}"))
        .collect();
    if parts.is_empty() { "-".into() } else { parts.join(" ") }
}

fn format_collision_dist(counts: &[u64]) -> String {
    let parts: Vec<_> = counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(depth, count)| format!("d{depth}:{count}"))
        .collect();
    if parts.is_empty() { "-".into() } else { parts.join(" ") }
}

fn print_queue(label: &str, queue: QueueStats) {
    println!(
        "  {label:<8} samples={:<7} avg_us={:>7.1} max_us={:>7.1}",
        queue.samples,
        average_wait_us(queue),
        max_wait_us(queue)
    );
}

fn print_root_block(
    search: &Search<BenchObserver>,
    root_is_black: bool,
    top: usize,
    heading: &str,
    nn: Option<&RootNnProbe>,
) {
    let Some(edges) = sorted_root_edges(search) else {
        return;
    };
    let root = root_stats(search.arena(), search.root_id());
    let root_n = edges
        .iter()
        .map(|edge| edge.completed_visits as u64)
        .sum::<u64>()
        .max(1);
    let top1 = edges
        .first()
        .map(|edge| edge.completed_visits as f64 * 100.0 / root_n as f64)
        .unwrap_or(0.0);
    let top3 = edges
        .iter()
        .take(3)
        .map(|edge| edge.completed_visits as u64)
        .sum::<u64>() as f64
        * 100.0
        / root_n as f64;
    println!("{heading}");
    if let Some(nn) = nn {
        println!("  nn:     Q={:.4} M={:.1}", nn.wl, nn.plies_left);
    }
    if let Some(root) = &root {
        println!(
            "  search: Q={:.4} N={}  concentration top1={top1:.1}% top3={top3:.1}%",
            root.q, root.completed_visits
        );
    }
    println!(
        "  candidates top {}/{} (by search N; P is nn prior)",
        edges.len().min(top),
        edges.len()
    );
    println!("  move       P    done flight       Q");
    for edge in edges.into_iter().take(top) {
        let mv = if root_is_black { edge.mv.flip() } else { edge.mv };
        println!(
            "  {:<6} {:>7.4} {:>7} {:>6} {:>7.4}",
            mv.to_uci(),
            edge.prior,
            edge.completed_visits,
            edge.started_visits.saturating_sub(edge.completed_visits),
            edge.q
        );
    }
    if let Some(nn) = nn {
        let mut by_p = nn.policies.clone();
        by_p.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        println!("  nn top {}/{} (by P)", by_p.len().min(top), by_p.len());
        println!("  move       P");
        for (mv, prior) in by_p.into_iter().take(top) {
            let mv = if root_is_black { mv.flip() } else { mv };
            println!("  {:<6} {:>7.4}", mv.to_uci(), prior);
        }
    }
}

fn print_tree_funnel(search: &Search<BenchObserver>, root_is_black: bool, max_depth: usize, top: usize) {
    println!("Tree funnel (depth<={max_depth}, top={top}/node; path-local cycle stop)");
    let mut path = HashSet::new();
    print_tree_node(
        search,
        search.root_id(),
        root_is_black,
        0,
        max_depth,
        top,
        None,
        &mut path,
    );
}

#[allow(clippy::too_many_arguments)]
fn print_tree_node(
    search: &Search<BenchObserver>,
    node_id: NodeId,
    root_is_black: bool,
    depth: usize,
    max_depth: usize,
    top: usize,
    via: Option<(String, f32, u32, f32)>,
    path: &mut HashSet<NodeId>,
) {
    if depth > max_depth || !path.insert(node_id) {
        return;
    }
    let Some(node) = search.arena().get(node_id) else {
        path.remove(&node_id);
        return;
    };
    let indent = "  ".repeat(depth + 1);
    match via {
        Some((mv, prior, n, q)) => println!(
            "{indent}{mv:<6} P={prior:.4} N={n:<6} Q={q:.4}  node_N={} M={:.1}",
            node.completed_visits(),
            node.m()
        ),
        None => println!(
            "{indent}root     N={:<6} Q={:.4} M={:.1}",
            node.completed_visits(),
            node.q(),
            node.m()
        ),
    }
    if depth == max_depth {
        path.remove(&node_id);
        return;
    }
    let edge_table = node.edges();
    let mut edges: Vec<_> = edge_table.iter().collect();
    edges.sort_unstable_by(|left, right| {
        right
            .completed_visits()
            .cmp(&left.completed_visits())
            .then_with(|| right.visits().cmp(&left.visits()))
            .then_with(|| right.prior().total_cmp(&left.prior()))
    });
    for edge in edges.into_iter().take(top) {
        if edge.completed_visits() == 0 && edge.visits() == 0 {
            continue;
        }
        let mv = if root_is_black { edge.mv().flip() } else { edge.mv() };
        let Some(child) = edge.child() else {
            continue;
        };
        print_tree_node(
            search,
            child,
            root_is_black,
            depth + 1,
            max_depth,
            top,
            Some((mv.to_uci(), edge.prior(), edge.completed_visits(), edge.q())),
            path,
        );
    }
    path.remove(&node_id);
}

#[allow(clippy::too_many_arguments)]
fn print_run_report(
    gather_workers: usize,
    eval_workers: usize,
    run_index: usize,
    seconds: f64,
    stats: &Stats,
    bench: &BenchStats,
    search: &Search<BenchObserver>,
    root_is_black: bool,
    args: &Args,
    nn: &RootNnProbe,
) {
    let ms = seconds * 1e3;
    let nps = stats.completed_playouts as f64 / seconds;
    let eps = stats.network_evaluations as f64 / seconds;
    let attempts = bench.submitted_playouts.max(1);
    let collision_rate = bench.collisions as f64 * 100.0 / attempts as f64;
    let cache_denom = (stats.network_evaluations + bench.cache_hits).max(1);
    let cache_hit_rate = bench.cache_hits as f64 * 100.0 / cache_denom as f64;
    let batch_avg = if bench.network_batches == 0 {
        0.0
    } else {
        stats.network_evaluations as f64 / bench.network_batches as f64
    };

    println!("=== G={gather_workers} E={eval_workers} run={run_index} ===");
    println!("Throughput");
    println!(
        "  ms={ms:.1}  nps={nps:.0}  eps={eps:.0}  completed={}  submitted={}  peak_inflight={}",
        stats.completed_playouts, bench.submitted_playouts, bench.peak_in_flight
    );
    println!(
        "  pipeline  submitted={} -> collision={} ({collision_rate:.1}%) -> nn_eval={} -> completed={}",
        bench.submitted_playouts, bench.collisions, stats.network_evaluations, stats.completed_playouts
    );

    println!("Collisions");
    println!(
        "  count={}  rate={collision_rate:.1}%  (of submitted={})",
        bench.collisions, bench.submitted_playouts
    );
    if args.show_collision_dist {
        println!("  depth dist  {}", format_collision_dist(&bench.collisions_by_depth));
    }

    println!("NN / Cache");
    println!(
        "  n_eval={}  cache_hits={} ({cache_hit_rate:.1}%)  batches={}  batch avg={batch_avg:.2} max={}",
        stats.network_evaluations, bench.cache_hits, bench.network_batches, bench.network_batch_size_max
    );
    println!(
        "  batch dist (size×times)  {}",
        format_batch_dist(&bench.batches_by_size)
    );

    println!("Queues (avg/max us)");
    print_queue("gather", bench.gather_queue);
    print_queue("eval", bench.eval_queue);
    print_queue("nn", bench.nn_queue);
    print_queue("backprop", bench.backprop_queue);

    println!("Search depth");
    println!("  avg={}  max={}", stats.average_depth, stats.max_depth);

    print_root_block(search, root_is_black, args.root_top, "Root", Some(nn));

    if let Some(depth) = args.tree_depth {
        print_tree_funnel(search, root_is_black, depth, args.tree_top);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if !args.onnx.is_file() {
        return Err(format!("onnx missing: {}", args.onnx.display()).into());
    }
    let (backend, provider, recommended) = make_backend(&args.onnx)?;
    let target_batch = args.eval_batch.unwrap_or(recommended).max(1);
    let positions = load_positions(&args)?;
    println!(
        "onnx={} provider={} virtual_mean_fpu_scale={:.2} recommended_batch={} target_batch={} nn_window={} budget={} repeat={} worker_matrix={} positions={}",
        args.onnx.display(),
        provider,
        args.virtual_mean_fpu_scale,
        recommended,
        target_batch,
        args.nn_window,
        args.playouts
            .map(|n| format!("playouts={n}"))
            .unwrap_or_else(|| format!("movetime={}ms", args.movetime.unwrap_or(0))),
        args.repeat,
        args.gathers.len() * args.evals.len(),
        positions.len(),
    );
    println!(
        "note: warmup each bench position (also captures nn P/Q/M); shared backend; \
         fresh tree + clear_cache each run; collisions + depth + root always printed; \
         --collision-dist / --tree-depth only for distribution shapes; \
         nps=completed/s, eps=nn_eval/s, submitted includes collisions"
    );
    let mut generation = 0;
    for (name, fen) in positions {
        let state = GameState::from_fen_moves(&fen, &args.moves)?;
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        println!("position: {name}");
        let nn_probe = warmup_position(backend.as_ref(), history.as_ref(), target_batch)?;
        for &gather_workers in &args.gathers {
            for &eval_workers in &args.evals {
                for run_index in 1..=args.repeat {
                    generation += 1;
                    backend.clear_cache();
                    let mut search = Search::new_with_observer(
                        Arc::clone(&backend),
                        generation,
                        Arc::clone(&history),
                        SearchConfig {
                            eval_batch_size: target_batch,
                            nn_window: args.nn_window,
                            gather_workers,
                            eval_workers,
                            params: SearchParams {
                                virtual_mean_fpu_scale: args.virtual_mean_fpu_scale,
                                ..SearchParams::default()
                            },
                            ..SearchConfig::default()
                        },
                        BenchObserver::new(),
                    );

                    let started = Instant::now();
                    let stats = if args.trace.is_empty() {
                        search.run_with_limits(SearchLimits {
                            max_playouts: args.playouts,
                            deadline: args.movetime.map(|ms| Instant::now() + Duration::from_millis(ms)),
                            ..Default::default()
                        })?
                    } else {
                        let playouts = args.playouts.expect("trace requires playouts");
                        for &milestone in &args.trace {
                            search.run_playouts(milestone)?;
                            println!("--- trace completed={milestone} ---");
                            print_root_block(
                                &search,
                                root_is_black,
                                args.root_top,
                                &format!("Root snapshot @ completed={milestone}"),
                                Some(&nn_probe),
                            );
                        }
                        if args.trace.last().copied().unwrap_or(0) < playouts {
                            search.run_playouts(playouts)?
                        } else {
                            search.stats()
                        }
                    };
                    let seconds = started.elapsed().as_secs_f64().max(1e-9);
                    let bench = search.observer().snapshot();
                    print_run_report(
                        gather_workers,
                        eval_workers,
                        run_index,
                        seconds,
                        &stats,
                        &bench,
                        &search,
                        root_is_black,
                        &args,
                        &nn_probe,
                    );
                    search.stop_and_finish();
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
