//! 固定节点下观察 cPUCT/FPU 对根边分流的影响。
//!
//! worker、batch、缓存吞吐实验见 `benchmark`。本工具固定正式默认的 worker
//! 拓扑，只改变 `SearchParams`，并始终从 fresh graph 开始。
//! 根过滤沿用 Engine 的 `searchmoves` 语义。

use std::path::PathBuf;
use std::sync::Arc;

use engin::neural::backend::Backend;
use engin::neural::onnx::OnnxBackend;
use engin::search::{Search, SearchConfig, SearchParams, root_stats};
use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

struct Args {
    onnx: PathBuf,
    fen: String,
    moves: Vec<String>,
    playouts: u64,
    trace: Vec<u64>,
    track: Vec<String>,
    searchmoves: Vec<String>,
    cpucts: Vec<f32>,
    cpuct_bases: Vec<f32>,
    cpuct_factors: Vec<f32>,
    fpu_reduction: f32,
    lcb_stdevs: f32,
    lcb_min_visit_fraction: f32,
    root_top: usize,
}

fn usage() -> &'static str {
    "usage: search_benchmark [--onnx data/x7.onnx] [--fen \"...\"] [--moves \"c3c4 h7h3 ...\"] [--playouts 2048] \\
     [--trace 128,256,512] [--track g6g9,i0g0] [--searchmoves \"g6g9 i0g0\"] [--cpuct 1.0,1.745] \\
     [--cpuct-base 20000,38739] [--cpuct-factor 2.5,3.894] [--fpu-reduction 0.330] \\
     [--lcb-stdevs 5] [--lcb-min-visit-fraction 0.15] [--root-top 8]"
}

fn parse_args() -> Result<Args, String> {
    let defaults = SearchParams::default();
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut moves = Vec::new();
    let mut playouts = 2_048;
    let mut trace = Vec::new();
    let mut track = Vec::new();
    let mut searchmoves = Vec::new();
    let mut cpucts = vec![defaults.cpuct];
    let mut cpuct_bases = vec![defaults.cpuct_base];
    let mut cpuct_factors = vec![defaults.cpuct_factor];
    let mut fpu_reduction = defaults.fpu_reduction;
    let mut lcb_stdevs = defaults.lcb_stdevs;
    let mut lcb_min_visit_fraction = defaults.lcb_min_visit_fraction;
    let mut root_top = 8;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--onnx" => onnx = PathBuf::from(args.next().ok_or("--onnx requires a path")?),
            "--fen" => fen = args.next().ok_or("--fen requires a quoted FEN")?,
            "--moves" => {
                moves = args
                    .next()
                    .ok_or("--moves requires space-separated ICCS moves")?
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect()
            }
            "--playouts" => {
                playouts = args
                    .next()
                    .ok_or("--playouts requires an integer")?
                    .parse()
                    .map_err(|_| "--playouts must be an unsigned integer")?
            }
            "--trace" => trace = parse_u64_list(&args.next().ok_or("--trace requires a list")?)?,
            "--track" => track = parse_move_list(&args.next().ok_or("--track requires a move list")?)?,
            "--searchmoves" => {
                searchmoves = args
                    .next()
                    .ok_or("--searchmoves requires ICCS moves")?
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect()
            }
            "--cpuct" => cpucts = parse_float_list("--cpuct", &args.next().ok_or("--cpuct requires a number")?, false)?,
            "--cpuct-base" => {
                cpuct_bases = parse_float_list(
                    "--cpuct-base",
                    &args.next().ok_or("--cpuct-base requires a number")?,
                    true,
                )?
            }
            "--cpuct-factor" => {
                cpuct_factors = parse_float_list(
                    "--cpuct-factor",
                    &args.next().ok_or("--cpuct-factor requires a number")?,
                    false,
                )?
            }
            "--fpu-reduction" => {
                fpu_reduction = parse_float(
                    "--fpu-reduction",
                    &args.next().ok_or("--fpu-reduction requires a number")?,
                    false,
                )?
            }
            "--lcb-stdevs" => {
                lcb_stdevs = parse_float(
                    "--lcb-stdevs",
                    &args.next().ok_or("--lcb-stdevs requires a number")?,
                    false,
                )?
            }
            "--lcb-min-visit-fraction" => {
                lcb_min_visit_fraction = parse_float(
                    "--lcb-min-visit-fraction",
                    &args.next().ok_or("--lcb-min-visit-fraction requires a number")?,
                    false,
                )?;
                if lcb_min_visit_fraction > 1.0 {
                    return Err("--lcb-min-visit-fraction must not exceed 1".into());
                }
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
    if playouts == 0 || root_top == 0 {
        return Err("--playouts and --root-top must be positive".into());
    }
    if trace.windows(2).any(|pair| pair[0] >= pair[1]) || trace.iter().any(|&value| value == 0 || value > playouts) {
        return Err("--trace milestones must be strictly increasing and within --playouts".into());
    }
    Ok(Args {
        onnx,
        fen,
        moves,
        playouts,
        trace,
        track,
        searchmoves,
        cpucts,
        cpuct_bases,
        cpuct_factors,
        fpu_reduction,
        lcb_stdevs,
        lcb_min_visit_fraction,
        root_top,
    })
}

fn parse_float(name: &str, value: &str, positive: bool) -> Result<f32, String> {
    let value = value.parse::<f32>().map_err(|_| format!("{name} must be finite"))?;
    if !value.is_finite() || value < 0.0 || (positive && value == 0.0) {
        return Err(format!(
            "{name} must be {}",
            if positive { "positive" } else { "non-negative" }
        ));
    }
    Ok(value)
}

fn parse_float_list(name: &str, text: &str, positive: bool) -> Result<Vec<f32>, String> {
    let values: Result<Vec<_>, _> = text
        .split(',')
        .map(|value| parse_float(name, value.trim(), positive))
        .collect();
    let values = values?;
    if values.is_empty() {
        Err(format!("{name} requires at least one number"))
    } else {
        Ok(values)
    }
}

fn parse_u64_list(text: &str) -> Result<Vec<u64>, String> {
    text.split(',')
        .map(|part| {
            part.trim()
                .parse()
                .map_err(|_| format!("invalid trace milestone: {part}"))
        })
        .collect()
}

fn parse_move_list(text: &str) -> Result<Vec<String>, String> {
    let moves: Vec<_> = text
        .split(',')
        .map(str::trim)
        .filter(|move_text| !move_text.is_empty())
        .map(str::to_owned)
        .collect();
    if moves.is_empty() {
        Err("--track requires at least one move".into())
    } else {
        Ok(moves)
    }
}

/// 与 Engine 保持相同的根着过滤语义。
fn root_filter(history: &PositionHistory, requested: &[String]) -> Result<Vec<xiangqi_core::Move>, String> {
    let board = history.last().board();
    let legal = board.generate_legal_moves();
    let selected: Vec<_> = requested
        .iter()
        .filter_map(|text| board.parse_move(text).ok())
        .filter(|mv| legal.contains(mv))
        .collect();
    if !requested.is_empty() && selected.is_empty() {
        Err("No legal searchmoves.".into())
    } else {
        Ok(selected)
    }
}

fn sorted_root_edges(search: &Search) -> Option<Vec<engin::search::RootEdgeStats>> {
    let root = root_stats(search.repository(), search.root_key())?;
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

fn print_roots(search: &Search, root_is_black: bool, top: usize, tracked: &[String]) {
    let Some(edges) = sorted_root_edges(search) else {
        return;
    };
    println!(
        "    root candidates top {}/{}: move       P    done flight       Q",
        edges.len().min(top),
        edges.len()
    );
    for edge in edges.iter().take(top) {
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
    if tracked.is_empty() {
        return;
    }
    println!("    tracked root moves: move  rank       P    done flight       Q");
    for move_text in tracked {
        let found = edges.iter().enumerate().find(|(_, edge)| {
            let mv = if root_is_black { edge.mv.flip() } else { edge.mv };
            mv.to_uci() == *move_text
        });
        match found {
            Some((rank, edge)) => println!(
                "                       {:<6} {:>5} {:>7.4} {:>7} {:>6} {:>7.4}",
                move_text,
                rank + 1,
                edge.prior,
                edge.completed_visits,
                edge.started_visits.saturating_sub(edge.completed_visits),
                edge.q
            ),
            None => println!("                       {:<6}  --  not legal at root", move_text),
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if !args.onnx.is_file() {
        return Err(format!("onnx missing: {}", args.onnx.display()).into());
    }
    let state = GameState::from_fen_moves(&args.fen, &args.moves)?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    let root_is_black = history.is_black_to_move();
    let filter = root_filter(&history, &args.searchmoves)?;
    let backend = OnnxBackend::from_file(&args.onnx)?;
    println!(
        "onnx={} provider={} playouts={} cpuct={:?} cpuct_base={:?} cpuct_factor={:?} fpu_reduction={:.3} lcb_stdevs={:.3} lcb_min_visit_fraction={:.3}",
        args.onnx.display(),
        backend.provider().name(),
        args.playouts,
        args.cpucts,
        args.cpuct_bases,
        args.cpuct_factors,
        args.fpu_reduction,
        args.lcb_stdevs,
        args.lcb_min_visit_fraction,
    );
    println!(
        "note: fresh graph; workers=4 Search / 4 Eval; batch uses backend default; trace drains at each milestone"
    );
    let mut generation = 0;
    for &cpuct in &args.cpucts {
        for &cpuct_base in &args.cpuct_bases {
            for &cpuct_factor in &args.cpuct_factors {
                generation += 1;
                let params = SearchParams {
                    cpuct,
                    cpuct_base,
                    cpuct_factor,
                    fpu_reduction: args.fpu_reduction,
                    lcb_stdevs: args.lcb_stdevs,
                    lcb_min_visit_fraction: args.lcb_min_visit_fraction,
                };
                println!(
                    "params: cpuct={cpuct:.3} cpuct_base={cpuct_base:.0} cpuct_factor={cpuct_factor:.3} fpu={:.3} lcb={:.3}/{:.3}",
                    args.fpu_reduction, args.lcb_stdevs, args.lcb_min_visit_fraction
                );
                let mut search = Search::new(
                    Arc::new(OnnxBackend::from_file(&args.onnx)?) as Arc<dyn Backend>,
                    generation,
                    Arc::clone(&history),
                    SearchConfig {
                        params,
                        root_move_filter: filter.clone(),
                        ..SearchConfig::default()
                    },
                );
                for &milestone in &args.trace {
                    // `Search::run_playouts` 的参数是当前 Search 的累计目标；trace
                    // milestone 不能再减去上一项，否则 100→1000 会错误停在 1000 而
                    // 非“额外跑 900”后的 1000。
                    search.run_playouts(milestone)?;
                    println!("    trace completed={milestone}");
                    print_roots(&search, root_is_black, args.root_top, &args.track);
                }
                if args.trace.last().copied().unwrap_or(0) < args.playouts {
                    search.run_playouts(args.playouts)?;
                }
                print_roots(&search, root_is_black, args.root_top, &args.track);
                search.stop_and_finish();
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("search_benchmark: {error}");
        std::process::exit(2);
    }
}
