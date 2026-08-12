//! 一次性 NN 探针：FEN + moves → 合法着 policy / WDL / moves_left，并可选测量延迟。
//!
//! 复用 `PositionHistory` 编码与 `OnnxBackend`，即与搜索相同的路径；不属于 UCI 路径。
//!
//! ```text
//! cargo run -p engin --bin nn_eval -- --onnx data/x7.onnx
//! cargo run -p engin --bin nn_eval -- --fen "..." --moves "h2e2 h7e7" --top 16
//! cargo run -p engin --bin nn_eval -- --bench 50 --batch 1,16,64
//! ```

use std::path::PathBuf;
use std::time::Instant;

use engin::neural::backend::Backend;
use engin::neural::onnx::OnnxBackend;
use engin::neural::{
    EncodedBatch, FillEmptyHistory, InputPlanes, encode_position_input_planes, eval_result_from_encoded_row,
};
use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

struct Args {
    onnx: PathBuf,
    fen: String,
    moves: Vec<String>,
    top: usize,
    bench_iters: Option<usize>,
    batches: Vec<usize>,
}

fn usage() -> &'static str {
    "usage: nn_eval [--onnx data/x7.onnx] [--fen \"...\"] [--moves \"m1 m2 ...\"] [--top 32] [--bench N] [--batch 1,16,64]"
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut moves = Vec::new();
    let mut top = 32;
    let mut bench_iters = None;
    let mut batches = vec![1, 16, 64];
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--onnx" => onnx = PathBuf::from(args.next().ok_or("--onnx requires a path")?),
            "--fen" => fen = args.next().ok_or("--fen requires a quoted FEN")?,
            "--moves" => {
                let text = args.next().ok_or("--moves requires a quoted move list")?;
                moves = text
                    .split_whitespace()
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
            "--top" => top = parse_usize(&mut args, "--top")?,
            "--bench" => bench_iters = Some(parse_usize(&mut args, "--bench")?),
            "--batch" => {
                batches = parse_batches(&args.next().ok_or("--batch requires list like 1,16,64")?)?;
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if top == 0 {
        return Err("--top must be > 0".into());
    }
    if batches.contains(&0) {
        return Err("--batch sizes must be > 0".into());
    }
    Ok(Args {
        onnx,
        fen,
        moves,
        top,
        bench_iters,
        batches,
    })
}

fn parse_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires an integer"))?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn parse_batches(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid --batch entry: {part}"))
        })
        .collect()
}

fn wdl_from_eval(wl: f32, d: f32) -> (f32, f32, f32) {
    // value = [W,D,L]；EvalResult 保存 wl=W-L、d=D。
    let w = ((1.0 - d + wl) * 0.5).clamp(0.0, 1.0);
    let l = ((1.0 - d - wl) * 0.5).clamp(0.0, 1.0);
    (w, d.clamp(0.0, 1.0), l)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if !args.onnx.is_file() {
        return Err(format!("onnx missing: {}", args.onnx.display()).into());
    }
    let size_mb = args.onnx.metadata()?.len() as f64 / (1024.0 * 1024.0);
    let backend = OnnxBackend::from_file(&args.onnx)?;
    println!(
        "onnx={} size_mb={size_mb:.2} provider={} has_wdl={} has_mlh={}",
        args.onnx.display(),
        backend.provider().name(),
        backend.attributes().has_wdl,
        backend.attributes().has_mlh
    );

    let move_refs: Vec<&str> = args.moves.iter().map(String::as_str).collect();
    let state = GameState::from_fen_moves(&args.fen, &move_refs)?;
    let history = PositionHistory::from_positions(state.positions());
    let legal = history.last().board().generate_legal_moves();
    println!(
        "fen={} moves={} side={} legal={}",
        state.current_position().to_fen(),
        if args.moves.is_empty() {
            "-".to_owned()
        } else {
            args.moves.join(" ")
        },
        if history.is_black_to_move() { "black" } else { "red" },
        legal.len()
    );

    let t0 = Instant::now();
    let eval = evaluate(&backend, &history, &legal)?;
    let eval_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (w, d, l) = wdl_from_eval(eval.wl, eval.d);
    println!(
        "eval_ms={eval_ms:.3} wdl W={w:.6} D={d:.6} L={l:.6} Q(wl)={:.6} plies_left={:.6}",
        eval.wl, eval.plies_left
    );

    let mut ranked: Vec<(Move, f32)> = legal.iter().copied().zip(eval.policies.iter().copied()).collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let show = ranked.len().min(args.top);
    println!("policy top {show}/{} (legal softmax):", ranked.len());
    for (i, (mv, p)) in ranked.iter().take(show).enumerate() {
        println!("  {:>2}  {}  {:.6}", i + 1, mv.to_uci(), p);
    }
    let policy_sum: f32 = eval.policies.iter().sum();
    println!("policy_sum={policy_sum:.6}");

    if let Some(iters) = args.bench_iters {
        run_bench(&backend, &history, iters, &args.batches)?;
    }
    Ok(())
}

fn run_bench(
    backend: &OnnxBackend,
    history: &PositionHistory,
    iters: usize,
    batches: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("bench iters={iters} (warmup 3, exclude from stats)");
    let planes = encode_position_input_planes(history, FillEmptyHistory::FenOnly);
    for &batch in batches {
        for _ in 0..3 {
            timed_batch(backend, &planes, batch)?;
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            samples.push(timed_batch(backend, &planes, batch)?);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sum: f64 = samples.iter().sum();
        let avg = sum / samples.len() as f64;
        let p50 = samples[samples.len() / 2];
        let p95 = samples[((samples.len() as f64 * 0.95) as usize).min(samples.len() - 1)];
        let pos_per_s = (batch as f64) / (avg / 1e3);
        println!("  batch={batch:>3}  avg_ms={avg:.3}  p50_ms={p50:.3}  p95_ms={p95:.3}  pos/s={pos_per_s:.0}");
    }
    Ok(())
}

fn timed_batch(backend: &OnnxBackend, planes: &InputPlanes, batch: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let samples = vec![*planes; batch];
    let mut logits = Vec::new();
    let mut wdl = Vec::new();
    let mut moves_left = Vec::new();
    let t0 = Instant::now();
    backend.infer_input_planes_into(&samples, &mut logits, &mut wdl, &mut moves_left)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3)
}

fn evaluate(
    backend: &OnnxBackend,
    history: &PositionHistory,
    legal_moves: &[Move],
) -> Result<std::sync::Arc<engin::neural::backend::EvalResult>, Box<dyn std::error::Error>> {
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
    Ok(eval_result_from_encoded_row(&output, 0, legal_moves)?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nn_eval: {error}");
        std::process::exit(2);
    }
}
