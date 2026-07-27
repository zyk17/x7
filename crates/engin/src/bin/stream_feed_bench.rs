//! Sweep Gather × Eval × Backprop for stream NPS / collisions.
//!
//! Topology: independent NN thread (ONNX only). Eval does terminal/cache/encode;
//! NN merges queued tensors up to `--eval-batch`. Default workers: 4/2/1.
//! Not a UCI path.
//!
//! ```text
//! cargo run -p engin --release --bin stream_feed_bench -- --cache --playouts 20000
//! cargo run -p engin --release --bin stream_feed_bench -- --gathers 4,8 --evals 1,2 --backprops 1,2 --cache
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use engin::neural::backend::{Backend, CachingBackend};
use engin::neural::onnx::OnnxBackend;
use engin::search::stream::{Search, SearchConfig, SearchGeneration};
use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

struct Args {
    onnx: PathBuf,
    fen: String,
    playouts: u64,
    gathers: Vec<usize>,
    evals: Vec<usize>,
    backprops: Vec<usize>,
    eval_batch: Option<usize>,
    cache: bool,
}

fn usage() -> &'static str {
    "usage: stream_feed_bench [--onnx data/x7.onnx] [--fen \"...\"] [--playouts 20000] \
     [--gathers 4,8] [--evals 1,2] [--backprops 1,2] [--eval-batch 64] [--cache]"
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut playouts = 20_000;
    let mut gathers = vec![4, 8];
    let mut evals = vec![1, 2];
    let mut backprops = vec![1, 2];
    let mut eval_batch = None;
    let mut cache = false;
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
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if playouts == 0 {
        return Err("--playouts must be > 0".into());
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
    Ok(Args {
        onnx,
        fen,
        playouts,
        gathers,
        evals,
        backprops,
        eval_batch,
        cache,
    })
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

fn make_backend(
    path: &PathBuf,
    cache: bool,
) -> Result<BackendSetup, Box<dyn std::error::Error>> {
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
    println!(
        "onnx={} provider={} cache={} recommended_batch={} target_batch={} playouts={} matrix={}",
        args.onnx.display(),
        provider,
        args.cache,
        recommended,
        target_batch,
        args.playouts,
        cells
    );
    println!(
        "note: fresh backend/cache per cell; nn_eval=GPU-only; NN merges queued tensors up to target_batch"
    );
    println!(
        "{:>3} {:>3} {:>3} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "G", "E", "B", "ms", "nps", "done", "coll", "nn_eval", "avg_b", "max_b"
    );

    let state = GameState::from_fen_moves(&args.fen, &[] as &[&str])?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    let mut generation = 0u64;

    for &gather_workers in &args.gathers {
        for &eval_workers in &args.evals {
            for &backprop_workers in &args.backprops {
                generation += 1;
                let (backend, _, _) = make_backend(&args.onnx, args.cache)?;
                let search = Search::new(
                    backend,
                    SearchGeneration(generation),
                    Arc::clone(&history),
                    SearchConfig {
                        eval_batch_size: target_batch,
                        gather_workers,
                        eval_workers,
                        backprop_workers,
                        ..SearchConfig::default()
                    },
                );
                let t0 = Instant::now();
                let stats = search.run_playouts(args.playouts)?;
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let nps = if ms > 0.0 {
                    stats.completed_playouts as f64 / (ms / 1e3)
                } else {
                    0.0
                };
                let avg_batch = if stats.network_batches > 0 {
                    stats.network_evaluations as f64 / stats.network_batches as f64
                } else {
                    0.0
                };
                println!(
                    "{:>3} {:>3} {:>3} {:>8.1} {:>8.0} {:>8} {:>8} {:>8} {:>8.2} {:>8}",
                    gather_workers,
                    eval_workers,
                    backprop_workers,
                    ms,
                    nps,
                    stats.completed_playouts,
                    stats.collisions,
                    stats.network_evaluations,
                    avg_batch,
                    stats.network_batch_size_max
                );
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("stream_feed_bench: {error}");
        std::process::exit(2);
    }
}
