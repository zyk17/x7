//! 固定节点预算后的 stream MCGS 图形状快照。
//!
//! 复用 `Node` / `Edge` 的只读统计；不改变 worker 或正式 UCI。可通过参数覆盖 PUCT，
//! 图按已完成 visit 展开，参考 LC3 Overview 的 "Node structure"。每一行代表一个已访问 node，
//! 只展示进入该 node 的 `P/N/Q/M`，不重复打印 edge 或内部状态。`M` 是 node 聚合的
//! moves-left 平均值，单位为 ply。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use engin::neural::backend::Backend;
use engin::neural::onnx::OnnxBackend;
use engin::search::{NodeKey, NodeRepository, Search, SearchConfig, SearchGeneration, SearchParams};
use xiangqi_core::{GameState, Move, STARTPOS_FEN};

struct Args {
    onnx: PathBuf,
    fen: String,
    moves: Vec<String>,
    playouts: u64,
    depth: usize,
    top: usize,
    cpuct: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
    fpu_reduction: f32,
}

fn usage() -> &'static str {
    "usage: graph_shape [--onnx data/x7.onnx] [--fen \"...\"] [--moves \"c3c4 h7h3 ...\"] \\
     [--playouts 2000] [--depth 4] [--top 8] [--cpuct 1.5] [--cpuct-base 2000] \\
     [--cpuct-factor 1.347017] [--fpu-reduction 0.2]"
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut moves = Vec::new();
    let mut playouts = 2_000;
    let mut depth = 4;
    let mut top = 8;
    let defaults = SearchParams::default();
    let mut cpuct = defaults.cpuct;
    let mut cpuct_base = defaults.cpuct_base;
    let mut cpuct_factor = defaults.cpuct_factor;
    let mut fpu_reduction = defaults.fpu_reduction;
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
            "--depth" => {
                depth = args
                    .next()
                    .ok_or("--depth requires an integer")?
                    .parse()
                    .map_err(|_| "--depth must be an unsigned integer")?
            }
            "--top" => {
                top = args
                    .next()
                    .ok_or("--top requires an integer")?
                    .parse()
                    .map_err(|_| "--top must be an unsigned integer")?
            }
            "--cpuct" => {
                cpuct = args
                    .next()
                    .ok_or("--cpuct requires a number")?
                    .parse()
                    .map_err(|_| "--cpuct must be non-negative")?
            }
            "--cpuct-base" => {
                cpuct_base = args
                    .next()
                    .ok_or("--cpuct-base requires a number")?
                    .parse()
                    .map_err(|_| "--cpuct-base must be positive")?
            }
            "--cpuct-factor" => {
                cpuct_factor = args
                    .next()
                    .ok_or("--cpuct-factor requires a number")?
                    .parse()
                    .map_err(|_| "--cpuct-factor must be non-negative")?
            }
            "--fpu-reduction" => {
                fpu_reduction = args
                    .next()
                    .ok_or("--fpu-reduction requires a number")?
                    .parse()
                    .map_err(|_| "--fpu-reduction must be non-negative")?
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if playouts == 0
        || top == 0
        || !cpuct.is_finite()
        || cpuct < 0.0
        || !cpuct_base.is_finite()
        || cpuct_base <= 0.0
        || !cpuct_factor.is_finite()
        || cpuct_factor < 0.0
        || !fpu_reduction.is_finite()
        || fpu_reduction < 0.0
    {
        return Err(
            "--playouts, --top, and --cpuct-base must be positive; search parameters must be finite and non-negative"
                .into(),
        );
    }
    Ok(Args {
        onnx,
        fen,
        moves,
        playouts,
        depth,
        top,
        cpuct,
        cpuct_base,
        cpuct_factor,
        fpu_reduction,
    })
}

struct Child {
    mv: Move,
    key: Option<NodeKey>,
    prior: f32,
    completed: u32,
}

struct PrintStyle {
    root_is_black: bool,
    display_depth: usize,
    top: usize,
}

struct PrintContext<'a> {
    style: &'a PrintStyle,
    path: &'a mut HashSet<NodeKey>,
}

fn children(repository: &NodeRepository, key: NodeKey) -> Vec<Child> {
    let Some(node) = repository.get(key) else {
        return Vec::new();
    };
    let mut children: Vec<_> = node
        .edges()
        .iter()
        .filter(|edge| edge.completed_visits() > 0)
        .map(|edge| Child {
            mv: edge.mv(),
            key: edge.child_key(),
            prior: edge.prior(),
            completed: edge.completed_visits(),
        })
        .collect();
    children.sort_unstable_by(|left, right| {
        right
            .completed
            .cmp(&left.completed)
            .then_with(|| right.prior.total_cmp(&left.prior))
    });
    children
}

fn move_text(mv: Move, root_is_black: bool, parent_depth: usize) -> String {
    let flip = root_is_black != (parent_depth % 2 == 1);
    if flip { mv.flip() } else { mv }.to_uci()
}

fn print_node(
    repository: &NodeRepository,
    key: NodeKey,
    incoming: Option<&Child>,
    remaining_depth: usize,
    prefix: &str,
    is_last: bool,
    context: &mut PrintContext<'_>,
) {
    let Some(node) = repository.get(key) else {
        return;
    };
    let children = children(repository, key);
    let visited = children.len();
    match incoming {
        None => println!(
            "root  N={}  Q={:.4}  M={:.1}",
            node.completed_visits(),
            node.q(),
            node.m()
        ),
        Some(child) => println!(
            "{prefix}{} {}  P={:.4}  N={}  Q={:.4}  M={:.1}",
            if is_last { "└─" } else { "├─" },
            move_text(
                child.mv,
                context.style.root_is_black,
                context.style.display_depth - remaining_depth - 1,
            ),
            child.prior,
            node.completed_visits(),
            node.q(),
            node.m()
        ),
    }
    if remaining_depth == 0 || visited == 0 {
        return;
    }
    let shown = children.iter().take(context.style.top).collect::<Vec<_>>();
    let child_prefix = match incoming {
        None => String::new(),
        Some(_) if is_last => format!("{prefix}   "),
        Some(_) => format!("{prefix}│  "),
    };
    for (index, child) in shown.iter().enumerate() {
        let last = index + 1 == shown.len() && visited == shown.len();
        let branch = if last { "└─" } else { "├─" };
        let Some(child_key) = child.key else {
            println!(
                "{child_prefix}{branch} {}  P={:.4}  N={}  ↻ local cycle",
                move_text(
                    child.mv,
                    context.style.root_is_black,
                    context.style.display_depth - remaining_depth - 1,
                ),
                child.prior,
                child.completed,
            );
            continue;
        };
        if context.path.contains(&child_key) {
            println!(
                "{child_prefix}{branch} {}  P={:.4}  N={}  ↻ graph cycle",
                move_text(
                    child.mv,
                    context.style.root_is_black,
                    context.style.display_depth - remaining_depth - 1,
                ),
                child.prior,
                child.completed,
            );
            continue;
        }
        context.path.insert(child_key);
        print_node(
            repository,
            child_key,
            Some(child),
            remaining_depth - 1,
            &child_prefix,
            last,
            context,
        );
        context.path.remove(&child_key);
    }
    if visited > shown.len() {
        println!("{child_prefix}└─ ... {} nodes omitted by --top", visited - shown.len());
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if !args.onnx.is_file() {
        return Err(format!("onnx missing: {}", args.onnx.display()).into());
    }
    let state = GameState::from_fen_moves(&args.fen, &args.moves)?;
    let history = Arc::new(state.position_history());
    let root_is_black = history.is_black_to_move();
    let backend = OnnxBackend::from_file(&args.onnx)?;
    let mut search = Search::new(
        Arc::new(backend) as Arc<dyn Backend>,
        SearchGeneration(1),
        history,
        SearchConfig {
            params: SearchParams {
                cpuct: args.cpuct,
                cpuct_base: args.cpuct_base,
                cpuct_factor: args.cpuct_factor,
                fpu_reduction: args.fpu_reduction,
            },
            ..SearchConfig::default()
        },
    );
    let stats = search.run_playouts(args.playouts)?;
    let attempts = stats.completed_playouts + stats.collisions;
    let collision_rate = if attempts == 0 {
        0.0
    } else {
        stats.collisions as f64 * 100.0 / attempts as f64
    };
    println!(
        "total  N={}  collisions={}  rate={collision_rate:.1}%",
        stats.completed_playouts, stats.collisions
    );
    let style = PrintStyle {
        root_is_black,
        display_depth: args.depth,
        top: args.top,
    };
    let mut path = HashSet::from([search.root_key()]);
    let mut context = PrintContext {
        style: &style,
        path: &mut path,
    };
    print_node(
        search.repository(),
        search.root_key(),
        None,
        args.depth,
        "",
        true,
        &mut context,
    );
    search.stop_and_finish();
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("graph_shape: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use xiangqi_core::Square;

    use super::{Move, move_text};

    #[test]
    // px0 的 board mirror 每 ply 交替；诊断输出必须恢复外部 ICCS 坐标。
    fn move_text_restores_orientation_on_each_ply() {
        let mv = Move::new(Square::parse("e0").expect("from"), Square::parse("e1").expect("to"));
        assert_eq!(move_text(mv, true, 0), "e9e8");
        assert_eq!(move_text(mv, true, 1), "e0e1");
        assert_eq!(move_text(mv, false, 0), "e0e1");
        assert_eq!(move_text(mv, false, 1), "e9e8");
    }
}
