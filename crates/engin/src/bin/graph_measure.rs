//! 在 stream MCGS 图上测量 board-key node 复用。
//!
//! 参考 `MCGS.md`：从 root 的已完成 edge 只遍历一次每个共享 node。`merged_edges`
//! 表示同一 child 被多个 parent edge 指向的额外入边。这是一跳 fan-in 指标，不能替代
//! 旧 tree 快照中递归展开后的 board 重复率。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use engin::neural::backend::Backend;
use engin::neural::onnx::OnnxBackend;
use engin::search::{NodeKey, NodeRepository, Search, SearchConfig};
use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

struct Args {
    onnx: PathBuf,
    fen: String,
    moves: Vec<String>,
    playouts: u64,
    root_moves: Vec<String>,
}

fn usage() -> &'static str {
    "usage: graph_measure [--onnx data/x7.onnx] [--fen \"...\"] [--moves \"c3c4 h7h3 ...\"] [--playouts 10000] [--root-moves b7b8,b4c6]"
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = PathBuf::from("data/x7.onnx");
    let mut fen = STARTPOS_FEN.to_owned();
    let mut moves = Vec::new();
    let mut playouts = 10_000;
    let mut root_moves = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--onnx" => onnx = PathBuf::from(arguments.next().ok_or("--onnx requires a path")?),
            "--fen" => fen = arguments.next().ok_or("--fen requires a quoted FEN")?,
            "--moves" => {
                moves = arguments
                    .next()
                    .ok_or("--moves requires space-separated ICCS moves")?
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect();
            }
            "--playouts" => {
                playouts = arguments
                    .next()
                    .ok_or("--playouts requires an integer")?
                    .parse()
                    .map_err(|_| "--playouts must be an unsigned integer")?;
            }
            "--root-moves" => {
                root_moves = arguments
                    .next()
                    .ok_or("--root-moves requires comma-separated ICCS moves")?
                    .split(',')
                    .filter(|mv| !mv.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if playouts == 0 {
        return Err("--playouts must be positive".into());
    }
    Ok(Args {
        onnx,
        fen,
        moves,
        playouts,
        root_moves,
    })
}

#[derive(Default)]
struct Measure {
    nodes: HashSet<NodeKey>,
    inbound_edges: HashMap<NodeKey, u64>,
    edges: u64,
    path_terminal_edges: u64,
}

/// 只看实际 repository，而不枚举图的所有 variation。后者会随着多 parent 组合指数
/// 膨胀；它是另一种完整展开指标，不能与这里的一跳 fan-in 混为一谈。
fn measure_graph(repository: &NodeRepository, root: NodeKey) -> Measure {
    let mut measure = Measure::default();
    let mut pending = vec![root];
    while let Some(key) = pending.pop() {
        if !measure.nodes.insert(key) {
            continue;
        }
        let Some(node) = repository.get(key) else {
            continue;
        };
        for edge in node.edges().iter().filter(|edge| edge.completed_visits() > 0) {
            measure.edges += 1;
            if let Some(child) = edge.child_key() {
                *measure.inbound_edges.entry(child).or_default() += 1;
                pending.push(child);
            } else {
                measure.path_terminal_edges += 1;
            }
        }
    }
    measure
}

fn root_child_key(repository: &NodeRepository, root: NodeKey, mv: &str) -> Result<NodeKey, String> {
    let node = repository.get(root).ok_or("root node is missing")?;
    node.edges()
        .iter()
        .find(|edge| edge.mv().to_string() == mv)
        .and_then(|edge| edge.child_key())
        .ok_or_else(|| format!("root move {mv} was not completed"))
}

fn print_measure(measure: &Measure) {
    let child_edges: u64 = measure.inbound_edges.values().sum();
    let shared_nodes = measure.inbound_edges.values().filter(|&&count| count > 1).count() as u64;
    let merged_edges: u64 = measure
        .inbound_edges
        .values()
        .map(|count| count.saturating_sub(1))
        .sum();
    println!(
        "reachable graph: nodes={} edges={} path_terminal_edges={}",
        measure.nodes.len(),
        measure.edges,
        measure.path_terminal_edges,
    );
    println!(
        "direct fan-in: child_edges={child_edges} merged_edges={merged_edges} ({:.1}%) shared_child_nodes={shared_nodes}",
        percent(merged_edges, child_edges),
    );
}

fn percent(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if !args.onnx.is_file() {
        return Err(format!("onnx missing: {}", args.onnx.display()).into());
    }
    let state = GameState::from_fen_moves(&args.fen, &args.moves)?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    let backend = OnnxBackend::from_file(&args.onnx)?;
    println!(
        "onnx={} provider={} playouts={} history=startpos/fen + {} moves",
        args.onnx.display(),
        backend.provider().name(),
        args.playouts,
        args.moves.len(),
    );
    let mut search = Search::new(
        Arc::new(backend) as Arc<dyn Backend>,
        1,
        Arc::clone(&history),
        SearchConfig::default(),
    );
    let stats = search.run_playouts(args.playouts)?;

    let measure = measure_graph(search.repository(), search.root_key());
    println!(
        "search: completed={} nn_evals={} batches={} cache_hits={}",
        stats.completed_playouts, stats.network_evaluations, stats.network_batches, stats.cache_hits,
    );
    print_measure(&measure);
    if !args.root_moves.is_empty() {
        let mut subgraphs = Vec::with_capacity(args.root_moves.len());
        for mv in &args.root_moves {
            let child = root_child_key(search.repository(), search.root_key(), mv)?;
            let subgraph = measure_graph(search.repository(), child);
            println!(
                "root {mv}: nodes={} edges={} shared_children={}",
                subgraph.nodes.len(),
                subgraph.edges,
                subgraph.inbound_edges.values().filter(|&&count| count > 1).count(),
            );
            subgraphs.push((mv, subgraph));
        }
        for left in 0..subgraphs.len() {
            for right in left + 1..subgraphs.len() {
                let overlap = subgraphs[left].1.nodes.intersection(&subgraphs[right].1.nodes).count();
                println!(
                    "overlap {} / {}: nodes={} ({:.1}% / {:.1}%)",
                    subgraphs[left].0,
                    subgraphs[right].0,
                    overlap,
                    percent(overlap as u64, subgraphs[left].1.nodes.len() as u64),
                    percent(overlap as u64, subgraphs[right].1.nodes.len() as u64),
                );
            }
        }
    }
    search.stop_and_finish();
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("graph_measure: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use super::{measure_graph, percent};
    use engin::search::{NodeKey, NodeRepository};

    #[test]
    fn percent_handles_empty_total() {
        assert_eq!(percent(1, 0), 0.0);
        assert_eq!(percent(1, 4), 25.0);
    }

    #[test]
    fn measure_counts_two_parents_of_one_child() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = PositionHistory::from_positions(state.positions());
        let first = history.last().board().parse_move("a0a1").expect("legal move");
        let second = history.last().board().parse_move("a3a4").expect("legal move");

        let repository = NodeRepository::default();
        let root_key = NodeKey::board(history.last().board().hash());
        let left_key = NodeKey::board(1);
        let right_key = NodeKey::board(2);
        let shared_key = NodeKey::board(3);
        let root = repository.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(first, 0.5), (second, 0.5)]);
        root.edges()[0].bind_child_key(left_key);
        root.edges()[1].bind_child_key(right_key);
        root.reserve_edge(0).expect("root edge").complete();
        root.reserve_edge(1).expect("root edge").complete();

        for key in [left_key, right_key] {
            let node = repository.get_or_insert(key);
            assert!(node.try_begin_evaluation());
            node.publish_edges(vec![(first, 1.0)]);
            node.edges()[0].bind_child_key(shared_key);
            node.reserve_edge(0).expect("child edge").complete();
        }
        repository.get_or_insert(shared_key);

        let measure = measure_graph(&repository, root_key);

        assert_eq!(measure.nodes.len(), 4);
        assert_eq!(measure.edges, 4);
        assert_eq!(measure.inbound_edges[&shared_key], 2);
        assert_eq!(measure.path_terminal_edges, 0);
    }
}
