//! 小评测集：`nn-leaf off` vs `main` 的 `bestmove` 对照（见任务 3 / `docs/value-probe.md`）。

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use xiangqi_core::Position;

use crate::eval::{NNLeafMode, NnEvalSession};
use crate::policy_onnx::PolicyOnnx;
use crate::search::{root_search_iterative, RootSearchShared, SearchAblation, SearchLimits};
use crate::tt::TranspositionTable;

/// 固定局面：注释为人读标签，不参与搜索。
pub struct ValueProbeCase {
    pub id: &'static str,
    pub fen: &'static str,
    pub kind: &'static str,
}

pub const VALUE_PROBE_CASES: &[ValueProbeCase] = &[
    ValueProbeCase {
        id: "start",
        fen: "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        kind: "开局静面",
    },
    ValueProbeCase {
        id: "rook_lift",
        fen: "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/R8/1NBAKABNR b - - 1 1",
        kind: "车抬头后对顶",
    },
    ValueProbeCase {
        id: "endgame_kings_blocked",
        // 双将在 e 路但中间有红兵垫住，避免将帅照面；棋规合法、可达类残面。
        fen: "4k4/9/9/9/4P4/9/9/9/4K4/9 w - - 0 1",
        kind: "残局（中路有子隔断，无照面）",
    },
    ValueProbeCase {
        id: "black_down_bishop",
        fen: "rn1akabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        kind: "黑缺象（子力失衡）",
    },
];

fn verdict(off_bm: &str, main_bm: &str) -> &'static str {
    if off_bm == main_bm {
        "一样"
    } else {
        "不同"
    }
}

/// 生成 Markdown 表的输入（避免过长参数列表）。
pub struct ValueProbeTableArgs<'a> {
    pub depth: u32,
    pub hash_mb: usize,
    pub policy_ordering: bool,
    pub nn_eval_budget: u64,
    pub onnx_path: &'a Path,
    pub policy: &'a Arc<Mutex<Option<PolicyOnnx>>>,
    pub vocab: &'a HashMap<String, usize>,
    pub vocab_size: usize,
}

/// 生成 Markdown 表：`bestmove` 对照 + 「一样/不同」；**更好/更差** 需人填（见文档）。
///
/// `onnx_path`：已成功加载的模型路径，写入表头以免与「双侧均未用 value」混淆。
pub fn markdown_table_off_vs_main(args: &ValueProbeTableArgs<'_>) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("# Value 小评测（`--nn-leaf off` vs `main`）\n\n");
    out.push_str("> **口径**：本工具**仅**在成功加载 ONNX 后运行；`MainLeafOnly` 在门控允许时使用 value 头，与 `Off`（物质叶值）形成对照。若未加载模型，CLI 会直接退出，避免出现「假对比」。\n\n");
    out.push_str(&format!("> - ONNX: `{}`\n\n", args.onnx_path.display()));
    out.push_str(&format!(
        "- depth: {}, hash_mb: {}, policy_ordering: {}, nn_eval_budget: {}\n\n",
        args.depth, args.hash_mb, args.policy_ordering, args.nn_eval_budget
    ));
    out.push_str("| id | 类型 | off bestmove | main bestmove | 自动判定 | 人填更好/更差 |\n");
    out.push_str("|----|------|--------------|---------------|----------|----------------|\n");

    for c in VALUE_PROBE_CASES {
        let mut pos_off = Position::from_fen(c.fen).map_err(|e| format!("{}: {e}", c.id))?;
        let mut tt_off = TranspositionTable::new(args.hash_mb);
        let mut nn_off = NnEvalSession {
            nn_eval_budget: args.nn_eval_budget,
            ..Default::default()
        };
        let mut shared_off = RootSearchShared {
            policy: args.policy,
            vocab: args.vocab,
            vocab_size: args.vocab_size,
            tt: &mut tt_off,
            stop: None,
            ablation: SearchAblation {
                policy_ordering: args.policy_ordering,
                nn_leaf_mode: NNLeafMode::Off,
            },
            nn_eval: &mut nn_off,
        };
        let bm_off = root_search_iterative(&mut pos_off, args.depth, &mut shared_off, SearchLimits::none())
            .map(|r| r.best_uci)
            .unwrap_or_else(|| "(none)".to_string());

        let mut pos_main = Position::from_fen(c.fen).map_err(|e| format!("{}: {e}", c.id))?;
        let mut tt_main = TranspositionTable::new(args.hash_mb);
        let mut nn_main = NnEvalSession {
            nn_eval_budget: args.nn_eval_budget,
            ..Default::default()
        };
        let mut shared_main = RootSearchShared {
            policy: args.policy,
            vocab: args.vocab,
            vocab_size: args.vocab_size,
            tt: &mut tt_main,
            stop: None,
            ablation: SearchAblation {
                policy_ordering: args.policy_ordering,
                nn_leaf_mode: NNLeafMode::MainLeafOnly,
            },
            nn_eval: &mut nn_main,
        };
        let bm_main = root_search_iterative(&mut pos_main, args.depth, &mut shared_main, SearchLimits::none())
            .map(|r| r.best_uci)
            .unwrap_or_else(|| "(none)".to_string());

        let v = verdict(&bm_off, &bm_main);
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | （待填：更好/一样/更差） |\n",
            c.id, c.kind, bm_off, bm_main, v
        ));
    }

    out.push_str("\n说明：`自动判定` 仅比较 UCI 字符串是否相同；**价值是否更好** 需棋理解或对照强引擎后手写最后一列。\n");
    Ok(out)
}
