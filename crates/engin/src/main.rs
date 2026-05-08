//! 面向用户的 **UCI 引擎** 入口：包装搜索、联动象棋核心与神经网络推理。
//! 数据标注、数据集打包等请使用 **`xiangqi_dataset`** crate，不包含在本二进制中。

use xiangqi_core::{legal_moves_uci, Position, START_FEN};

fn main() {
    let pos = Position::from_fen(START_FEN).expect("startpos");
    let _n = legal_moves_uci(&pos);
    println!(
        "engin: UCI engine placeholder (xiangqi_core linked). \
         Wire Alpha-Beta / TT / ONNX policy here for distribution."
    );
}
