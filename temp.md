主要是分析, policy 和辅助头的训练情况:

(.venv) PS C:\projects\77xiangqi_engine\nn> python scripts/train/train_policy.py --train-dir ../data/xrsh_train --val-dir ../data/xrsh_val --vocab ../data/move_vocab.json --out ../data/checkpoints/policy_v3.pt --batch-size 512 --epochs 40 --val-every 5 --device cuda
torch 2.11.0+cu128 | cuda.is_available=True | device=cuda (NVIDIA GeForce RTX 5070 Ti)
train rows=3945784 val rows=1722978 vocab=2238
train eager cache: hit
value filtering: 已自动跳过未知结局样本 (train=0, val=938)
train data: XRSH（Rust .xrsh；合法着已物化为下标；v3 含结局字段供 value）
dataset mode: train=eager | val=lazy
val data: XRSH（与 train 同 major 版本即可）
train recipe: game-batch | mirror_p=0.5 | fen weight 1/sqrt(count) | label_smooth=0.08 | lr warmup=1ep + cosine→1e-05
aux heads: BCE | attack_scale=0.25 | aux_loss_weight=0.2
aux target stats: attack μ=0.0240 σ=0.0547 | danger μ=0.1626 σ=0.0710 | tactical μ=0.0333 σ=0.0469
aux loss shaping: pos_weight(atk/dan/tac)=(6.37/2.27/5.38) | const-baseline≈0.2749
value head: 默认开启 | tanh MSE vs 结局×progress^1.50 | value_loss_weight=0.5 | target_weight_alpha=1.50
value target stats: μ=0.0012 σ=0.2546 | mean|v|=0.1415 | zero-baseline-mse≈0.0648
警告: Windows 上 train_num_workers>0 会对大 XRSH Dataset 走 spawn 多进程复制；若首个 batch 长时间卡在 0/xxxx，建议先降 train worker 或改 lazy
DataLoader train batches/epoch≈7707 train_num_workers=8 val_num_workers=0 pin_memory=True
amp: off | val_every=5
parameters=2,670,146 (~10.68 MiB fp32 权重)
续训: 已加载 ..\data\checkpoints\policy_v3.pt | 已完成 epoch 计数=20 | 本次将再训练 40 个 epoch（至 epoch 60）
提示: 续训已按总 epoch 重建 scheduler，避免原余弦周期在到达最小 lr 后回升
epoch 21/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [04:13<00:00, 30.35it/s]
train loss 2.4241 lr=2.32e-04
skip val: epoch 21/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 22/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:56<00:00, 32.55it/s]
train loss 2.4143 lr=2.25e-04
skip val: epoch 22/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 23/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.69it/s]
train loss 2.3899 lr=2.18e-04
skip val: epoch 23/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 24/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:56<00:00, 32.65it/s]
train loss 2.3651 lr=2.11e-04
skip val: epoch 24/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 25/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.79it/s]
train loss 2.3412 lr=2.04e-04
val loss 1.7540 acc 0.4899 | val_aux_bce 0.5769 | atk 0.1657 | dan 0.4872 | tac 0.2092 | val_value_mse 0.0598
val human_NLL mean=1.7540 std=1.4094 | entropy(nat) mean=1.9615 | top1=0.4899 top3=0.7361 top5=0.8254
val by ply bin:
  ply0-19 n=827851 NLL=1.2677+-1.1849 H=1.6255 top1=0.6025 top3=0.8658
  ply20-39 n=533700 NLL=2.0359+-1.4703 H=2.1602 top1=0.4322 top3=0.6663
  ply40-59 n=249371 NLL=2.4252+-1.4110 H=2.4245 top1=0.3285 top3=0.5551
  ply60+ n=112056 NLL=2.5106+-1.3209 H=2.4670 top1=0.2913 top3=0.5138
val by pgn_source:
   n=1722978 NLL=1.7540+-1.4094 H=1.9615 top1=0.4899 top3=0.7361
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 26/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:56<00:00, 32.65it/s]
train loss 2.3174 lr=1.97e-04
skip val: epoch 26/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 27/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.76it/s]
train loss 2.2945 lr=1.89e-04
skip val: epoch 27/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 28/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.72it/s]
train loss 2.2716 lr=1.82e-04
skip val: epoch 28/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 29/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.68it/s]
train loss 2.2499 lr=1.74e-04
skip val: epoch 29/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 30/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.79it/s]
train loss 2.2285 lr=1.67e-04
val loss 1.6891 acc 0.5198 | val_aux_bce 0.5766 | atk 0.1703 | dan 0.4824 | tac 0.2119 | val_value_mse 0.0596
val human_NLL mean=1.6891 std=1.4696 | entropy(nat) mean=1.8254 | top1=0.5198 top3=0.7529 top5=0.8362
val by ply bin:
  ply0-19 n=827851 NLL=1.2095+-1.2359 H=1.4883 top1=0.6387 top3=0.8775
  ply20-39 n=533700 NLL=1.9578+-1.5424 H=2.0172 top1=0.4586 top3=0.6877
  ply40-59 n=249371 NLL=2.3610+-1.4988 H=2.2934 top1=0.3506 top3=0.5771
  ply60+ n=112056 NLL=2.4565+-1.3993 H=2.3614 top1=0.3102 top3=0.5341
val by pgn_source:
   n=1722978 NLL=1.6891+-1.4696 H=1.8254 top1=0.5198 top3=0.7529
checkpoint -> ..\data\checkpoints\policy_v3.pt
best checkpoint -> ..\data\checkpoints\policy_v3.best.pt (val_loss=1.6891 epoch=30/60)

(.venv) PS C:\projects\77xiangqi_engine\nn> python scripts/train/train_policy.py --train-dir ../data/xrsh_train --val-dir ../data/xrsh_val --vocab ../data/move_vocab.json --out ../data/checkpoints/policy_v3.pt --batch-size 512 --epochs 40 --val-every 5 --device cuda
torch 2.11.0+cu128 | cuda.is_available=True | device=cuda (NVIDIA GeForce RTX 5070 Ti)
train rows=3945784 val rows=1722978 vocab=2238
train eager cache: hit
value filtering: 已自动跳过未知结局样本 (train=0, val=938)
train data: XRSH（Rust .xrsh；合法着已物化为下标；v3 含结局字段供 value）
dataset mode: train=eager | val=lazy
val data: XRSH（与 train 同 major 版本即可）
train recipe: game-batch | mirror_p=0.5 | fen weight 1/sqrt(count) | label_smooth=0.08 | lr warmup=1ep + cosine→1e-05
aux heads: BCE | attack_scale=0.25 | aux_loss_weight=0.2
aux target stats: attack μ=0.0240 σ=0.0547 | danger μ=0.1626 σ=0.0710 | tactical μ=0.0333 σ=0.0469
aux loss shaping: pos_weight(atk/dan/tac)=(6.37/2.27/5.38) | const-baseline≈0.2749
value head: 默认开启 | tanh MSE vs 结局×progress^1.50 | value_loss_weight=0.5 | target_weight_alpha=1.50
value target stats: μ=0.0012 σ=0.2546 | mean|v|=0.1415 | zero-baseline-mse≈0.0648
警告: Windows 上 train_num_workers>0 会对大 XRSH Dataset 走 spawn 多进程复制；若首个 batch 长时间卡在 0/xxxx，建议先降 train worker 或改 lazy
DataLoader train batches/epoch≈7707 train_num_workers=8 val_num_workers=0 pin_memory=True
amp: off | val_every=5
parameters=2,670,146 (~10.68 MiB fp32 权重)
续训: 已加载 ..\data\checkpoints\policy_v3.pt | 已完成 epoch 计数=20 | 本次将再训练 40 个 epoch（至 epoch 60）
提示: 续训已按总 epoch 重建 scheduler，避免原余弦周期在到达最小 lr 后回升
epoch 21/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [04:13<00:00, 30.35it/s]
train loss 2.4241 lr=2.32e-04
skip val: epoch 21/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 22/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:56<00:00, 32.55it/s]
train loss 2.4143 lr=2.25e-04
skip val: epoch 22/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 23/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.69it/s]
train loss 2.3899 lr=2.18e-04
skip val: epoch 23/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 24/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:56<00:00, 32.65it/s]
train loss 2.3651 lr=2.11e-04
skip val: epoch 24/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 25/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.79it/s]
train loss 2.3412 lr=2.04e-04
val loss 1.7540 acc 0.4899 | val_aux_bce 0.5769 | atk 0.1657 | dan 0.4872 | tac 0.2092 | val_value_mse 0.0598
val human_NLL mean=1.7540 std=1.4094 | entropy(nat) mean=1.9615 | top1=0.4899 top3=0.7361 top5=0.8254
val by ply bin:
  ply0-19 n=827851 NLL=1.2677+-1.1849 H=1.6255 top1=0.6025 top3=0.8658
  ply20-39 n=533700 NLL=2.0359+-1.4703 H=2.1602 top1=0.4322 top3=0.6663
  ply40-59 n=249371 NLL=2.4252+-1.4110 H=2.4245 top1=0.3285 top3=0.5551
  ply60+ n=112056 NLL=2.5106+-1.3209 H=2.4670 top1=0.2913 top3=0.5138
val by pgn_source:
   n=1722978 NLL=1.7540+-1.4094 H=1.9615 top1=0.4899 top3=0.7361
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 26/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:56<00:00, 32.65it/s]
train loss 2.3174 lr=1.97e-04
skip val: epoch 26/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 27/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.76it/s]
train loss 2.2945 lr=1.89e-04
skip val: epoch 27/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 28/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.72it/s]
train loss 2.2716 lr=1.82e-04
skip val: epoch 28/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 29/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.68it/s]
train loss 2.2499 lr=1.74e-04
skip val: epoch 29/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 30/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [03:55<00:00, 32.79it/s]
train loss 2.2285 lr=1.67e-04
val loss 1.6891 acc 0.5198 | val_aux_bce 0.5766 | atk 0.1703 | dan 0.4824 | tac 0.2119 | val_value_mse 0.0596
val human_NLL mean=1.6891 std=1.4696 | entropy(nat) mean=1.8254 | top1=0.5198 top3=0.7529 top5=0.8362
val by ply bin:
  ply0-19 n=827851 NLL=1.2095+-1.2359 H=1.4883 top1=0.6387 top3=0.8775
  ply20-39 n=533700 NLL=1.9578+-1.5424 H=2.0172 top1=0.4586 top3=0.6877
  ply40-59 n=249371 NLL=2.3610+-1.4988 H=2.2934 top1=0.3506 top3=0.5771
  ply60+ n=112056 NLL=2.4565+-1.3993 H=2.3614 top1=0.3102 top3=0.5341
val by pgn_source:
   n=1722978 NLL=1.6891+-1.4696 H=1.8254 top1=0.5198 top3=0.7529
checkpoint -> ..\data\checkpoints\policy_v3.pt
best checkpoint -> ..\data\checkpoints\policy_v3.best.pt (val_loss=1.6891 epoch=30/60)
epoch 31/60 train:  24%|█████████████████▍                                                       | 1847/7707 [00:57<02:58, 32.88it/s]epoch 31/60 train:  24%|█████████████████▌                                                       | 1851/7707 [00:57<03:00, 32.51it/epoch 31/60 train:  24%|█████████████████▌                                                       | 1855/7707 [00:57<03:00, 32.47it/sepoch 31/60 train:  24%|█████████████████▌                                                       | 1859/7707 [00:57<02:59, 32.51it/sepoch 31/60 train:  24%|█████████████████▋                                                       | 1863/7707 [00:57<02:59, 32.54it/sepochepoch 31/6epochepoch 31/60 traepoch 31/60 traepoch 31/60 traepochepoch 31/6epoch 31/60 train:  epoch 31/60 train: 100%|█████████████████████████████████████████████████████████████████████████| 7707/7707 [04:16<00:00, 30.10it/s]
train loss 2.2069 lr=1.59e-04
skip val: epoch 31/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 32/60 train: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 7707/7707 [04:34<00:00, 28.07it/s]
train loss 2.1859 lr=1.51e-04
skip val: epoch 32/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 33/60 train: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 7707/7707 [04:35<00:00, 27.93it/s]
train loss 2.1660 lr=1.43e-04
skip val: epoch 33/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 34/60 train: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 7707/7707 [04:42<00:00, 27.31it/s]
train loss 2.1460 lr=1.36e-04
skip val: epoch 34/60（--val-every=5）
checkpoint -> ..\data\checkpoints\policy_v3.pt
epoch 35/60 train: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 7707/7707 [04:36<00:00, 27.83it/s]
train loss 2.1260 lr=1.28e-04
val loss 1.6603 acc 0.5325 | val_aux_bce 0.5772 | atk 0.1839 | dan 0.4883 | tac 0.2257 | val_value_mse 0.0594
val human_NLL mean=1.6603 std=1.4791 | entropy(nat) mean=1.8123 | top1=0.5325 top3=0.7630 top5=0.8425
val by ply bin:
  ply0-19 n=827851 NLL=1.1911+-1.2192 H=1.5079 top1=0.6462 top3=0.8841
  ply20-39 n=533700 NLL=1.9186+-1.5726 H=1.9761 top1=0.4763 top3=0.6997
  ply40-59 n=249371 NLL=2.3224+-1.5402 H=2.2485 top1=0.3679 top3=0.5921
  ply60+ n=112056 NLL=2.4223+-1.4485 H=2.3110 top1=0.3269 top3=0.5500
val by pgn_source:
   n=1722978 NLL=1.6603+-1.4791 H=1.8123 top1=0.5325 top3=0.7630
checkpoint -> ..\data\checkpoints\policy_v3.pt