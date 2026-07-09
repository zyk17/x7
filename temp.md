(.venv) PS C:\projects\77xiangqi_engine> python.exe nn\scripts\train\train_px0.py `
>>   --px0-version 710 `
>>   --init-from data\checkpoints\baseline_px0_katago_v1.best.pt `
>>   --out data\checkpoints\baseline_px0_katago_v1_qmix050.pt `
>>   --width 128 `
>>   --blocks 8 `
>>   --batch-size 256 `
>>   --steps 100000 `
>>   --eval-every 1000 `
>>   --val-batches 64 `
>>   --num-workers 4 `
>>   --device cuda `
>>   --q-ratio 0.5
torch 2.11.0+cu128 | cuda.is_available=True | device=cuda
init from data\checkpoints\baseline_px0_katago_v1.best.pt | start new phase with q_ratio=0.500
px0: train_files=360092 val_files=40010 batch_size=256 steps=100000 q_ratio=0.500
px0_kaggle: version=710 root=C:\work\px0data val_ratio=0.100
step 1/100000 train_loss=3.1961 train_policy=2.6877 train_value_ce=0.5016 train_value_q_mse=0.0258 train_moves_left=0.0022 val_loss=3.3713 val_policy=2.4178 val_value_ce=0.9285 val_value_q_mse=0.0944 val_moves_left=0.0092 lr=1.00e-03
step 1000/100000 train_loss=4.0644 train_policy=2.7162 train_value_ce=1.3316 train_value_q_mse=0.0651 train_moves_left=0.0024 val_loss=3.5427 val_policy=2.4692 val_value_ce=1.0402 val_value_q_mse=0.1258 val_moves_left=0.0116 lr=1.00e-03
step 2000/100000 train_loss=3.1528 train_policy=2.7695 train_value_ce=0.3816 train_value_q_mse=0.0065 train_moves_left=0.0004 val_loss=3.5198 val_policy=2.4613 val_value_ce=1.0251 val_value_q_mse=0.1266 val_moves_left=0.0115 lr=9.99e-04
step 3000/100000 train_loss=3.1505 train_policy=2.5056 train_value_ce=0.6309 train_value_q_mse=0.0542 train_moves_left=0.0031 val_loss=3.4826 val_policy=2.4573 val_value_ce=0.9939 val_value_q_mse=0.1206 val_moves_left=0.0080 lr=9.98e-04
step 4000/100000 train_loss=3.3429 train_policy=2.6889 train_value_ce=0.6367 train_value_q_mse=0.0659 train_moves_left=0.0055 val_loss=3.4680 val_policy=2.4668 val_value_ce=0.9727 val_value_q_mse=0.1097 val_moves_left=0.0078 lr=9.96e-04
step 5000/100000 train_loss=3.4245 train_policy=2.4869 train_value_ce=0.9070 train_value_q_mse=0.1192 train_moves_left=0.0058 val_loss=3.4756 val_policy=2.4645 val_value_ce=0.9811 val_value_q_mse=0.1139 val_moves_left=0.0099 lr=9.94e-04
step 6000/100000 train_loss=2.7540 train_policy=2.4749 train_value_ce=0.2750 train_value_q_mse=0.0152 train_moves_left=0.0024 val_loss=3.5285 val_policy=2.4723 val_value_ce=1.0255 val_value_q_mse=0.1163 val_moves_left=0.0112 lr=9.91e-04
step 7000/100000 train_loss=3.2318 train_policy=2.5405 train_value_ce=0.6775 train_value_q_mse=0.0536 train_moves_left=0.0029 val_loss=3.5729 val_policy=2.4764 val_value_ce=1.0588 val_value_q_mse=0.1416 val_moves_left=0.0149 lr=9.88e-04
step 8000/100000 train_loss=3.2447 train_policy=2.4144 train_value_ce=0.8224 train_value_q_mse=0.0310 train_moves_left=0.0015 val_loss=3.4943 val_policy=2.4646 val_value_ce=0.9977 val_value_q_mse=0.1205 val_moves_left=0.0120 lr=9.84e-04
step 9000/100000 train_loss=3.6079 train_policy=2.6305 train_value_ce=0.9500 train_value_q_mse=0.1080 train_moves_left=0.0028 val_loss=3.4634 val_policy=2.4557 val_value_ce=0.9783 val_value_q_mse=0.1123 val_moves_left=0.0088 lr=9.80e-04
step 10000/100000 train_loss=3.3246 train_policy=2.4834 train_value_ce=0.8184 train_value_q_mse=0.0898 train_moves_left=0.0024 val_loss=3.4795 val_policy=2.4605 val_value_ce=0.9890 val_value_q_mse=0.1129 val_moves_left=0.0121 lr=9.76e-04
step 11000/100000 train_loss=3.2438 train_policy=2.5011 train_value_ce=0.7295 train_value_q_mse=0.0486 train_moves_left=0.0063 val_loss=3.5077 val_policy=2.4568 val_value_ce=1.0169 val_value_q_mse=0.1303 val_moves_left=0.0096 lr=9.71e-04
step 12000/100000 train_loss=3.5621 train_policy=2.5265 train_value_ce=1.0020 train_value_q_mse=0.1325 train_moves_left=0.0035 val_loss=3.4759 val_policy=2.4509 val_value_ce=0.9946 val_value_q_mse=0.1154 val_moves_left=0.0105 lr=9.65e-04
step 13000/100000 train_loss=3.1501 train_policy=2.3374 train_value_ce=0.7819 train_value_q_mse=0.1207 train_moves_left=0.0044 val_loss=3.4636 val_policy=2.4512 val_value_ce=0.9809 val_value_q_mse=0.1202 val_moves_left=0.0099 lr=9.59e-04
step 14000/100000 train_loss=3.4302 train_policy=2.4338 train_value_ce=0.9823 train_value_q_mse=0.0549 train_moves_left=0.0020 val_loss=3.4703 val_policy=2.4510 val_value_ce=0.9882 val_value_q_mse=0.1184 val_moves_left=0.0097 lr=9.53e-04
step 15000/100000 train_loss=3.1881 train_policy=2.5511 train_value_ce=0.6096 train_value_q_mse=0.1088 train_moves_left=0.0008 val_loss=3.4378 val_policy=2.4500 val_value_ce=0.9582 val_value_q_mse=0.1128 val_moves_left=0.0091 lr=9.46e-04
step 16000/100000 train_loss=3.8844 train_policy=2.5531 train_value_ce=1.3254 train_value_q_mse=0.0225 train_moves_left=0.0024 val_loss=3.5050 val_policy=2.4426 val_value_ce=1.0292 val_value_q_mse=0.1256 val_moves_left=0.0116 lr=9.39e-04
step 17000/100000 train_loss=3.1119 train_policy=2.6405 train_value_ce=0.4693 train_value_q_mse=0.0079 train_moves_left=0.0006 val_loss=3.5217 val_policy=2.4506 val_value_ce=1.0383 val_value_q_mse=0.1237 val_moves_left=0.0122 lr=9.31e-04
step 18000/100000 train_loss=3.5706 train_policy=2.5591 train_value_ce=0.9662 train_value_q_mse=0.1714 train_moves_left=0.0163 val_loss=3.4775 val_policy=2.4406 val_value_ce=1.0069 val_value_q_mse=0.1134 val_moves_left=0.0109 lr=9.23e-04
step 19000/100000 train_loss=3.3424 train_policy=2.4137 train_value_ce=0.9067 train_value_q_mse=0.0802 train_moves_left=0.0129 val_loss=3.4787 val_policy=2.4407 val_value_ce=1.0079 val_value_q_mse=0.1151 val_moves_left=0.0092 lr=9.14e-04
step 20000/100000 train_loss=3.0329 train_policy=2.6041 train_value_ce=0.4136 train_value_q_mse=0.0576 train_moves_left=0.0049 val_loss=3.4541 val_policy=2.4394 val_value_ce=0.9850 val_value_q_mse=0.1137 val_moves_left=0.0092 lr=9.05e-04
step 21000/100000 train_loss=2.9927 train_policy=2.6670 train_value_ce=0.3241 train_value_q_mse=0.0062 train_moves_left=0.0004 val_loss=3.5293 val_policy=2.4561 val_value_ce=1.0390 val_value_q_mse=0.1270 val_moves_left=0.0156 lr=8.96e-04
step 22000/100000 train_loss=3.3039 train_policy=2.7764 train_value_ce=0.5090 train_value_q_mse=0.0647 train_moves_left=0.0153 val_loss=3.4857 val_policy=2.4379 val_value_ce=1.0173 val_value_q_mse=0.1171 val_moves_left=0.0080 lr=8.86e-04
step 23000/100000 train_loss=3.0344 train_policy=2.7117 train_value_ce=0.3127 train_value_q_mse=0.0346 train_moves_left=0.0093 val_loss=3.4429 val_policy=2.4473 val_value_ce=0.9673 val_value_q_mse=0.1061 val_moves_left=0.0121 lr=8.76e-04
step 24000/100000 train_loss=2.9511 train_policy=2.3521 train_value_ce=0.5882 train_value_q_mse=0.0337 train_moves_left=0.0157 val_loss=3.5054 val_policy=2.4397 val_value_ce=1.0353 val_value_q_mse=0.1154 val_moves_left=0.0107 lr=8.66e-04
step 25000/100000 train_loss=3.0066 train_policy=2.7836 train_value_ce=0.2222 train_value_q_mse=0.0031 train_moves_left=0.0005 val_loss=3.5553 val_policy=2.4400 val_value_ce=1.0790 val_value_q_mse=0.1377 val_moves_left=0.0125 lr=8.55e-04
step 26000/100000 train_loss=3.0663 train_policy=2.5565 train_value_ce=0.5003 train_value_q_mse=0.0374 train_moves_left=0.0010 val_loss=3.4852 val_policy=2.4280 val_value_ce=1.0272 val_value_q_mse=0.1136 val_moves_left=0.0103 lr=8.44e-04
step 27000/100000 train_loss=2.8539 train_policy=2.4511 train_value_ce=0.3980 train_value_q_mse=0.0185 train_moves_left=0.0005 val_loss=3.4714 val_policy=2.4336 val_value_ce=1.0072 val_value_q_mse=0.1164 val_moves_left=0.0096 lr=8.32e-04
step 28000/100000 train_loss=3.1705 train_policy=2.4531 train_value_ce=0.6890 train_value_q_mse=0.1123 train_moves_left=0.0020 val_loss=3.4591 val_policy=2.4292 val_value_ce=1.0004 val_value_q_mse=0.1125 val_moves_left=0.0090 lr=8.21e-04
step 29000/100000 train_loss=2.9458 train_policy=2.2429 train_value_ce=0.6741 train_value_q_mse=0.1126 train_moves_left=0.0048 val_loss=3.4334 val_policy=2.4285 val_value_ce=0.9769 val_value_q_mse=0.1057 val_moves_left=0.0100 lr=8.08e-04
step 30000/100000 train_loss=3.1140 train_policy=2.7192 train_value_ce=0.3920 train_value_q_mse=0.0096 train_moves_left=0.0026 val_loss=3.4152 val_policy=2.4199 val_value_ce=0.9668 val_value_q_mse=0.1091 val_moves_left=0.0085 lr=7.96e-04
step 31000/100000 train_loss=3.4462 train_policy=2.3785 train_value_ce=1.0447 train_value_q_mse=0.0914 train_moves_left=0.0013 val_loss=3.4615 val_policy=2.4295 val_value_ce=1.0027 val_value_q_mse=0.1099 val_moves_left=0.0119 lr=7.83e-04
step 32000/100000 train_loss=2.9691 train_policy=2.5085 train_value_ce=0.4483 train_value_q_mse=0.0476 train_moves_left=0.0030 val_loss=3.4314 val_policy=2.4236 val_value_ce=0.9773 val_value_q_mse=0.1163 val_moves_left=0.0096 lr=7.70e-04
step 33000/100000 train_loss=3.3003 train_policy=2.5735 train_value_ce=0.6958 train_value_q_mse=0.1214 train_moves_left=0.0039 val_loss=3.4048 val_policy=2.4214 val_value_ce=0.9570 val_value_q_mse=0.1002 val_moves_left=0.0095 lr=7.57e-04
step 34000/100000 train_loss=3.0263 train_policy=2.5636 train_value_ce=0.4596 train_value_q_mse=0.0109 train_moves_left=0.0020 val_loss=3.4714 val_policy=2.4218 val_value_ce=1.0190 val_value_q_mse=0.1162 val_moves_left=0.0100 lr=7.43e-04
step 35000/100000 train_loss=3.0843 train_policy=2.7330 train_value_ce=0.3352 train_value_q_mse=0.0611 train_moves_left=0.0061 val_loss=3.4311 val_policy=2.4186 val_value_ce=0.9847 val_value_q_mse=0.1058 val_moves_left=0.0091 lr=7.30e-04
step 36000/100000 train_loss=3.5034 train_policy=2.8225 train_value_ce=0.6564 train_value_q_mse=0.0964 train_moves_left=0.0020 val_loss=3.4720 val_policy=2.4186 val_value_ce=1.0238 val_value_q_mse=0.1118 val_moves_left=0.0110 lr=7.16e-04
step 37000/100000 train_loss=3.0187 train_policy=2.7090 train_value_ce=0.3072 train_value_q_mse=0.0091 train_moves_left=0.0008 val_loss=3.4247 val_policy=2.4133 val_value_ce=0.9839 val_value_q_mse=0.1046 val_moves_left=0.0089 lr=7.02e-04
step 38000/100000 train_loss=3.1470 train_policy=2.6177 train_value_ce=0.5242 train_value_q_mse=0.0186 train_moves_left=0.0030 val_loss=3.3891 val_policy=2.4116 val_value_ce=0.9511 val_value_q_mse=0.1005 val_moves_left=0.0080 lr=6.87e-04
step 39000/100000 train_loss=3.0289 train_policy=2.7898 train_value_ce=0.2376 train_value_q_mse=0.0047 train_moves_left=0.0023 val_loss=3.5426 val_policy=2.4125 val_value_ce=1.0976 val_value_q_mse=0.1222 val_moves_left=0.0127 lr=6.73e-04
step 40000/100000 train_loss=3.3072 train_policy=2.4632 train_value_ce=0.8225 train_value_q_mse=0.0839 train_moves_left=0.0037 val_loss=3.3805 val_policy=2.4068 val_value_ce=0.9471 val_value_q_mse=0.1019 val_moves_left=0.0075 lr=6.58e-04
step 41000/100000 train_loss=3.5667 train_policy=2.4267 train_value_ce=1.1072 train_value_q_mse=0.1302 train_moves_left=0.0017 val_loss=3.3733 val_policy=2.4089 val_value_ce=0.9392 val_value_q_mse=0.0962 val_moves_left=0.0081 lr=6.43e-04
step 42000/100000 train_loss=2.8772 train_policy=2.6943 train_value_ce=0.1813 train_value_q_mse=0.0047 train_moves_left=0.0028 val_loss=3.4664 val_policy=2.4115 val_value_ce=1.0266 val_value_q_mse=0.1068 val_moves_left=0.0103 lr=6.28e-04
step 43000/100000 train_loss=2.8830 train_policy=2.3292 train_value_ce=0.5448 train_value_q_mse=0.0212 train_moves_left=0.0238 val_loss=3.3913 val_policy=2.4055 val_value_ce=0.9596 val_value_q_mse=0.0997 val_moves_left=0.0089 lr=6.13e-04
step 44000/100000 train_loss=3.4883 train_policy=2.5590 train_value_ce=0.9251 train_value_q_mse=0.0165 train_moves_left=0.0004 val_loss=3.4040 val_policy=2.4086 val_value_ce=0.9681 val_value_q_mse=0.1030 val_moves_left=0.0100 lr=5.98e-04
step 45000/100000 train_loss=2.7802 train_policy=2.7198 train_value_ce=0.0602 train_value_q_mse=0.0002 train_moves_left=0.0010 val_loss=3.4268 val_policy=2.4052 val_value_ce=0.9935 val_value_q_mse=0.1050 val_moves_left=0.0122 lr=5.82e-04
step 46000/100000 train_loss=2.5946 train_policy=2.1166 train_value_ce=0.4711 train_value_q_mse=0.0240 train_moves_left=0.0055 val_loss=3.4598 val_policy=2.4076 val_value_ce=1.0215 val_value_q_mse=0.1149 val_moves_left=0.0135 lr=5.67e-04
step 47000/100000 train_loss=3.0626 train_policy=2.6820 train_value_ce=0.3772 train_value_q_mse=0.0122 train_moves_left=0.0026 val_loss=3.4228 val_policy=2.4000 val_value_ce=0.9960 val_value_q_mse=0.1008 val_moves_left=0.0101 lr=5.52e-04
step 48000/100000 train_loss=3.1253 train_policy=2.4234 train_value_ce=0.6933 train_value_q_mse=0.0331 train_moves_left=0.0024 val_loss=3.4472 val_policy=2.4017 val_value_ce=1.0159 val_value_q_mse=0.1122 val_moves_left=0.0099 lr=5.36e-04
step 49000/100000 train_loss=3.1228 train_policy=2.5301 train_value_ce=0.5813 train_value_q_mse=0.0449 train_moves_left=0.0010 val_loss=3.4162 val_policy=2.4029 val_value_ce=0.9854 val_value_q_mse=0.1059 val_moves_left=0.0099 lr=5.21e-04
step 50000/100000 train_loss=3.0827 train_policy=2.3651 train_value_ce=0.7020 train_value_q_mse=0.0596 train_moves_left=0.0047 val_loss=3.3814 val_policy=2.3998 val_value_ce=0.9548 val_value_q_mse=0.1018 val_moves_left=0.0090 lr=5.05e-04
step 51000/100000 train_loss=2.9574 train_policy=2.5643 train_value_ce=0.3846 train_value_q_mse=0.0283 train_moves_left=0.0100 val_loss=3.3959 val_policy=2.3951 val_value_ce=0.9751 val_value_q_mse=0.0976 val_moves_left=0.0082 lr=4.89e-04
step 52000/100000 train_loss=3.7303 train_policy=2.2959 train_value_ce=1.3750 train_value_q_mse=0.2358 train_moves_left=0.0031 val_loss=3.3811 val_policy=2.3915 val_value_ce=0.9637 val_value_q_mse=0.0973 val_moves_left=0.0101 lr=4.74e-04
step 53000/100000 train_loss=2.9051 train_policy=2.5110 train_value_ce=0.3891 train_value_q_mse=0.0176 train_moves_left=0.0037 val_loss=3.3869 val_policy=2.3939 val_value_ce=0.9669 val_value_q_mse=0.0986 val_moves_left=0.0098 lr=4.58e-04
step 54000/100000 train_loss=2.5037 train_policy=2.4280 train_value_ce=0.0745 train_value_q_mse=0.0014 train_moves_left=0.0051 val_loss=3.4204 val_policy=2.4013 val_value_ce=0.9906 val_value_q_mse=0.1077 val_moves_left=0.0103 lr=4.43e-04
step 55000/100000 train_loss=3.1066 train_policy=2.6596 train_value_ce=0.4410 train_value_q_mse=0.0221 train_moves_left=0.0031 val_loss=3.4504 val_policy=2.3986 val_value_ce=1.0235 val_value_q_mse=0.1058 val_moves_left=0.0119 lr=4.28e-04
step 56000/100000 train_loss=3.2412 train_policy=2.6572 train_value_ce=0.5634 train_value_q_mse=0.0818 train_moves_left=0.0010 val_loss=3.4171 val_policy=2.3898 val_value_ce=1.0006 val_value_q_mse=0.1019 val_moves_left=0.0080 lr=4.12e-04
step 57000/100000 train_loss=2.7023 train_policy=2.1907 train_value_ce=0.4909 train_value_q_mse=0.0795 train_moves_left=0.0058 val_loss=3.3571 val_policy=2.3893 val_value_ce=0.9421 val_value_q_mse=0.0976 val_moves_left=0.0084 lr=3.97e-04
step 58000/100000 train_loss=3.1993 train_policy=2.4907 train_value_ce=0.6976 train_value_q_mse=0.0424 train_moves_left=0.0026 val_loss=3.3700 val_policy=2.3859 val_value_ce=0.9588 val_value_q_mse=0.0958 val_moves_left=0.0093 lr=3.82e-04
step 59000/100000 train_loss=3.1339 train_policy=2.3949 train_value_ce=0.7188 train_value_q_mse=0.0800 train_moves_left=0.0018 val_loss=3.4122 val_policy=2.3920 val_value_ce=0.9935 val_value_q_mse=0.0998 val_moves_left=0.0118 lr=3.67e-04
step 60000/100000 train_loss=3.2059 train_policy=2.4522 train_value_ce=0.7419 train_value_q_mse=0.0452 train_moves_left=0.0027 val_loss=3.4210 val_policy=2.3901 val_value_ce=1.0036 val_value_q_mse=0.1028 val_moves_left=0.0109 lr=3.52e-04
step 61000/100000 train_loss=3.1647 train_policy=2.7017 train_value_ce=0.4589 train_value_q_mse=0.0149 train_moves_left=0.0026 val_loss=3.3930 val_policy=2.3990 val_value_ce=0.9679 val_value_q_mse=0.0983 val_moves_left=0.0106 lr=3.37e-04
step 62000/100000 train_loss=2.8725 train_policy=2.6607 train_value_ce=0.2076 train_value_q_mse=0.0158 train_moves_left=0.0009 val_loss=3.4133 val_policy=2.3892 val_value_ce=0.9979 val_value_q_mse=0.0997 val_moves_left=0.0089 lr=3.23e-04
step 63000/100000 train_loss=3.0752 train_policy=2.5488 train_value_ce=0.5201 train_value_q_mse=0.0231 train_moves_left=0.0037 val_loss=3.3728 val_policy=2.3819 val_value_ce=0.9664 val_value_q_mse=0.0921 val_moves_left=0.0098 lr=3.08e-04
step 64000/100000 train_loss=2.9233 train_policy=2.5408 train_value_ce=0.3755 train_value_q_mse=0.0278 train_moves_left=0.0001 val_loss=3.3736 val_policy=2.3816 val_value_ce=0.9668 val_value_q_mse=0.0947 val_moves_left=0.0102 lr=2.94e-04
step 65000/100000 train_loss=2.5856 train_policy=2.2422 train_value_ce=0.3390 train_value_q_mse=0.0161 train_moves_left=0.0023 val_loss=3.4174 val_policy=2.3836 val_value_ce=1.0070 val_value_q_mse=0.1011 val_moves_left=0.0100 lr=2.80e-04
step 66000/100000 train_loss=3.1414 train_policy=2.4962 train_value_ce=0.6320 train_value_q_mse=0.0512 train_moves_left=0.0032 val_loss=3.3436 val_policy=2.3836 val_value_ce=0.9346 val_value_q_mse=0.0957 val_moves_left=0.0093 lr=2.67e-04
step 67000/100000 train_loss=3.1762 train_policy=2.1730 train_value_ce=0.9884 train_value_q_mse=0.0550 train_moves_left=0.0066 val_loss=3.3482 val_policy=2.3752 val_value_ce=0.9487 val_value_q_mse=0.0915 val_moves_left=0.0094 lr=2.53e-04
step 68000/100000 train_loss=3.2619 train_policy=2.4238 train_value_ce=0.8210 train_value_q_mse=0.0663 train_moves_left=0.0038 val_loss=3.3620 val_policy=2.3781 val_value_ce=0.9591 val_value_q_mse=0.0929 val_moves_left=0.0100 lr=2.40e-04
step 69000/100000 train_loss=2.9740 train_policy=2.5714 train_value_ce=0.3981 train_value_q_mse=0.0173 train_moves_left=0.0017 val_loss=3.3819 val_policy=2.3786 val_value_ce=0.9770 val_value_q_mse=0.0992 val_moves_left=0.0101 lr=2.27e-04
step 70000/100000 train_loss=3.2676 train_policy=2.3377 train_value_ce=0.9127 train_value_q_mse=0.0642 train_moves_left=0.0078 val_loss=3.3412 val_policy=2.3750 val_value_ce=0.9425 val_value_q_mse=0.0901 val_moves_left=0.0078 lr=2.14e-04
step 71000/100000 train_loss=2.8788 train_policy=2.4037 train_value_ce=0.4661 train_value_q_mse=0.0349 train_moves_left=0.0019 val_loss=3.4017 val_policy=2.3807 val_value_ce=0.9946 val_value_q_mse=0.0982 val_moves_left=0.0123 lr=2.02e-04
step 72000/100000 train_loss=3.0835 train_policy=2.5176 train_value_ce=0.5458 train_value_q_mse=0.0688 train_moves_left=0.0189 val_loss=3.3502 val_policy=2.3765 val_value_ce=0.9494 val_value_q_mse=0.0912 val_moves_left=0.0103 lr=1.89e-04
step 73000/100000 train_loss=4.0874 train_policy=2.6481 train_value_ce=1.4257 train_value_q_mse=0.0487 train_moves_left=0.0094 val_loss=3.3475 val_policy=2.3728 val_value_ce=0.9511 val_value_q_mse=0.0887 val_moves_left=0.0093 lr=1.78e-04
step 74000/100000 train_loss=2.8685 train_policy=2.3508 train_value_ce=0.4997 train_value_q_mse=0.0712 train_moves_left=0.0010 val_loss=3.3713 val_policy=2.3736 val_value_ce=0.9719 val_value_q_mse=0.0966 val_moves_left=0.0110 lr=1.66e-04
step 75000/100000 train_loss=3.1229 train_policy=2.3800 train_value_ce=0.7321 train_value_q_mse=0.0400 train_moves_left=0.0050 val_loss=3.3671 val_policy=2.3716 val_value_ce=0.9703 val_value_q_mse=0.0955 val_moves_left=0.0091 lr=1.55e-04
step 76000/100000 train_loss=3.0928 train_policy=2.4221 train_value_ce=0.6571 train_value_q_mse=0.0516 train_moves_left=0.0045 val_loss=3.3480 val_policy=2.3681 val_value_ce=0.9551 val_value_q_mse=0.0935 val_moves_left=0.0092 lr=1.44e-04
step 77000/100000 train_loss=3.1848 train_policy=2.2879 train_value_ce=0.8755 train_value_q_mse=0.0823 train_moves_left=0.0057 val_loss=3.3792 val_policy=2.3718 val_value_ce=0.9819 val_value_q_mse=0.0952 val_moves_left=0.0112 lr=1.34e-04
step 78000/100000 train_loss=3.1349 train_policy=2.6560 train_value_ce=0.4731 train_value_q_mse=0.0229 train_moves_left=0.0009 val_loss=3.3448 val_policy=2.3685 val_value_ce=0.9522 val_value_q_mse=0.0905 val_moves_left=0.0094 lr=1.24e-04
step 79000/100000 train_loss=2.7259 train_policy=2.4716 train_value_ce=0.2518 train_value_q_mse=0.0098 train_moves_left=0.0006 val_loss=3.3437 val_policy=2.3706 val_value_ce=0.9487 val_value_q_mse=0.0916 val_moves_left=0.0097 lr=1.14e-04
step 80000/100000 train_loss=3.4567 train_policy=2.4167 train_value_ce=1.0118 train_value_q_mse=0.1093 train_moves_left=0.0057 val_loss=3.3318 val_policy=2.3674 val_value_ce=0.9410 val_value_q_mse=0.0880 val_moves_left=0.0088 lr=1.05e-04
step 81000/100000 train_loss=3.1040 train_policy=2.4708 train_value_ce=0.6226 train_value_q_mse=0.0404 train_moves_left=0.0030 val_loss=3.3663 val_policy=2.3680 val_value_ce=0.9735 val_value_q_mse=0.0931 val_moves_left=0.0102 lr=9.56e-05
step 82000/100000 train_loss=3.2429 train_policy=2.6702 train_value_ce=0.5668 train_value_q_mse=0.0226 train_moves_left=0.0022 val_loss=3.3859 val_policy=2.3732 val_value_ce=0.9866 val_value_q_mse=0.0976 val_moves_left=0.0111 lr=8.71e-05
step 83000/100000 train_loss=3.1992 train_policy=2.3294 train_value_ce=0.8446 train_value_q_mse=0.0992 train_moves_left=0.0025 val_loss=3.3548 val_policy=2.3682 val_value_ce=0.9616 val_value_q_mse=0.0938 val_moves_left=0.0097 lr=7.89e-05
step 84000/100000 train_loss=3.0905 train_policy=2.5750 train_value_ce=0.5110 train_value_q_mse=0.0170 train_moves_left=0.0017 val_loss=3.3310 val_policy=2.3625 val_value_ce=0.9449 val_value_q_mse=0.0894 val_moves_left=0.0082 lr=7.12e-05
step 85000/100000 train_loss=2.7703 train_policy=2.3453 train_value_ce=0.4128 train_value_q_mse=0.0476 train_moves_left=0.0026 val_loss=3.3363 val_policy=2.3626 val_value_ce=0.9496 val_value_q_mse=0.0913 val_moves_left=0.0085 lr=6.40e-05
step 86000/100000 train_loss=3.1213 train_policy=2.3224 train_value_ce=0.7827 train_value_q_mse=0.0566 train_moves_left=0.0141 val_loss=3.3409 val_policy=2.3654 val_value_ce=0.9517 val_value_q_mse=0.0902 val_moves_left=0.0089 lr=5.71e-05
step 87000/100000 train_loss=3.3721 train_policy=2.4535 train_value_ce=0.8937 train_value_q_mse=0.0991 train_moves_left=0.0013 val_loss=3.3920 val_policy=2.3679 val_value_ce=0.9978 val_value_q_mse=0.0986 val_moves_left=0.0109 lr=5.07e-05
step 88000/100000 train_loss=3.4382 train_policy=2.3151 train_value_ce=1.0996 train_value_q_mse=0.0926 train_moves_left=0.0028 val_loss=3.3303 val_policy=2.3619 val_value_ce=0.9448 val_value_q_mse=0.0897 val_moves_left=0.0083 lr=4.48e-05
step 89000/100000 train_loss=3.2692 train_policy=2.4474 train_value_ce=0.7973 train_value_q_mse=0.0967 train_moves_left=0.0018 val_loss=3.3262 val_policy=2.3630 val_value_ce=0.9399 val_value_q_mse=0.0884 val_moves_left=0.0083 lr=3.93e-05
step 90000/100000 train_loss=2.9459 train_policy=2.3966 train_value_ce=0.5425 train_value_q_mse=0.0254 train_moves_left=0.0034 val_loss=3.3497 val_policy=2.3646 val_value_ce=0.9610 val_value_q_mse=0.0912 val_moves_left=0.0092 lr=3.42e-05
step 91000/100000 train_loss=3.2614 train_policy=2.4165 train_value_ce=0.8375 train_value_q_mse=0.0286 train_moves_left=0.0017 val_loss=3.3075 val_policy=2.3598 val_value_ce=0.9251 val_value_q_mse=0.0859 val_moves_left=0.0078 lr=2.97e-05
step 92000/100000 train_loss=3.1901 train_policy=2.5371 train_value_ce=0.6453 train_value_q_mse=0.0241 train_moves_left=0.0118 val_loss=3.4085 val_policy=2.3692 val_value_ce=1.0119 val_value_q_mse=0.1026 val_moves_left=0.0116 lr=2.56e-05
step 93000/100000 train_loss=3.4891 train_policy=2.3627 train_value_ce=1.1164 train_value_q_mse=0.0379 train_moves_left=0.0030 val_loss=3.3879 val_policy=2.3656 val_value_ce=0.9969 val_value_q_mse=0.0953 val_moves_left=0.0106 lr=2.19e-05
step 94000/100000 train_loss=3.0123 train_policy=2.5921 train_value_ce=0.4061 train_value_q_mse=0.0552 train_moves_left=0.0023 val_loss=3.3742 val_policy=2.3637 val_value_ce=0.9857 val_value_q_mse=0.0934 val_moves_left=0.0094 lr=1.88e-05
step 95000/100000 train_loss=3.1619 train_policy=2.4764 train_value_ce=0.6784 train_value_q_mse=0.0251 train_moves_left=0.0053 val_loss=3.3767 val_policy=2.3661 val_value_ce=0.9854 val_value_q_mse=0.0944 val_moves_left=0.0102 lr=1.61e-05
step 96000/100000 train_loss=3.4248 train_policy=2.4141 train_value_ce=0.9993 train_value_q_mse=0.0450 train_moves_left=0.0007 val_loss=3.4037 val_policy=2.3680 val_value_ce=1.0091 val_value_q_mse=0.0994 val_moves_left=0.0114 lr=1.39e-05
step 97000/100000 train_loss=3.1666 train_policy=2.4418 train_value_ce=0.7151 train_value_q_mse=0.0378 train_moves_left=0.0017 val_loss=3.3646 val_policy=2.3648 val_value_ce=0.9748 val_value_q_mse=0.0941 val_moves_left=0.0097 lr=1.22e-05
step 98000/100000 train_loss=3.1029 train_policy=2.3197 train_value_ce=0.7667 train_value_q_mse=0.0629 train_moves_left=0.0049 val_loss=3.3474 val_policy=2.3641 val_value_ce=0.9592 val_value_q_mse=0.0902 val_moves_left=0.0101 lr=1.10e-05
step 99000/100000 train_loss=3.0844 train_policy=2.2606 train_value_ce=0.7994 train_value_q_mse=0.0876 train_moves_left=0.0164 val_loss=3.3767 val_policy=2.3658 val_value_ce=0.9857 val_value_q_mse=0.0944 val_moves_left=0.0103 lr=1.02e-05
step 100000/100000 train_loss=3.2842 train_policy=2.4608 train_value_ce=0.7882 train_value_q_mse=0.1360 train_moves_left=0.0082 val_loss=3.3033 val_policy=2.3601 val_value_ce=0.9203 val_value_q_mse=0.0866 val_moves_left=0.0081 lr=1.00e-05












(.venv) PS C:\projects\77xiangqi_engine> python.exe nn\scripts\train\train_px0.py `
>>   --px0-version 710 `
>>   --init-from data\checkpoints\baseline_px0_katago_v1.best.pt `
>>   --out data\checkpoints\baseline_px0_katago_v1_qmix050.pt `
>>   --width 128 `
>>   --blocks 8 `
>>   --batch-size 256 `
>>   --steps 100000 `
>>   --eval-every 1000 `
>>   --val-batches 64 `
>>   --num-workers 4 `
>>   --device cuda `
>>   --q-ratio 0.5
torch 2.11.0+cu128 | cuda.is_available=True | device=cuda
init from data\checkpoints\baseline_px0_katago_v1.best.pt | start new phase with q_ratio=0.500
px0: train_files=360092 val_files=40010 batch_size=256 steps=100000 q_ratio=0.500
px0_kaggle: version=710 root=C:\work\px0data val_ratio=0.100
step 1/100000 train_loss=3.1961 train_policy=2.6877 train_value_ce=0.5016 train_value_q_mse=0.0258 train_moves_left=0.0022 val_loss=3.3713 val_policy=2.4178 val_value_ce=0.9285 val_value_q_mse=0.0944 val_moves_left=0.0092 lr=1.00e-03
step 1000/100000 train_loss=4.0644 train_policy=2.7162 train_value_ce=1.3316 train_value_q_mse=0.0651 train_moves_left=0.0024 val_loss=3.5427 val_policy=2.4692 val_value_ce=1.0402 val_value_q_mse=0.1258 val_moves_left=0.0116 lr=1.00e-03
step 2000/100000 train_loss=3.1528 train_policy=2.7695 train_value_ce=0.3816 train_value_q_mse=0.0065 train_moves_left=0.0004 val_loss=3.5198 val_policy=2.4613 val_value_ce=1.0251 val_value_q_mse=0.1266 val_moves_left=0.0115 lr=9.99e-04
step 3000/100000 train_loss=3.1505 train_policy=2.5056 train_value_ce=0.6309 train_value_q_mse=0.0542 train_moves_left=0.0031 val_loss=3.4826 val_policy=2.4573 val_value_ce=0.9939 val_value_q_mse=0.1206 val_moves_left=0.0080 lr=9.98e-04
step 4000/100000 train_loss=3.3429 train_policy=2.6889 train_value_ce=0.6367 train_value_q_mse=0.0659 train_moves_left=0.0055 val_loss=3.4680 val_policy=2.4668 val_value_ce=0.9727 val_value_q_mse=0.1097 val_moves_left=0.0078 lr=9.96e-04
step 5000/100000 train_loss=3.4245 train_policy=2.4869 train_value_ce=0.9070 train_value_q_mse=0.1192 train_moves_left=0.0058 val_loss=3.4756 val_policy=2.4645 val_value_ce=0.9811 val_value_q_mse=0.1139 val_moves_left=0.0099 lr=9.94e-04
step 6000/100000 train_loss=2.7540 train_policy=2.4749 train_value_ce=0.2750 train_value_q_mse=0.0152 train_moves_left=0.0024 val_loss=3.5285 val_policy=2.4723 val_value_ce=1.0255 val_value_q_mse=0.1163 val_moves_left=0.0112 lr=9.91e-04
step 7000/100000 train_loss=3.2318 train_policy=2.5405 train_value_ce=0.6775 train_value_q_mse=0.0536 train_moves_left=0.0029 val_loss=3.5729 val_policy=2.4764 val_value_ce=1.0588 val_value_q_mse=0.1416 val_moves_left=0.0149 lr=9.88e-04
step 8000/100000 train_loss=3.2447 train_policy=2.4144 train_value_ce=0.8224 train_value_q_mse=0.0310 train_moves_left=0.0015 val_loss=3.4943 val_policy=2.4646 val_value_ce=0.9977 val_value_q_mse=0.1205 val_moves_left=0.0120 lr=9.84e-04
step 9000/100000 train_loss=3.6079 train_policy=2.6305 train_value_ce=0.9500 train_value_q_mse=0.1080 train_moves_left=0.0028 val_loss=3.4634 val_policy=2.4557 val_value_ce=0.9783 val_value_q_mse=0.1123 val_moves_left=0.0088 lr=9.80e-04
step 10000/100000 train_loss=3.3246 train_policy=2.4834 train_value_ce=0.8184 train_value_q_mse=0.0898 train_moves_left=0.0024 val_loss=3.4795 val_policy=2.4605 val_value_ce=0.9890 val_value_q_mse=0.1129 val_moves_left=0.0121 lr=9.76e-04
step 11000/100000 train_loss=3.2438 train_policy=2.5011 train_value_ce=0.7295 train_value_q_mse=0.0486 train_moves_left=0.0063 val_loss=3.5077 val_policy=2.4568 val_value_ce=1.0169 val_value_q_mse=0.1303 val_moves_left=0.0096 lr=9.71e-04
step 12000/100000 train_loss=3.5621 train_policy=2.5265 train_value_ce=1.0020 train_value_q_mse=0.1325 train_moves_left=0.0035 val_loss=3.4759 val_policy=2.4509 val_value_ce=0.9946 val_value_q_mse=0.1154 val_moves_left=0.0105 lr=9.65e-04
step 13000/100000 train_loss=3.1501 train_policy=2.3374 train_value_ce=0.7819 train_value_q_mse=0.1207 train_moves_left=0.0044 val_loss=3.4636 val_policy=2.4512 val_value_ce=0.9809 val_value_q_mse=0.1202 val_moves_left=0.0099 lr=9.59e-04
step 14000/100000 train_loss=3.4302 train_policy=2.4338 train_value_ce=0.9823 train_value_q_mse=0.0549 train_moves_left=0.0020 val_loss=3.4703 val_policy=2.4510 val_value_ce=0.9882 val_value_q_mse=0.1184 val_moves_left=0.0097 lr=9.53e-04
step 15000/100000 train_loss=3.1881 train_policy=2.5511 train_value_ce=0.6096 train_value_q_mse=0.1088 train_moves_left=0.0008 val_loss=3.4378 val_policy=2.4500 val_value_ce=0.9582 val_value_q_mse=0.1128 val_moves_left=0.0091 lr=9.46e-04
step 16000/100000 train_loss=3.8844 train_policy=2.5531 train_value_ce=1.3254 train_value_q_mse=0.0225 train_moves_left=0.0024 val_loss=3.5050 val_policy=2.4426 val_value_ce=1.0292 val_value_q_mse=0.1256 val_moves_left=0.0116 lr=9.39e-04
step 17000/100000 train_loss=3.1119 train_policy=2.6405 train_value_ce=0.4693 train_value_q_mse=0.0079 train_moves_left=0.0006 val_loss=3.5217 val_policy=2.4506 val_value_ce=1.0383 val_value_q_mse=0.1237 val_moves_left=0.0122 lr=9.31e-04
step 18000/100000 train_loss=3.5706 train_policy=2.5591 train_value_ce=0.9662 train_value_q_mse=0.1714 train_moves_left=0.0163 val_loss=3.4775 val_policy=2.4406 val_value_ce=1.0069 val_value_q_mse=0.1134 val_moves_left=0.0109 lr=9.23e-04
step 19000/100000 train_loss=3.3424 train_policy=2.4137 train_value_ce=0.9067 train_value_q_mse=0.0802 train_moves_left=0.0129 val_loss=3.4787 val_policy=2.4407 val_value_ce=1.0079 val_value_q_mse=0.1151 val_moves_left=0.0092 lr=9.14e-04
step 20000/100000 train_loss=3.0329 train_policy=2.6041 train_value_ce=0.4136 train_value_q_mse=0.0576 train_moves_left=0.0049 val_loss=3.4541 val_policy=2.4394 val_value_ce=0.9850 val_value_q_mse=0.1137 val_moves_left=0.0092 lr=9.05e-04
step 21000/100000 train_loss=2.9927 train_policy=2.6670 train_value_ce=0.3241 train_value_q_mse=0.0062 train_moves_left=0.0004 val_loss=3.5293 val_policy=2.4561 val_value_ce=1.0390 val_value_q_mse=0.1270 val_moves_left=0.0156 lr=8.96e-04
step 22000/100000 train_loss=3.3039 train_policy=2.7764 train_value_ce=0.5090 train_value_q_mse=0.0647 train_moves_left=0.0153 val_loss=3.4857 val_policy=2.4379 val_value_ce=1.0173 val_value_q_mse=0.1171 val_moves_left=0.0080 lr=8.86e-04
step 23000/100000 train_loss=3.0344 train_policy=2.7117 train_value_ce=0.3127 train_value_q_mse=0.0346 train_moves_left=0.0093 val_loss=3.4429 val_policy=2.4473 val_value_ce=0.9673 val_value_q_mse=0.1061 val_moves_left=0.0121 lr=8.76e-04
step 24000/100000 train_loss=2.9511 train_policy=2.3521 train_value_ce=0.5882 train_value_q_mse=0.0337 train_moves_left=0.0157 val_loss=3.5054 val_policy=2.4397 val_value_ce=1.0353 val_value_q_mse=0.1154 val_moves_left=0.0107 lr=8.66e-04
step 25000/100000 train_loss=3.0066 train_policy=2.7836 train_value_ce=0.2222 train_value_q_mse=0.0031 train_moves_left=0.0005 val_loss=3.5553 val_policy=2.4400 val_value_ce=1.0790 val_value_q_mse=0.1377 val_moves_left=0.0125 lr=8.55e-04
step 26000/100000 train_loss=3.0663 train_policy=2.5565 train_value_ce=0.5003 train_value_q_mse=0.0374 train_moves_left=0.0010 val_loss=3.4852 val_policy=2.4280 val_value_ce=1.0272 val_value_q_mse=0.1136 val_moves_left=0.0103 lr=8.44e-04
step 27000/100000 train_loss=2.8539 train_policy=2.4511 train_value_ce=0.3980 train_value_q_mse=0.0185 train_moves_left=0.0005 val_loss=3.4714 val_policy=2.4336 val_value_ce=1.0072 val_value_q_mse=0.1164 val_moves_left=0.0096 lr=8.32e-04
step 28000/100000 train_loss=3.1705 train_policy=2.4531 train_value_ce=0.6890 train_value_q_mse=0.1123 train_moves_left=0.0020 val_loss=3.4591 val_policy=2.4292 val_value_ce=1.0004 val_value_q_mse=0.1125 val_moves_left=0.0090 lr=8.21e-04
step 29000/100000 train_loss=2.9458 train_policy=2.2429 train_value_ce=0.6741 train_value_q_mse=0.1126 train_moves_left=0.0048 val_loss=3.4334 val_policy=2.4285 val_value_ce=0.9769 val_value_q_mse=0.1057 val_moves_left=0.0100 lr=8.08e-04
step 30000/100000 train_loss=3.1140 train_policy=2.7192 train_value_ce=0.3920 train_value_q_mse=0.0096 train_moves_left=0.0026 val_loss=3.4152 val_policy=2.4199 val_value_ce=0.9668 val_value_q_mse=0.1091 val_moves_left=0.0085 lr=7.96e-04
step 31000/100000 train_loss=3.4462 train_policy=2.3785 train_value_ce=1.0447 train_value_q_mse=0.0914 train_moves_left=0.0013 val_loss=3.4615 val_policy=2.4295 val_value_ce=1.0027 val_value_q_mse=0.1099 val_moves_left=0.0119 lr=7.83e-04
step 32000/100000 train_loss=2.9691 train_policy=2.5085 train_value_ce=0.4483 train_value_q_mse=0.0476 train_moves_left=0.0030 val_loss=3.4314 val_policy=2.4236 val_value_ce=0.9773 val_value_q_mse=0.1163 val_moves_left=0.0096 lr=7.70e-04
step 33000/100000 train_loss=3.3003 train_policy=2.5735 train_value_ce=0.6958 train_value_q_mse=0.1214 train_moves_left=0.0039 val_loss=3.4048 val_policy=2.4214 val_value_ce=0.9570 val_value_q_mse=0.1002 val_moves_left=0.0095 lr=7.57e-04
step 34000/100000 train_loss=3.0263 train_policy=2.5636 train_value_ce=0.4596 train_value_q_mse=0.0109 train_moves_left=0.0020 val_loss=3.4714 val_policy=2.4218 val_value_ce=1.0190 val_value_q_mse=0.1162 val_moves_left=0.0100 lr=7.43e-04
step 35000/100000 train_loss=3.0843 train_policy=2.7330 train_value_ce=0.3352 train_value_q_mse=0.0611 train_moves_left=0.0061 val_loss=3.4311 val_policy=2.4186 val_value_ce=0.9847 val_value_q_mse=0.1058 val_moves_left=0.0091 lr=7.30e-04
step 36000/100000 train_loss=3.5034 train_policy=2.8225 train_value_ce=0.6564 train_value_q_mse=0.0964 train_moves_left=0.0020 val_loss=3.4720 val_policy=2.4186 val_value_ce=1.0238 val_value_q_mse=0.1118 val_moves_left=0.0110 lr=7.16e-04
step 37000/100000 train_loss=3.0187 train_policy=2.7090 train_value_ce=0.3072 train_value_q_mse=0.0091 train_moves_left=0.0008 val_loss=3.4247 val_policy=2.4133 val_value_ce=0.9839 val_value_q_mse=0.1046 val_moves_left=0.0089 lr=7.02e-04
step 38000/100000 train_loss=3.1470 train_policy=2.6177 train_value_ce=0.5242 train_value_q_mse=0.0186 train_moves_left=0.0030 val_loss=3.3891 val_policy=2.4116 val_value_ce=0.9511 val_value_q_mse=0.1005 val_moves_left=0.0080 lr=6.87e-04
step 39000/100000 train_loss=3.0289 train_policy=2.7898 train_value_ce=0.2376 train_value_q_mse=0.0047 train_moves_left=0.0023 val_loss=3.5426 val_policy=2.4125 val_value_ce=1.0976 val_value_q_mse=0.1222 val_moves_left=0.0127 lr=6.73e-04
step 40000/100000 train_loss=3.3072 train_policy=2.4632 train_value_ce=0.8225 train_value_q_mse=0.0839 train_moves_left=0.0037 val_loss=3.3805 val_policy=2.4068 val_value_ce=0.9471 val_value_q_mse=0.1019 val_moves_left=0.0075 lr=6.58e-04
step 41000/100000 train_loss=3.5667 train_policy=2.4267 train_value_ce=1.1072 train_value_q_mse=0.1302 train_moves_left=0.0017 val_loss=3.3733 val_policy=2.4089 val_value_ce=0.9392 val_value_q_mse=0.0962 val_moves_left=0.0081 lr=6.43e-04
step 42000/100000 train_loss=2.8772 train_policy=2.6943 train_value_ce=0.1813 train_value_q_mse=0.0047 train_moves_left=0.0028 val_loss=3.4664 val_policy=2.4115 val_value_ce=1.0266 val_value_q_mse=0.1068 val_moves_left=0.0103 lr=6.28e-04
step 43000/100000 train_loss=2.8830 train_policy=2.3292 train_value_ce=0.5448 train_value_q_mse=0.0212 train_moves_left=0.0238 val_loss=3.3913 val_policy=2.4055 val_value_ce=0.9596 val_value_q_mse=0.0997 val_moves_left=0.0089 lr=6.13e-04
step 44000/100000 train_loss=3.4883 train_policy=2.5590 train_value_ce=0.9251 train_value_q_mse=0.0165 train_moves_left=0.0004 val_loss=3.4040 val_policy=2.4086 val_value_ce=0.9681 val_value_q_mse=0.1030 val_moves_left=0.0100 lr=5.98e-04
step 45000/100000 train_loss=2.7802 train_policy=2.7198 train_value_ce=0.0602 train_value_q_mse=0.0002 train_moves_left=0.0010 val_loss=3.4268 val_policy=2.4052 val_value_ce=0.9935 val_value_q_mse=0.1050 val_moves_left=0.0122 lr=5.82e-04
step 46000/100000 train_loss=2.5946 train_policy=2.1166 train_value_ce=0.4711 train_value_q_mse=0.0240 train_moves_left=0.0055 val_loss=3.4598 val_policy=2.4076 val_value_ce=1.0215 val_value_q_mse=0.1149 val_moves_left=0.0135 lr=5.67e-04
step 47000/100000 train_loss=3.0626 train_policy=2.6820 train_value_ce=0.3772 train_value_q_mse=0.0122 train_moves_left=0.0026 val_loss=3.4228 val_policy=2.4000 val_value_ce=0.9960 val_value_q_mse=0.1008 val_moves_left=0.0101 lr=5.52e-04
step 48000/100000 train_loss=3.1253 train_policy=2.4234 train_value_ce=0.6933 train_value_q_mse=0.0331 train_moves_left=0.0024 val_loss=3.4472 val_policy=2.4017 val_value_ce=1.0159 val_value_q_mse=0.1122 val_moves_left=0.0099 lr=5.36e-04
step 49000/100000 train_loss=3.1228 train_policy=2.5301 train_value_ce=0.5813 train_value_q_mse=0.0449 train_moves_left=0.0010 val_loss=3.4162 val_policy=2.4029 val_value_ce=0.9854 val_value_q_mse=0.1059 val_moves_left=0.0099 lr=5.21e-04
step 50000/100000 train_loss=3.0827 train_policy=2.3651 train_value_ce=0.7020 train_value_q_mse=0.0596 train_moves_left=0.0047 val_loss=3.3814 val_policy=2.3998 val_value_ce=0.9548 val_value_q_mse=0.1018 val_moves_left=0.0090 lr=5.05e-04
step 51000/100000 train_loss=2.9574 train_policy=2.5643 train_value_ce=0.3846 train_value_q_mse=0.0283 train_moves_left=0.0100 val_loss=3.3959 val_policy=2.3951 val_value_ce=0.9751 val_value_q_mse=0.0976 val_moves_left=0.0082 lr=4.89e-04
step 52000/100000 train_loss=3.7303 train_policy=2.2959 train_value_ce=1.3750 train_value_q_mse=0.2358 train_moves_left=0.0031 val_loss=3.3811 val_policy=2.3915 val_value_ce=0.9637 val_value_q_mse=0.0973 val_moves_left=0.0101 lr=4.74e-04
step 53000/100000 train_loss=2.9051 train_policy=2.5110 train_value_ce=0.3891 train_value_q_mse=0.0176 train_moves_left=0.0037 val_loss=3.3869 val_policy=2.3939 val_value_ce=0.9669 val_value_q_mse=0.0986 val_moves_left=0.0098 lr=4.58e-04
step 54000/100000 train_loss=2.5037 train_policy=2.4280 train_value_ce=0.0745 train_value_q_mse=0.0014 train_moves_left=0.0051 val_loss=3.4204 val_policy=2.4013 val_value_ce=0.9906 val_value_q_mse=0.1077 val_moves_left=0.0103 lr=4.43e-04
step 55000/100000 train_loss=3.1066 train_policy=2.6596 train_value_ce=0.4410 train_value_q_mse=0.0221 train_moves_left=0.0031 val_loss=3.4504 val_policy=2.3986 val_value_ce=1.0235 val_value_q_mse=0.1058 val_moves_left=0.0119 lr=4.28e-04
step 56000/100000 train_loss=3.2412 train_policy=2.6572 train_value_ce=0.5634 train_value_q_mse=0.0818 train_moves_left=0.0010 val_loss=3.4171 val_policy=2.3898 val_value_ce=1.0006 val_value_q_mse=0.1019 val_moves_left=0.0080 lr=4.12e-04
step 57000/100000 train_loss=2.7023 train_policy=2.1907 train_value_ce=0.4909 train_value_q_mse=0.0795 train_moves_left=0.0058 val_loss=3.3571 val_policy=2.3893 val_value_ce=0.9421 val_value_q_mse=0.0976 val_moves_left=0.0084 lr=3.97e-04
step 58000/100000 train_loss=3.1993 train_policy=2.4907 train_value_ce=0.6976 train_value_q_mse=0.0424 train_moves_left=0.0026 val_loss=3.3700 val_policy=2.3859 val_value_ce=0.9588 val_value_q_mse=0.0958 val_moves_left=0.0093 lr=3.82e-04
step 59000/100000 train_loss=3.1339 train_policy=2.3949 train_value_ce=0.7188 train_value_q_mse=0.0800 train_moves_left=0.0018 val_loss=3.4122 val_policy=2.3920 val_value_ce=0.9935 val_value_q_mse=0.0998 val_moves_left=0.0118 lr=3.67e-04
step 60000/100000 train_loss=3.2059 train_policy=2.4522 train_value_ce=0.7419 train_value_q_mse=0.0452 train_moves_left=0.0027 val_loss=3.4210 val_policy=2.3901 val_value_ce=1.0036 val_value_q_mse=0.1028 val_moves_left=0.0109 lr=3.52e-04
step 61000/100000 train_loss=3.1647 train_policy=2.7017 train_value_ce=0.4589 train_value_q_mse=0.0149 train_moves_left=0.0026 val_loss=3.3930 val_policy=2.3990 val_value_ce=0.9679 val_value_q_mse=0.0983 val_moves_left=0.0106 lr=3.37e-04
step 62000/100000 train_loss=2.8725 train_policy=2.6607 train_value_ce=0.2076 train_value_q_mse=0.0158 train_moves_left=0.0009 val_loss=3.4133 val_policy=2.3892 val_value_ce=0.9979 val_value_q_mse=0.0997 val_moves_left=0.0089 lr=3.23e-04
step 63000/100000 train_loss=3.0752 train_policy=2.5488 train_value_ce=0.5201 train_value_q_mse=0.0231 train_moves_left=0.0037 val_loss=3.3728 val_policy=2.3819 val_value_ce=0.9664 val_value_q_mse=0.0921 val_moves_left=0.0098 lr=3.08e-04
step 64000/100000 train_loss=2.9233 train_policy=2.5408 train_value_ce=0.3755 train_value_q_mse=0.0278 train_moves_left=0.0001 val_loss=3.3736 val_policy=2.3816 val_value_ce=0.9668 val_value_q_mse=0.0947 val_moves_left=0.0102 lr=2.94e-04
step 65000/100000 train_loss=2.5856 train_policy=2.2422 train_value_ce=0.3390 train_value_q_mse=0.0161 train_moves_left=0.0023 val_loss=3.4174 val_policy=2.3836 val_value_ce=1.0070 val_value_q_mse=0.1011 val_moves_left=0.0100 lr=2.80e-04
step 66000/100000 train_loss=3.1414 train_policy=2.4962 train_value_ce=0.6320 train_value_q_mse=0.0512 train_moves_left=0.0032 val_loss=3.3436 val_policy=2.3836 val_value_ce=0.9346 val_value_q_mse=0.0957 val_moves_left=0.0093 lr=2.67e-04
step 67000/100000 train_loss=3.1762 train_policy=2.1730 train_value_ce=0.9884 train_value_q_mse=0.0550 train_moves_left=0.0066 val_loss=3.3482 val_policy=2.3752 val_value_ce=0.9487 val_value_q_mse=0.0915 val_moves_left=0.0094 lr=2.53e-04
step 68000/100000 train_loss=3.2619 train_policy=2.4238 train_value_ce=0.8210 train_value_q_mse=0.0663 train_moves_left=0.0038 val_loss=3.3620 val_policy=2.3781 val_value_ce=0.9591 val_value_q_mse=0.0929 val_moves_left=0.0100 lr=2.40e-04
step 69000/100000 train_loss=2.9740 train_policy=2.5714 train_value_ce=0.3981 train_value_q_mse=0.0173 train_moves_left=0.0017 val_loss=3.3819 val_policy=2.3786 val_value_ce=0.9770 val_value_q_mse=0.0992 val_moves_left=0.0101 lr=2.27e-04
step 70000/100000 train_loss=3.2676 train_policy=2.3377 train_value_ce=0.9127 train_value_q_mse=0.0642 train_moves_left=0.0078 val_loss=3.3412 val_policy=2.3750 val_value_ce=0.9425 val_value_q_mse=0.0901 val_moves_left=0.0078 lr=2.14e-04
step 71000/100000 train_loss=2.8788 train_policy=2.4037 train_value_ce=0.4661 train_value_q_mse=0.0349 train_moves_left=0.0019 val_loss=3.4017 val_policy=2.3807 val_value_ce=0.9946 val_value_q_mse=0.0982 val_moves_left=0.0123 lr=2.02e-04
step 72000/100000 train_loss=3.0835 train_policy=2.5176 train_value_ce=0.5458 train_value_q_mse=0.0688 train_moves_left=0.0189 val_loss=3.3502 val_policy=2.3765 val_value_ce=0.9494 val_value_q_mse=0.0912 val_moves_left=0.0103 lr=1.89e-04
step 73000/100000 train_loss=4.0874 train_policy=2.6481 train_value_ce=1.4257 train_value_q_mse=0.0487 train_moves_left=0.0094 val_loss=3.3475 val_policy=2.3728 val_value_ce=0.9511 val_value_q_mse=0.0887 val_moves_left=0.0093 lr=1.78e-04
step 74000/100000 train_loss=2.8685 train_policy=2.3508 train_value_ce=0.4997 train_value_q_mse=0.0712 train_moves_left=0.0010 val_loss=3.3713 val_policy=2.3736 val_value_ce=0.9719 val_value_q_mse=0.0966 val_moves_left=0.0110 lr=1.66e-04
step 75000/100000 train_loss=3.1229 train_policy=2.3800 train_value_ce=0.7321 train_value_q_mse=0.0400 train_moves_left=0.0050 val_loss=3.3671 val_policy=2.3716 val_value_ce=0.9703 val_value_q_mse=0.0955 val_moves_left=0.0091 lr=1.55e-04
step 76000/100000 train_loss=3.0928 train_policy=2.4221 train_value_ce=0.6571 train_value_q_mse=0.0516 train_moves_left=0.0045 val_loss=3.3480 val_policy=2.3681 val_value_ce=0.9551 val_value_q_mse=0.0935 val_moves_left=0.0092 lr=1.44e-04
step 77000/100000 train_loss=3.1848 train_policy=2.2879 train_value_ce=0.8755 train_value_q_mse=0.0823 train_moves_left=0.0057 val_loss=3.3792 val_policy=2.3718 val_value_ce=0.9819 val_value_q_mse=0.0952 val_moves_left=0.0112 lr=1.34e-04
step 78000/100000 train_loss=3.1349 train_policy=2.6560 train_value_ce=0.4731 train_value_q_mse=0.0229 train_moves_left=0.0009 val_loss=3.3448 val_policy=2.3685 val_value_ce=0.9522 val_value_q_mse=0.0905 val_moves_left=0.0094 lr=1.24e-04
step 79000/100000 train_loss=2.7259 train_policy=2.4716 train_value_ce=0.2518 train_value_q_mse=0.0098 train_moves_left=0.0006 val_loss=3.3437 val_policy=2.3706 val_value_ce=0.9487 val_value_q_mse=0.0916 val_moves_left=0.0097 lr=1.14e-04
step 80000/100000 train_loss=3.4567 train_policy=2.4167 train_value_ce=1.0118 train_value_q_mse=0.1093 train_moves_left=0.0057 val_loss=3.3318 val_policy=2.3674 val_value_ce=0.9410 val_value_q_mse=0.0880 val_moves_left=0.0088 lr=1.05e-04
step 81000/100000 train_loss=3.1040 train_policy=2.4708 train_value_ce=0.6226 train_value_q_mse=0.0404 train_moves_left=0.0030 val_loss=3.3663 val_policy=2.3680 val_value_ce=0.9735 val_value_q_mse=0.0931 val_moves_left=0.0102 lr=9.56e-05
step 82000/100000 train_loss=3.2429 train_policy=2.6702 train_value_ce=0.5668 train_value_q_mse=0.0226 train_moves_left=0.0022 val_loss=3.3859 val_policy=2.3732 val_value_ce=0.9866 val_value_q_mse=0.0976 val_moves_left=0.0111 lr=8.71e-05
step 83000/100000 train_loss=3.1992 train_policy=2.3294 train_value_ce=0.8446 train_value_q_mse=0.0992 train_moves_left=0.0025 val_loss=3.3548 val_policy=2.3682 val_value_ce=0.9616 val_value_q_mse=0.0938 val_moves_left=0.0097 lr=7.89e-05
step 84000/100000 train_loss=3.0905 train_policy=2.5750 train_value_ce=0.5110 train_value_q_mse=0.0170 train_moves_left=0.0017 val_loss=3.3310 val_policy=2.3625 val_value_ce=0.9449 val_value_q_mse=0.0894 val_moves_left=0.0082 lr=7.12e-05
step 85000/100000 train_loss=2.7703 train_policy=2.3453 train_value_ce=0.4128 train_value_q_mse=0.0476 train_moves_left=0.0026 val_loss=3.3363 val_policy=2.3626 val_value_ce=0.9496 val_value_q_mse=0.0913 val_moves_left=0.0085 lr=6.40e-05
step 86000/100000 train_loss=3.1213 train_policy=2.3224 train_value_ce=0.7827 train_value_q_mse=0.0566 train_moves_left=0.0141 val_loss=3.3409 val_policy=2.3654 val_value_ce=0.9517 val_value_q_mse=0.0902 val_moves_left=0.0089 lr=5.71e-05
step 87000/100000 train_loss=3.3721 train_policy=2.4535 train_value_ce=0.8937 train_value_q_mse=0.0991 train_moves_left=0.0013 val_loss=3.3920 val_policy=2.3679 val_value_ce=0.9978 val_value_q_mse=0.0986 val_moves_left=0.0109 lr=5.07e-05
step 88000/100000 train_loss=3.4382 train_policy=2.3151 train_value_ce=1.0996 train_value_q_mse=0.0926 train_moves_left=0.0028 val_loss=3.3303 val_policy=2.3619 val_value_ce=0.9448 val_value_q_mse=0.0897 val_moves_left=0.0083 lr=4.48e-05
step 89000/100000 train_loss=3.2692 train_policy=2.4474 train_value_ce=0.7973 train_value_q_mse=0.0967 train_moves_left=0.0018 val_loss=3.3262 val_policy=2.3630 val_value_ce=0.9399 val_value_q_mse=0.0884 val_moves_left=0.0083 lr=3.93e-05
step 90000/100000 train_loss=2.9459 train_policy=2.3966 train_value_ce=0.5425 train_value_q_mse=0.0254 train_moves_left=0.0034 val_loss=3.3497 val_policy=2.3646 val_value_ce=0.9610 val_value_q_mse=0.0912 val_moves_left=0.0092 lr=3.42e-05
step 91000/100000 train_loss=3.2614 train_policy=2.4165 train_value_ce=0.8375 train_value_q_mse=0.0286 train_moves_left=0.0017 val_loss=3.3075 val_policy=2.3598 val_value_ce=0.9251 val_value_q_mse=0.0859 val_moves_left=0.0078 lr=2.97e-05
step 92000/100000 train_loss=3.1901 train_policy=2.5371 train_value_ce=0.6453 train_value_q_mse=0.0241 train_moves_left=0.0118 val_loss=3.4085 val_policy=2.3692 val_value_ce=1.0119 val_value_q_mse=0.1026 val_moves_left=0.0116 lr=2.56e-05
step 93000/100000 train_loss=3.4891 train_policy=2.3627 train_value_ce=1.1164 train_value_q_mse=0.0379 train_moves_left=0.0030 val_loss=3.3879 val_policy=2.3656 val_value_ce=0.9969 val_value_q_mse=0.0953 val_moves_left=0.0106 lr=2.19e-05
step 94000/100000 train_loss=3.0123 train_policy=2.5921 train_value_ce=0.4061 train_value_q_mse=0.0552 train_moves_left=0.0023 val_loss=3.3742 val_policy=2.3637 val_value_ce=0.9857 val_value_q_mse=0.0934 val_moves_left=0.0094 lr=1.88e-05
step 95000/100000 train_loss=3.1619 train_policy=2.4764 train_value_ce=0.6784 train_value_q_mse=0.0251 train_moves_left=0.0053 val_loss=3.3767 val_policy=2.3661 val_value_ce=0.9854 val_value_q_mse=0.0944 val_moves_left=0.0102 lr=1.61e-05
step 96000/100000 train_loss=3.4248 train_policy=2.4141 train_value_ce=0.9993 train_value_q_mse=0.0450 train_moves_left=0.0007 val_loss=3.4037 val_policy=2.3680 val_value_ce=1.0091 val_value_q_mse=0.0994 val_moves_left=0.0114 lr=1.39e-05
step 97000/100000 train_loss=3.1666 train_policy=2.4418 train_value_ce=0.7151 train_value_q_mse=0.0378 train_moves_left=0.0017 val_loss=3.3646 val_policy=2.3648 val_value_ce=0.9748 val_value_q_mse=0.0941 val_moves_left=0.0097 lr=1.22e-05
step 98000/100000 train_loss=3.1029 train_policy=2.3197 train_value_ce=0.7667 train_value_q_mse=0.0629 train_moves_left=0.0049 val_loss=3.3474 val_policy=2.3641 val_value_ce=0.9592 val_value_q_mse=0.0902 val_moves_left=0.0101 lr=1.10e-05
step 99000/100000 train_loss=3.0844 train_policy=2.2606 train_value_ce=0.7994 train_value_q_mse=0.0876 train_moves_left=0.0164 val_loss=3.3767 val_policy=2.3658 val_value_ce=0.9857 val_value_q_mse=0.0944 val_moves_left=0.0103 lr=1.02e-05
step 100000/100000 train_loss=3.2842 train_policy=2.4608 train_value_ce=0.7882 train_value_q_mse=0.1360 train_moves_left=0.0082 val_loss=3.3033 val_policy=2.3601 val_value_ce=0.9203 val_value_q_mse=0.0866 val_moves_left=0.0081 lr=1.00e-05
(.venv) PS C:\projects\77xiangqi_engine>
(.venv) PS C:\projects\77xiangqi_engine>
(.venv) PS C:\projects\77xiangqi_engine> python.exe nn\scripts\train\train_px0.py `
>>   --px0-version 710 `
>>   --init-from data\checkpoints\baseline_px0_katago_v1.best.pt `
>>   --out data\checkpoints\baseline_px0_katago_v1_qmix050.pt `
>>   --width 128 `
>>   --blocks 8 `
>>   --batch-size 256 `
>>   --steps 100000 `
>>   --eval-every 1000 `
>>   --val-batches 64 `
>>   --num-workers 4 `
>>   --device cuda `
>>   --q-ratio 0.75
torch 2.11.0+cu128 | cuda.is_available=True | device=cuda
--out 已存在时不要再传 --init-from；默认会直接从 --out 续训
(.venv) PS C:\projects\77xiangqi_engine> python.exe nn\scripts\train\train_px0.py `
>>   --px0-version 710 `
>>   --init-from data\checkpoints\baseline_px0_katago_v1.best.pt `
>>   --out data\checkpoints\baseline_px0_katago_v1_qmix075.pt `
>>   --width 128 `
>>   --blocks 8 `
>>   --batch-size 256 `
>>   --steps 100000 `
>>   --eval-every 1000 `
>>   --val-batches 64 `
>>   --num-workers 4 `
>>   --device cuda `
>>   --q-ratio 0.75
torch 2.11.0+cu128 | cuda.is_available=True | device=cuda
init from data\checkpoints\baseline_px0_katago_v1.best.pt | start new phase with q_ratio=0.750
px0: train_files=360092 val_files=40010 batch_size=256 steps=100000 q_ratio=0.750
px0_kaggle: version=710 root=C:\work\px0data val_ratio=0.100
step 1/100000 train_loss=3.2076 train_policy=2.6877 train_value_ce=0.5132 train_value_q_mse=0.0258 train_moves_left=0.0022 val_loss=3.3551 val_policy=2.4178 val_value_ce=0.9123 val_value_q_mse=0.0944 val_moves_left=0.0092 lr=1.00e-03
step 1000/100000 train_loss=3.7145 train_policy=2.7152 train_value_ce=0.9828 train_value_q_mse=0.0645 train_moves_left=0.0024 val_loss=3.4759 val_policy=2.4672 val_value_ce=0.9761 val_value_q_mse=0.1233 val_moves_left=0.0113 lr=1.00e-03
step 2000/100000 train_loss=3.2213 train_policy=2.7632 train_value_ce=0.4563 train_value_q_mse=0.0068 train_moves_left=0.0004 val_loss=3.4712 val_policy=2.4624 val_value_ce=0.9770 val_value_q_mse=0.1205 val_moves_left=0.0116 lr=9.99e-04
step 3000/100000 train_loss=3.1873 train_policy=2.5249 train_value_ce=0.6497 train_value_q_mse=0.0491 train_moves_left=0.0030 val_loss=3.4547 val_policy=2.4617 val_value_ce=0.9625 val_value_q_mse=0.1171 val_moves_left=0.0082 lr=9.98e-04
step 4000/100000 train_loss=3.3622 train_policy=2.6947 train_value_ce=0.6501 train_value_q_mse=0.0665 train_moves_left=0.0055 val_loss=3.4318 val_policy=2.4662 val_value_ce=0.9370 val_value_q_mse=0.1098 val_moves_left=0.0076 lr=9.96e-04
step 5000/100000 train_loss=3.4366 train_policy=2.4758 train_value_ce=0.9295 train_value_q_mse=0.1219 train_moves_left=0.0054 val_loss=3.4323 val_policy=2.4619 val_value_ce=0.9417 val_value_q_mse=0.1091 val_moves_left=0.0098 lr=9.94e-04
step 6000/100000 train_loss=2.7993 train_policy=2.4736 train_value_ce=0.3204 train_value_q_mse=0.0199 train_moves_left=0.0026 val_loss=3.4655 val_policy=2.4708 val_value_ce=0.9650 val_value_q_mse=0.1111 val_moves_left=0.0124 lr=9.91e-04
step 7000/100000 train_loss=3.2272 train_policy=2.5451 train_value_ce=0.6693 train_value_q_mse=0.0495 train_moves_left=0.0029 val_loss=3.5092 val_policy=2.4714 val_value_ce=1.0027 val_value_q_mse=0.1310 val_moves_left=0.0156 lr=9.88e-04
step 8000/100000 train_loss=3.1791 train_policy=2.4049 train_value_ce=0.7665 train_value_q_mse=0.0298 train_moves_left=0.0014 val_loss=3.4466 val_policy=2.4585 val_value_ce=0.9570 val_value_q_mse=0.1169 val_moves_left=0.0120 lr=9.84e-04
step 9000/100000 train_loss=3.4011 train_policy=2.6179 train_value_ce=0.7634 train_value_q_mse=0.0779 train_moves_left=0.0025 val_loss=3.4293 val_policy=2.4581 val_value_ce=0.9424 val_value_q_mse=0.1095 val_moves_left=0.0095 lr=9.80e-04
step 10000/100000 train_loss=3.2922 train_policy=2.4773 train_value_ce=0.7947 train_value_q_mse=0.0796 train_moves_left=0.0022 val_loss=3.4446 val_policy=2.4596 val_value_ce=0.9551 val_value_q_mse=0.1128 val_moves_left=0.0117 lr=9.76e-04
step 11000/100000 train_loss=3.2305 train_policy=2.5094 train_value_ce=0.7082 train_value_q_mse=0.0479 train_moves_left=0.0060 val_loss=3.4444 val_policy=2.4539 val_value_ce=0.9599 val_value_q_mse=0.1169 val_moves_left=0.0096 lr=9.71e-04
step 12000/100000 train_loss=3.5586 train_policy=2.5476 train_value_ce=0.9814 train_value_q_mse=0.1162 train_moves_left=0.0034 val_loss=3.4444 val_policy=2.4558 val_value_ce=0.9589 val_value_q_mse=0.1133 val_moves_left=0.0096 lr=9.65e-04
step 13000/100000 train_loss=3.1515 train_policy=2.3124 train_value_ce=0.8097 train_value_q_mse=0.1161 train_moves_left=0.0029 val_loss=3.4117 val_policy=2.4449 val_value_ce=0.9386 val_value_q_mse=0.1071 val_moves_left=0.0092 lr=9.59e-04
step 14000/100000 train_loss=3.2545 train_policy=2.4299 train_value_ce=0.8107 train_value_q_mse=0.0544 train_moves_left=0.0018 val_loss=3.4222 val_policy=2.4490 val_value_ce=0.9444 val_value_q_mse=0.1096 val_moves_left=0.0093 lr=9.53e-04
step 15000/100000 train_loss=3.2292 train_policy=2.5531 train_value_ce=0.6502 train_value_q_mse=0.1031 train_moves_left=0.0008 val_loss=3.4088 val_policy=2.4466 val_value_ce=0.9333 val_value_q_mse=0.1102 val_moves_left=0.0089 lr=9.46e-04
step 16000/100000 train_loss=3.5628 train_policy=2.5366 train_value_ce=1.0204 train_value_q_mse=0.0219 train_moves_left=0.0022 val_loss=3.4420 val_policy=2.4422 val_value_ce=0.9686 val_value_q_mse=0.1178 val_moves_left=0.0117 lr=9.39e-04
step 17000/100000 train_loss=3.1594 train_policy=2.6383 train_value_ce=0.5192 train_value_q_mse=0.0073 train_moves_left=0.0005 val_loss=3.4531 val_policy=2.4480 val_value_ce=0.9740 val_value_q_mse=0.1173 val_moves_left=0.0117 lr=9.31e-04
step 18000/100000 train_loss=3.5591 train_policy=2.5478 train_value_ce=0.9658 train_value_q_mse=0.1721 train_moves_left=0.0167 val_loss=3.4280 val_policy=2.4377 val_value_ce=0.9604 val_value_q_mse=0.1134 val_moves_left=0.0108 lr=9.23e-04
step 19000/100000 train_loss=3.3299 train_policy=2.3989 train_value_ce=0.9085 train_value_q_mse=0.0826 train_moves_left=0.0126 val_loss=3.4179 val_policy=2.4363 val_value_ce=0.9530 val_value_q_mse=0.1091 val_moves_left=0.0089 lr=9.14e-04
step 20000/100000 train_loss=3.0440 train_policy=2.6004 train_value_ce=0.4284 train_value_q_mse=0.0577 train_moves_left=0.0053 val_loss=3.4112 val_policy=2.4330 val_value_ce=0.9488 val_value_q_mse=0.1125 val_moves_left=0.0090 lr=9.05e-04
step 21000/100000 train_loss=3.0540 train_policy=2.6642 train_value_ce=0.3881 train_value_q_mse=0.0066 train_moves_left=0.0003 val_loss=3.4682 val_policy=2.4508 val_value_ce=0.9842 val_value_q_mse=0.1230 val_moves_left=0.0159 lr=8.96e-04
step 22000/100000 train_loss=3.3017 train_policy=2.7743 train_value_ce=0.5083 train_value_q_mse=0.0664 train_moves_left=0.0158 val_loss=3.4372 val_policy=2.4356 val_value_ce=0.9722 val_value_q_mse=0.1128 val_moves_left=0.0080 lr=8.86e-04
step 23000/100000 train_loss=3.0943 train_policy=2.7070 train_value_ce=0.3741 train_value_q_mse=0.0466 train_moves_left=0.0103 val_loss=3.4091 val_policy=2.4439 val_value_ce=0.9373 val_value_q_mse=0.1041 val_moves_left=0.0122 lr=8.76e-04
step 24000/100000 train_loss=2.9354 train_policy=2.3520 train_value_ce=0.5736 train_value_q_mse=0.0296 train_moves_left=0.0154 val_loss=3.4282 val_policy=2.4339 val_value_ce=0.9656 val_value_q_mse=0.1085 val_moves_left=0.0105 lr=8.66e-04
step 25000/100000 train_loss=3.0782 train_policy=2.7922 train_value_ce=0.2847 train_value_q_mse=0.0046 train_moves_left=0.0008 val_loss=3.4829 val_policy=2.4342 val_value_ce=1.0137 val_value_q_mse=0.1330 val_moves_left=0.0114 lr=8.55e-04
step 26000/100000 train_loss=3.1794 train_policy=2.5437 train_value_ce=0.6259 train_value_q_mse=0.0387 train_moves_left=0.0011 val_loss=3.4154 val_policy=2.4239 val_value_ce=0.9629 val_value_q_mse=0.1084 val_moves_left=0.0105 lr=8.44e-04
step 27000/100000 train_loss=2.9484 train_policy=2.4415 train_value_ce=0.5023 train_value_q_mse=0.0182 train_moves_left=0.0006 val_loss=3.4128 val_policy=2.4314 val_value_ce=0.9527 val_value_q_mse=0.1087 val_moves_left=0.0096 lr=8.32e-04
step 28000/100000 train_loss=3.1816 train_policy=2.4470 train_value_ce=0.7056 train_value_q_mse=0.1146 train_moves_left=0.0021 val_loss=3.4056 val_policy=2.4236 val_value_ce=0.9537 val_value_q_mse=0.1077 val_moves_left=0.0090 lr=8.21e-04
step 29000/100000 train_loss=2.9741 train_policy=2.2316 train_value_ce=0.7153 train_value_q_mse=0.1055 train_moves_left=0.0054 val_loss=3.3820 val_policy=2.4225 val_value_ce=0.9328 val_value_q_mse=0.1007 val_moves_left=0.0101 lr=8.08e-04
step 30000/100000 train_loss=3.1517 train_policy=2.7208 train_value_ce=0.4281 train_value_q_mse=0.0095 train_moves_left=0.0028 val_loss=3.3850 val_policy=2.4175 val_value_ce=0.9388 val_value_q_mse=0.1098 val_moves_left=0.0084 lr=7.96e-04
step 31000/100000 train_loss=3.3398 train_policy=2.3731 train_value_ce=0.9454 train_value_q_mse=0.0844 train_moves_left=0.0015 val_loss=3.3954 val_policy=2.4247 val_value_ce=0.9426 val_value_q_mse=0.1051 val_moves_left=0.0119 lr=7.83e-04
step 32000/100000 train_loss=2.9909 train_policy=2.5072 train_value_ce=0.4714 train_value_q_mse=0.0473 train_moves_left=0.0030 val_loss=3.3960 val_policy=2.4187 val_value_ce=0.9476 val_value_q_mse=0.1131 val_moves_left=0.0097 lr=7.70e-04
step 33000/100000 train_loss=3.3323 train_policy=2.5758 train_value_ce=0.7262 train_value_q_mse=0.1186 train_moves_left=0.0040 val_loss=3.3706 val_policy=2.4170 val_value_ce=0.9275 val_value_q_mse=0.0985 val_moves_left=0.0099 lr=7.57e-04
step 34000/100000 train_loss=3.0576 train_policy=2.5661 train_value_ce=0.4890 train_value_q_mse=0.0092 train_moves_left=0.0018 val_loss=3.4083 val_policy=2.4166 val_value_ce=0.9628 val_value_q_mse=0.1091 val_moves_left=0.0102 lr=7.43e-04
step 35000/100000 train_loss=3.0648 train_policy=2.7347 train_value_ce=0.3172 train_value_q_mse=0.0475 train_moves_left=0.0070 val_loss=3.3739 val_policy=2.4138 val_value_ce=0.9338 val_value_q_mse=0.0999 val_moves_left=0.0091 lr=7.30e-04
step 36000/100000 train_loss=3.4854 train_policy=2.8231 train_value_ce=0.6403 train_value_q_mse=0.0868 train_moves_left=0.0019 val_loss=3.4042 val_policy=2.4131 val_value_ce=0.9629 val_value_q_mse=0.1062 val_moves_left=0.0110 lr=7.16e-04
step 37000/100000 train_loss=3.1046 train_policy=2.7103 train_value_ce=0.3919 train_value_q_mse=0.0091 train_moves_left=0.0008 val_loss=3.3671 val_policy=2.4069 val_value_ce=0.9341 val_value_q_mse=0.0996 val_moves_left=0.0086 lr=7.02e-04
step 38000/100000 train_loss=3.1760 train_policy=2.6117 train_value_ce=0.5603 train_value_q_mse=0.0147 train_moves_left=0.0027 val_loss=3.3465 val_policy=2.4061 val_value_ce=0.9154 val_value_q_mse=0.0948 val_moves_left=0.0082 lr=6.87e-04
step 39000/100000 train_loss=3.0935 train_policy=2.7821 train_value_ce=0.3095 train_value_q_mse=0.0061 train_moves_left=0.0023 val_loss=3.4598 val_policy=2.4085 val_value_ce=1.0200 val_value_q_mse=0.1175 val_moves_left=0.0128 lr=6.73e-04
step 40000/100000 train_loss=3.3610 train_policy=2.4633 train_value_ce=0.8705 train_value_q_mse=0.1067 train_moves_left=0.0037 val_loss=3.3405 val_policy=2.4019 val_value_ce=0.9131 val_value_q_mse=0.0979 val_moves_left=0.0074 lr=6.58e-04
step 41000/100000 train_loss=3.4320 train_policy=2.4258 train_value_ce=0.9730 train_value_q_mse=0.1319 train_moves_left=0.0018 val_loss=3.3505 val_policy=2.4069 val_value_ce=0.9188 val_value_q_mse=0.0937 val_moves_left=0.0083 lr=6.43e-04
step 42000/100000 train_loss=2.9322 train_policy=2.6912 train_value_ce=0.2393 train_value_q_mse=0.0052 train_moves_left=0.0027 val_loss=3.3916 val_policy=2.4087 val_value_ce=0.9558 val_value_q_mse=0.1018 val_moves_left=0.0105 lr=6.28e-04
step 43000/100000 train_loss=2.9537 train_policy=2.3486 train_value_ce=0.5957 train_value_q_mse=0.0235 train_moves_left=0.0240 val_loss=3.3451 val_policy=2.4014 val_value_ce=0.9185 val_value_q_mse=0.0951 val_moves_left=0.0094 lr=6.13e-04
step 44000/100000 train_loss=3.3294 train_policy=2.5471 train_value_ce=0.7783 train_value_q_mse=0.0159 train_moves_left=0.0005 val_loss=3.3590 val_policy=2.4035 val_value_ce=0.9295 val_value_q_mse=0.0977 val_moves_left=0.0107 lr=5.98e-04
step 45000/100000 train_loss=2.7880 train_policy=2.7094 train_value_ce=0.0783 train_value_q_mse=0.0002 train_moves_left=0.0014 val_loss=3.3754 val_policy=2.3995 val_value_ce=0.9483 val_value_q_mse=0.1026 val_moves_left=0.0128 lr=5.82e-04
step 46000/100000 train_loss=2.6159 train_policy=2.1216 train_value_ce=0.4875 train_value_q_mse=0.0238 train_moves_left=0.0055 val_loss=3.4113 val_policy=2.4047 val_value_ce=0.9768 val_value_q_mse=0.1108 val_moves_left=0.0141 lr=5.67e-04
step 47000/100000 train_loss=3.1205 train_policy=2.6593 train_value_ce=0.4579 train_value_q_mse=0.0119 train_moves_left=0.0024 val_loss=3.3727 val_policy=2.3948 val_value_ce=0.9516 val_value_q_mse=0.0992 val_moves_left=0.0106 lr=5.52e-04
step 48000/100000 train_loss=3.1546 train_policy=2.4177 train_value_ce=0.7291 train_value_q_mse=0.0299 train_moves_left=0.0026 val_loss=3.3811 val_policy=2.3957 val_value_ce=0.9576 val_value_q_mse=0.1055 val_moves_left=0.0100 lr=5.36e-04
step 49000/100000 train_loss=3.1453 train_policy=2.5220 train_value_ce=0.6108 train_value_q_mse=0.0493 train_moves_left=0.0010 val_loss=3.3616 val_policy=2.3977 val_value_ce=0.9372 val_value_q_mse=0.1008 val_moves_left=0.0101 lr=5.21e-04
step 50000/100000 train_loss=3.0900 train_policy=2.3610 train_value_ce=0.7137 train_value_q_mse=0.0586 train_moves_left=0.0046 val_loss=3.3448 val_policy=2.3936 val_value_ce=0.9251 val_value_q_mse=0.0990 val_moves_left=0.0088 lr=5.05e-04
step 51000/100000 train_loss=3.0115 train_policy=2.5642 train_value_ce=0.4393 train_value_q_mse=0.0257 train_moves_left=0.0104 val_loss=3.3458 val_policy=2.3912 val_value_ce=0.9294 val_value_q_mse=0.0956 val_moves_left=0.0083 lr=4.89e-04
step 52000/100000 train_loss=3.4727 train_policy=2.2813 train_value_ce=1.1379 train_value_q_mse=0.2123 train_moves_left=0.0030 val_loss=3.3350 val_policy=2.3888 val_value_ce=0.9211 val_value_q_mse=0.0936 val_moves_left=0.0111 lr=4.74e-04
step 53000/100000 train_loss=2.9902 train_policy=2.5149 train_value_ce=0.4706 train_value_q_mse=0.0167 train_moves_left=0.0034 val_loss=3.3488 val_policy=2.3916 val_value_ce=0.9316 val_value_q_mse=0.0961 val_moves_left=0.0104 lr=4.58e-04
step 54000/100000 train_loss=2.5278 train_policy=2.4304 train_value_ce=0.0961 train_value_q_mse=0.0021 train_moves_left=0.0055 val_loss=3.3724 val_policy=2.4013 val_value_ce=0.9437 val_value_q_mse=0.1034 val_moves_left=0.0103 lr=4.43e-04
step 55000/100000 train_loss=3.1331 train_policy=2.6572 train_value_ce=0.4696 train_value_q_mse=0.0233 train_moves_left=0.0032 val_loss=3.3745 val_policy=2.3946 val_value_ce=0.9529 val_value_q_mse=0.1009 val_moves_left=0.0123 lr=4.28e-04
step 56000/100000 train_loss=3.2422 train_policy=2.6513 train_value_ce=0.5704 train_value_q_mse=0.0817 train_moves_left=0.0013 val_loss=3.3439 val_policy=2.3847 val_value_ce=0.9343 val_value_q_mse=0.0951 val_moves_left=0.0079 lr=4.12e-04
step 57000/100000 train_loss=2.6490 train_policy=2.1939 train_value_ce=0.4369 train_value_q_mse=0.0697 train_moves_left=0.0055 val_loss=3.3256 val_policy=2.3834 val_value_ce=0.9166 val_value_q_mse=0.0968 val_moves_left=0.0085 lr=3.97e-04
step 58000/100000 train_loss=3.2243 train_policy=2.4825 train_value_ce=0.7301 train_value_q_mse=0.0452 train_moves_left=0.0027 val_loss=3.3245 val_policy=2.3817 val_value_ce=0.9184 val_value_q_mse=0.0920 val_moves_left=0.0092 lr=3.82e-04
step 59000/100000 train_loss=3.1354 train_policy=2.3774 train_value_ce=0.7392 train_value_q_mse=0.0741 train_moves_left=0.0018 val_loss=3.3498 val_policy=2.3949 val_value_ce=0.9298 val_value_q_mse=0.0931 val_moves_left=0.0120 lr=3.67e-04
step 60000/100000 train_loss=3.1968 train_policy=2.4503 train_value_ce=0.7364 train_value_q_mse=0.0389 train_moves_left=0.0026 val_loss=3.3556 val_policy=2.3962 val_value_ce=0.9338 val_value_q_mse=0.0956 val_moves_left=0.0114 lr=3.52e-04
step 61000/100000 train_loss=3.1394 train_policy=2.6923 train_value_ce=0.4428 train_value_q_mse=0.0156 train_moves_left=0.0029 val_loss=3.3450 val_policy=2.3944 val_value_ce=0.9252 val_value_q_mse=0.0946 val_moves_left=0.0113 lr=3.37e-04
step 62000/100000 train_loss=2.9405 train_policy=2.6662 train_value_ce=0.2699 train_value_q_mse=0.0173 train_moves_left=0.0008 val_loss=3.3459 val_policy=2.3865 val_value_ce=0.9347 val_value_q_mse=0.0936 val_moves_left=0.0089 lr=3.23e-04
step 63000/100000 train_loss=3.1131 train_policy=2.5483 train_value_ce=0.5591 train_value_q_mse=0.0209 train_moves_left=0.0037 val_loss=3.3189 val_policy=2.3770 val_value_ce=0.9182 val_value_q_mse=0.0893 val_moves_left=0.0096 lr=3.08e-04
step 64000/100000 train_loss=3.0103 train_policy=2.5424 train_value_ce=0.4613 train_value_q_mse=0.0263 train_moves_left=0.0001 val_loss=3.3222 val_policy=2.3791 val_value_ce=0.9190 val_value_q_mse=0.0903 val_moves_left=0.0102 lr=2.94e-04
step 65000/100000 train_loss=2.5681 train_policy=2.2422 train_value_ce=0.3212 train_value_q_mse=0.0174 train_moves_left=0.0022 val_loss=3.3546 val_policy=2.3793 val_value_ce=0.9494 val_value_q_mse=0.0978 val_moves_left=0.0099 lr=2.80e-04
step 66000/100000 train_loss=3.1921 train_policy=2.4939 train_value_ce=0.6854 train_value_q_mse=0.0490 train_moves_left=0.0034 val_loss=3.3185 val_policy=2.3824 val_value_ce=0.9117 val_value_q_mse=0.0912 val_moves_left=0.0099 lr=2.67e-04
step 67000/100000 train_loss=3.0356 train_policy=2.1722 train_value_ce=0.8481 train_value_q_mse=0.0571 train_moves_left=0.0070 val_loss=3.3042 val_policy=2.3718 val_value_ce=0.9090 val_value_q_mse=0.0877 val_moves_left=0.0094 lr=2.53e-04
step 68000/100000 train_loss=3.2569 train_policy=2.4190 train_value_ce=0.8231 train_value_q_mse=0.0572 train_moves_left=0.0037 val_loss=3.3186 val_policy=2.3745 val_value_ce=0.9202 val_value_q_mse=0.0895 val_moves_left=0.0100 lr=2.40e-04
step 69000/100000 train_loss=3.0642 train_policy=2.5686 train_value_ce=0.4913 train_value_q_mse=0.0163 train_moves_left=0.0017 val_loss=3.3257 val_policy=2.3729 val_value_ce=0.9281 val_value_q_mse=0.0930 val_moves_left=0.0100 lr=2.27e-04
step 70000/100000 train_loss=3.2053 train_policy=2.3291 train_value_ce=0.8583 train_value_q_mse=0.0665 train_moves_left=0.0084 val_loss=3.2979 val_policy=2.3687 val_value_ce=0.9064 val_value_q_mse=0.0865 val_moves_left=0.0078 lr=2.14e-04
step 71000/100000 train_loss=2.8935 train_policy=2.4011 train_value_ce=0.4834 train_value_q_mse=0.0350 train_moves_left=0.0019 val_loss=3.3458 val_policy=2.3814 val_value_ce=0.9387 val_value_q_mse=0.0939 val_moves_left=0.0145 lr=2.02e-04
step 72000/100000 train_loss=3.1292 train_policy=2.5202 train_value_ce=0.5887 train_value_q_mse=0.0701 train_moves_left=0.0186 val_loss=3.3130 val_policy=2.3730 val_value_ce=0.9161 val_value_q_mse=0.0893 val_moves_left=0.0105 lr=1.89e-04
step 73000/100000 train_loss=3.7165 train_policy=2.6398 train_value_ce=1.0639 train_value_q_mse=0.0456 train_moves_left=0.0095 val_loss=3.3037 val_policy=2.3668 val_value_ce=0.9140 val_value_q_mse=0.0862 val_moves_left=0.0092 lr=1.78e-04
step 74000/100000 train_loss=2.8532 train_policy=2.3407 train_value_ce=0.4951 train_value_q_mse=0.0691 train_moves_left=0.0010 val_loss=3.3161 val_policy=2.3675 val_value_ce=0.9239 val_value_q_mse=0.0924 val_moves_left=0.0109 lr=1.66e-04
step 75000/100000 train_loss=3.1375 train_policy=2.3738 train_value_ce=0.7529 train_value_q_mse=0.0400 train_moves_left=0.0049 val_loss=3.3142 val_policy=2.3654 val_value_ce=0.9246 val_value_q_mse=0.0918 val_moves_left=0.0089 lr=1.55e-04
step 76000/100000 train_loss=3.1320 train_policy=2.4239 train_value_ce=0.6946 train_value_q_mse=0.0515 train_moves_left=0.0042 val_loss=3.2985 val_policy=2.3618 val_value_ce=0.9130 val_value_q_mse=0.0894 val_moves_left=0.0090 lr=1.44e-04
step 77000/100000 train_loss=3.1726 train_policy=2.2816 train_value_ce=0.8712 train_value_q_mse=0.0758 train_moves_left=0.0053 val_loss=3.3210 val_policy=2.3661 val_value_ce=0.9306 val_value_q_mse=0.0905 val_moves_left=0.0111 lr=1.34e-04
step 78000/100000 train_loss=3.1729 train_policy=2.6466 train_value_ce=0.5208 train_value_q_mse=0.0215 train_moves_left=0.0009 val_loss=3.3009 val_policy=2.3631 val_value_ce=0.9146 val_value_q_mse=0.0875 val_moves_left=0.0094 lr=1.24e-04
step 79000/100000 train_loss=2.7291 train_policy=2.4615 train_value_ce=0.2655 train_value_q_mse=0.0079 train_moves_left=0.0007 val_loss=3.3064 val_policy=2.3652 val_value_ce=0.9171 val_value_q_mse=0.0901 val_moves_left=0.0098 lr=1.14e-04
step 80000/100000 train_loss=3.3561 train_policy=2.4024 train_value_ce=0.9261 train_value_q_mse=0.1071 train_moves_left=0.0056 val_loss=3.2903 val_policy=2.3620 val_value_ce=0.9057 val_value_q_mse=0.0852 val_moves_left=0.0086 lr=1.05e-04
step 81000/100000 train_loss=3.1212 train_policy=2.4665 train_value_ce=0.6443 train_value_q_mse=0.0400 train_moves_left=0.0029 val_loss=3.3112 val_policy=2.3626 val_value_ce=0.9248 val_value_q_mse=0.0893 val_moves_left=0.0100 lr=9.56e-05
