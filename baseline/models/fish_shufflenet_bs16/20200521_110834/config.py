BATCH_SIZE = 16   # 8 for resnet50   2 for resnet152   5 for densenet121 2 for dense161
PROPOSAL_NUM = 6
CAT_NUM = 4
CLASS_NUM = 1000
INPUT_SIZE = (448, 448)  # (w, h)  res(448, 448)  (299, 299)
LR = 0.001
WD = 1e-4
SAVE_FREQ = 1
resume = ''
test_model = 'model.ckpt'
save_dir = './models/fish_shufflenet_bs16/'
