# Experiment all tricks without center loss with re-ranking : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# with re-ranking
python3 tools/test.py --config_file='configs/softmax_triplet_cam_with_center_initial.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" TEST.RE_RANKING "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home/haoluo/log/gu/reid_baseline_review/Opensource_test/market1501/Experiment-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_model_120.pth')"