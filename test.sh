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

declare -a Models=("baseline_rep" "k1_m0.1" "k1_m0.2" "k1_m0.3" "k2_m0.1" )

for MODEL in ${Models[@]}
do
    python3 tools/test.py --config_file='configs/softmax_triplet_cam_with_center_initial.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('log/train/${MODEL}/resnet50_model_120.pth')" OUTPUT_DIR "('log/test/${MODEL}')"
done