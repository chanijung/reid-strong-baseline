# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center lossOUTPUT

K=5
M=0.1
python3 tools/train.py -k $K -m $M --config_file='configs/softmax_triplet_cam_with_center_initial.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" MODEL.PRETRAIN_PATH "('/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')" OUTPUT_DIR "('log/train/k${K}_m${M}')"

