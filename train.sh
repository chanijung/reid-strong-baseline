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
# python3 tools/train.py -k  -m 0.4 --config_file='configs/softmax_triplet_cam_with_center_initial.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" MODEL.PRETRAIN_PATH "('/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')" OUTPUT_DIR "('log/test')"
# python3 tools/train.py --config_file='configs/softmax_triplet_cam_with_center_initial.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" MODEL.PRETRAIN_PATH "('/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')" OUTPUT_DIR "('log/test')"

for K in 1 2
do
    for M in 0.1 0.4
    do
        # if [[ $K -eq 2 ]] && [[ $M -eq 0.1 ]]
        # then
        #     continue
        # fi
        python3 tools/train.py -k $K -m $M --config_file='configs/softmax_triplet_cam_with_center_initial.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" MODEL.PRETRAIN_PATH "('/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')" OUTPUT_DIR "('log/train/k${K}_m${M}')"
    done
done
