_base_ = ["./vitb_ade20k_linear_config.py"]

model = dict(
    backbone=dict(out_indices=[0, 1, 2, 3]),
    decode_head=dict(in_channels=[1024], channels=1024),
)

work_dir = "./work_dirs/segmentation_eval/swin_base_ade20k_linear"
