_base_ = [
    '../_base_/models/cyclegan_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_runtime.py'
]
model = dict(id_loss=dict(type='L1Loss', loss_weight=0, reduction='mean'))
dataroot = './data/kaggle_mri2ct/images'
data = dict(
    samples_per_gpu=8,
    val_samples_per_gpu=16,
    train=dict(dataroot=dataroot),
    val=dict(dataroot=dataroot),
    test=dict(dataroot=dataroot))

optimizer = dict(
    generators=dict(type='Adam', lr=0.001, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.001, betas=(0.5, 0.999)))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-2)
checkpoint_config = dict(interval=50000, save_optimizer=False, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['fake_b'],
        interval=50000)
]

runner = None
use_ddp_wrapper = True
total_iters = 270000
workflow = [('train', 1)]
metrics = dict(
    FID=dict(type='FID', num_images=140, image_shape=(3, 256, 256)),
    IS=dict(type='IS', num_images=140, image_shape=(3, 256, 256)))
