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
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
lr_config = None
checkpoint_config = dict(interval=20000, save_optimizer=False, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['fake_b'],
        interval=10000)
]

runner = None
use_ddp_wrapper = True
total_iters = 100000
workflow = [('train', 1)]
metrics = dict(
    FID=dict(type='FID', num_images=140, image_shape=(3, 256, 256)),
    IS=dict(type='IS', num_images=140, image_shape=(3, 256, 256)))
