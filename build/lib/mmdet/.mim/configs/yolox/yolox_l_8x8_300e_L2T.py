_base_ = './yolox_l_8x8_300e_coco.py'

# model settings
model = dict(
    bbox_head=dict(in_channels=256, feat_channels=256, num_classes=6)
    )

dataset_type = 'COCODataset'
classes = (
    'triangle offset' ,
    'leftover material',
    'left and right side material',
    'open on both sides',
    'big hole opening',
    'white border')

data = dict(
    train=dict(
        dataset=dict(
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
            classes=classes,
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json')),
    val=dict(
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/HP_transparent_all.json'),
    test=dict(
        img_prefix='/media/data3/yebh/dataset/BIS/HP/HP_transparent/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/HP/HP_transparent/HP_transparent_all.json'))

