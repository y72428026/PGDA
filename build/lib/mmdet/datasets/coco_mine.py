from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class CocoDataset_mine(CocoDataset):
    CLASSES=(
        'triangle offset' ,
        'leftover material',
        'left and right side material',
        'open on both sides',
        'big hole opening',
        'white border')
    def __init__(self, classes, **kwargs) -> None:
        super(CocoDataset, self).__init__(**kwargs)
        self.CLASSES = classes