from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir

ZERO_MOULD_CONFIG_PATH = Path(__file__).parent / "config" / "mould.py"
ZERO_MOULD_CATS_PATH = Path(__file__).parent / "config" / "mould.yml"

@DATASETS.register_module
class ZeroMouldDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size, crop_size, split, 
            img_suffix='.jpg', seg_map_suffix='.tiff',
            config_path = ZERO_MOULD_CONFIG_PATH
        )
        self.names, self.colors = utils.dataset_cat_description(ZERO_MOULD_CATS_PATH)
        self.n_cls = 4
        self.ignore_label = 3
        self.reduce_zero_label = False