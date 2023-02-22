import os
from typing import List, Union, Sequence

import oneflow as flow
import oneflow.nn as nn

from .register import DATA


class OFRecordDataLoader(nn.Module):
    def __init__(
        self,
        ofrecord_root: str = None,
        n_ofrecord_parts: int = 1,
        ofrecord_image_key: str = None,
        ofrecord_label_key: str = None,
        is_training: bool = True,
        color_space: str = "RGB",
        image_size: int = 224,
        mean: Sequence[float] = None,
        std: Sequence[float] = None,
        dataset_size: int = 9469,
        batch_size: int = 1,
        total_batch_size: int = 1,
        part_name_suffix_length: int = 6,
        channel_last: bool = False,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()

        self.train_record_reader = nn.OfrecordReader(
            ofrecord_root,
            batch_size=batch_size,
            data_part_num=n_ofrecord_parts,
            part_name_suffix_length=part_name_suffix_length,
            random_shuffle=is_training,
            shuffle_after_epoch=is_training,
            placement=placement,
            sbp=sbp,
        )
        self.record_label_decoder = nn.OFRecordRawDecoder(
            ofrecord_label_key,
            shape=(),
            dtype=flow.int32
        )

        self.record_image_decoder = (
            nn.OFRecordImageDecoderRandomCrop(
                ofrecord_image_key,
                color_space=color_space
            )
            if is_training else
            nn.OFRecordImageDecoder(
                ofrecord_image_key,
                color_space=color_space
            )
        )

        self.resize = (
            nn.image.Resize(target_size=[image_size, image_size])
            if ofrecord_image_key else
            nn.image.Resize(
                resize_side="shorter", keep_aspect_ratio=True, target_size=int(image_size/7*8)
            )
        )

        self.flip = (
            nn.CoinFlip(batch_size=batch_size, placement=placement, sbp=sbp)
            if is_training else None
        )

        output_layout = "NHWC" if channel_last else "NCHW"
        self.crop_mirror_norm = (
            nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                mean=mean,
                std=std,
                output_dtype=flow.float,
            )
            if is_training else
            nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                crop_h=image_size,
                crop_w=image_size,
                crop_pos_y=0.5,
                crop_pos_x=0.5,
                mean=mean,
                std=std,
                output_dtype=flow.float,
            )
        )

        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size // self.total_batch_size

    def forward(self):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        image_raw_buffer = self.record_image_decoder(train_record)
        image = self.resize(image_raw_buffer)[0]
        rng = self.flip() if self.flip != None else None
        image = self.crop_mirror_norm(image, rng)

        return image, label


@DATA.register
def image_ofrecord_loader(root, image_size, mean, std, batch_size, n_ofrecord_parts, channel_last, **kwargs):
    return (
        # train set dataloader
        OFRecordDataLoader(
            ofrecord_root=os.path.join(root, "train"),
            n_ofrecord_parts=n_ofrecord_parts,
            ofrecord_image_key="image",
            ofrecord_label_key="label",
            is_training=True,
            color_space="RGB",
            image_size=image_size,
            mean=mean,
            std=std,
            batch_size=batch_size,
            channel_last=channel_last,
            **kwargs
        ),
        # val set dataloader
        OFRecordDataLoader(
            ofrecord_root=os.path.join(root, "val"),
            n_ofrecord_parts=n_ofrecord_parts,
            ofrecord_image_key="image",
            ofrecord_label_key="label",
            is_training=False,
            color_space="RGB",
            image_size=image_size,
            mean=mean,
            std=std,
            batch_size=batch_size,
            channel_last=channel_last,
            **kwargs
        )
    )
