import os
from typing import List, Union, Sequence

import oneflow as flow
import oneflow.nn as nn

from .register import DATA

from codebase.utils.distributed import is_dist_init


class OFRecordDataLoader(flow.nn.Module):
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
        dataset_size: int = None,
        batch_size: int = None,
        part_name_suffix_length: int = 6,
        channel_last: bool = False,
        use_gpu: bool = False,
        num_workers: int = 4,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.is_training = is_training
        self.use_gpu = use_gpu

        self.ofrecord_reader = nn.OfrecordReader(
            ofrecord_dir=ofrecord_root,
            batch_size=batch_size,
            data_part_num=n_ofrecord_parts,
            part_name_suffix_length=part_name_suffix_length,
            random_shuffle=self.is_training,
            shuffle_after_epoch=self.is_training,
            placement=placement,
            sbp=sbp,
        )

        self.label_decoder = nn.OfrecordRawDecoder(
            ofrecord_label_key, shape=tuple(), dtype=flow.int32
        )

        if channel_last:
            os.environ["ONEFLOW_ENABLE_NHWC"] = "1"

        if is_training:
            if self.use_gpu:
                self.bytesdecoder_img = nn.OFRecordBytesDecoder(ofrecord_image_key)
                self.image_decoder = nn.OFRecordImageGpuDecoderRandomCropResize(
                    target_width=image_size,
                    target_height=image_size,
                    num_workers=num_workers,
                    warmup_size=2048,
                )
            else:
                self.image_decoder = nn.OFRecordImageDecoderRandomCrop(
                    ofrecord_image_key, color_space=color_space
                )
                self.resize = nn.image.Resize(
                    target_size=[image_size, image_size]
                )
            self.flip = nn.CoinFlip(
                batch_size=self.batch_size, placement=placement, sbp=sbp
            )
            self.crop_mirror_norm = nn.CropMirrorNormalize(
                color_space=color_space,
                mean=[item * 256 for item in mean],
                std=[item * 256 for item in std],
                output_dtype=flow.float,
            )
        else:
            self.image_decoder = nn.OFRecordImageDecoder(
                ofrecord_image_key, color_space=color_space
            )
            self.resize = nn.image.Resize(
                resize_side="shorter",
                keep_aspect_ratio=True,
                target_size=int(image_size/7*8),
            )
            self.crop_mirror_norm = nn.CropMirrorNormalize(
                color_space=color_space,
                crop_h=image_size,
                crop_w=image_size,
                crop_pos_y=0.5,
                crop_pos_x=0.5,
                mean=[item * 256 for item in mean],
                std=[item * 256 for item in std],
                output_dtype=flow.float,
            )

    def __len__(self):
        return self.dataset_size // self.batch_size

    def forward(self):
        if self.is_training:
            record = self.ofrecord_reader()
            if self.use_gpu:
                encoded = self.bytesdecoder_img(record)
                image = self.image_decoder(encoded)
            else:
                image_raw_bytes = self.image_decoder(record)
                image = self.resize(image_raw_bytes)[0]
                # image = image.to("cuda")

            flip_code = self.flip()
            if self.use_gpu:
                flip_code = flip_code.to("cuda")
            image = self.crop_mirror_norm(image, flip_code)
        else:
            record = self.ofrecord_reader()
            image_raw_bytes = self.image_decoder(record)

            image = self.resize(image_raw_bytes)[0]
            image = self.crop_mirror_norm(image)

        label = self.label_decoder(record)
        return image, label


@DATA.register
def image_ofrecord_loader(
    train_root,
    val_root,
    ofrecord_image_key,
    ofrecord_label_key,
    color_space,
    image_size,
    mean,
    std,
    batch_size,
    part_name_suffix_length,
    n_ofrecord_parts,
    channel_last,
    use_gpu,
    num_workers,
    **kwargs
):
    placement = flow.placement.all("cpu") if is_dist_init() else None
    sbp = flow.sbp.split(0) if is_dist_init() else None
    return (
        # train set dataloader
        OFRecordDataLoader(
            ofrecord_root=train_root,
            n_ofrecord_parts=n_ofrecord_parts,
            ofrecord_image_key=ofrecord_image_key,
            ofrecord_label_key=ofrecord_label_key,
            is_training=True,
            color_space=color_space,
            image_size=image_size,
            mean=mean,
            std=std,
            batch_size=batch_size,
            part_name_suffix_length=part_name_suffix_length,
            channel_last=channel_last,
            use_gpu=use_gpu,
            num_workers=num_workers,
            placement=placement,
            sbp=sbp,
            **kwargs
        ),
        # val set dataloader
        OFRecordDataLoader(
            ofrecord_root=val_root,
            n_ofrecord_parts=n_ofrecord_parts,
            ofrecord_image_key=ofrecord_image_key,
            ofrecord_label_key=ofrecord_label_key,
            is_training=False,
            color_space=color_space,
            image_size=image_size,
            mean=mean,
            std=std,
            batch_size=batch_size,
            part_name_suffix_length=part_name_suffix_length,
            channel_last=channel_last,
            placement=placement,
            use_gpu=use_gpu,
            num_workers=num_workers,
            sbp=sbp,
            **kwargs
        )
    )
