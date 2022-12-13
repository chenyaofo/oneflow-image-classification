import os
import numpy as np
import oneflow as flow

import oneflow as flow
import oneflow.nn as nn
import oneflow.optim as optim
import oneflow.cuda as cuda
import oneflow.amp as amp
import oneflow.env as env

import typing


def make_data_loader(args, mode, is_global=False, synthetic=False):
    assert mode in ("train", "validation")

    if mode == "train":
        total_batch_size = args.train_global_batch_size
        batch_size = args.train_batch_size
        num_samples = args.samples_per_epoch
    else:
        total_batch_size = args.val_global_batch_size
        batch_size = args.val_batch_size
        num_samples = args.val_samples_per_epoch

    placement = None
    sbp = None

    if is_global:
        placement = flow.env.all_device_placement("cpu")
        sbp = flow.sbp.split(0)
        # NOTE(zwx): global view, only consider logical batch size
        batch_size = total_batch_size

    if synthetic:
        data_loader = SyntheticDataLoader(
            batch_size=batch_size,
            num_classes=args.num_classes,
            placement=placement,
            sbp=sbp,
            channel_last=args.channel_last,
        )
        return data_loader.to("cuda")

    ofrecord_data_loader = OFRecordDataLoader(
        root=args.ofrecord_path,
        num_shards=args.ofrecord_part_num,
        dataset_size=num_samples,
        mode=mode,
        batch_size=batch_size,
        total_batch_size=total_batch_size,
        channel_last=args.channel_last,
        placement=placement,
        sbp=sbp,
        use_gpu_decode=args.use_gpu_decode,
    )
    return ofrecord_data_loader


class OFRecordDataLoader(flow.nn.Module):
    def __init__(
        self,
        root: str = "./ofrecord",
        color_space: str = "RGB",
        mean: typing.List[float] = None,
        std: typing.List[float] = None,
        num_shards: int = 1,
        dataset_size: int = 9469,
        is_training: bool = None,
        batch_size: int = 1,
        total_batch_size: int = 1,
        channel_last=False,
        placement=None,
        sbp=None,
        use_gpu_decode: int = False,
    ):
        super().__init__()

        # assert mode in ("train", "validation")
        assert isinstance(is_training, bool)

        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.dataset_size = dataset_size
        self.is_training = is_training
        self.use_gpu_decode = use_gpu_decode

        self.ofrecord_reader = nn.OfrecordReader(
            ofrecord_dir=os.path.join(root, "train" if self.is_training else "val"),
            batch_size=batch_size,
            data_part_num=num_shards,
            part_name_suffix_length=5,
            random_shuffle=self.is_training,
            shuffle_after_epoch=self.is_training,
            placement=placement,
            sbp=sbp,
        )

        self.label_decoder = nn.OfrecordRawDecoder(
            "class/label", shape=tuple(), dtype=flow.int32
        )

        if channel_last:
            os.environ["ONEFLOW_ENABLE_NHWC"] = "1"

        image_height = 224
        image_width = 224
        resize_shorter = 256

        if is_training:
            if self.use_gpu_decode:
                self.bytesdecoder_img = nn.OFRecordBytesDecoder("encoded")
                self.image_decoder = nn.OFRecordImageGpuDecoderRandomCropResize(
                    target_width=image_width,
                    target_height=image_height,
                    num_workers=3,
                    warmup_size=2048,
                )
            else:
                self.image_decoder = nn.OFRecordImageDecoderRandomCrop(
                    "encoded", color_space=color_space
                )
                self.resize = nn.image.Resize(
                    target_size=[image_width, image_height]
                )
            self.flip = nn.CoinFlip(
                batch_size=self.batch_size, placement=placement, sbp=sbp
            )
            self.crop_mirror_norm = flow.nn.CropMirrorNormalize(
                color_space=color_space,
                mean=[item * 256 for item in mean],
                std=[item * 256 for item in std],
                output_dtype=flow.float,
            )
        else:
            self.image_decoder = nn.OFRecordImageDecoder(
                "encoded", color_space=color_space
            )
            self.resize = nn.image.Resize(
                resize_side="shorter",
                keep_aspect_ratio=True,
                target_size=resize_shorter,
            )
            self.crop_mirror_norm = nn.CropMirrorNormalize(
                color_space=color_space,
                crop_h=image_height,
                crop_w=image_width,
                crop_pos_y=0.5,
                crop_pos_x=0.5,
                mean=[item * 256 for item in mean],
                std=[item * 256 for item in std],
                output_dtype=flow.float,
            )

    def __len__(self):
        return self.dataset_size // self.total_batch_size

    def forward(self):
        if self.is_training:
            record = self.ofrecord_reader()
            if self.use_gpu_decode:
                encoded = self.bytesdecoder_img(record)
                image = self.image_decoder(encoded)
            else:
                image_raw_bytes = self.image_decoder(record)
                image = self.resize(image_raw_bytes)[0]
                image = image.to("cuda")

            flip_code = self.flip()
            flip_code = flip_code.to("cuda")
            image = self.crop_mirror_norm(image, flip_code)
        else:
            record = self.ofrecord_reader()
            image_raw_bytes = self.image_decoder(record)

            image = self.resize(image_raw_bytes)[0]
            image = self.crop_mirror_norm(image)

        label = self.label_decoder(record)
        return image, label


class SyntheticDataLoader(flow.nn.Module):
    def __init__(
        self,
        batch_size,
        image_size=224,
        num_classes=1000,
        placement=None,
        sbp=None,
        channel_last=False,
    ):
        super().__init__()

        if channel_last:
            self.image_shape = (batch_size, image_size, image_size, 3)
        else:
            self.image_shape = (batch_size, 3, image_size, image_size)
        self.label_shape = (batch_size,)
        self.num_classes = num_classes
        self.placement = placement
        self.sbp = sbp

        if self.placement is not None and self.sbp is not None:
            self.image = flow.nn.Parameter(
                flow.randint(
                    0,
                    high=256,
                    size=self.image_shape,
                    dtype=flow.float32,
                    placement=self.placement,
                    sbp=self.sbp,
                ),
                requires_grad=False,
            )
            self.label = flow.nn.Parameter(
                flow.randint(
                    0,
                    high=self.num_classes,
                    size=self.label_shape,
                    placement=self.placement,
                    sbp=self.sbp,
                ).to(dtype=flow.int32),
                requires_grad=False,
            )
        else:
            self.image = flow.randint(
                0, high=256, size=self.image_shape, dtype=flow.float32, device="cuda"
            )
            self.label = flow.randint(
                0, high=self.num_classes, size=self.label_shape, device="cuda",
            ).to(dtype=flow.int32)

    def forward(self):
        return self.image, self.label
