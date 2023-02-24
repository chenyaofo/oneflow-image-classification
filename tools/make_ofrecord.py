'''
This script aims to create tfrecord tar shards with multi-processing.
'''

import os
import six
import random
import datetime
import json
import struct
from functools import lru_cache
from multiprocessing import Process
from flowvision.datasets.folder import ImageFolder

import oneflow.core.record.record_pb2 as ofrecord

CODES_LEN = 4
ENCODING = "utf-8"


def int32_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int32_list=ofrecord.Int32List(value=value))


def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(float_list=ofrecord.FloatList(value=value))


def double_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(double_list=ofrecord.DoubleList(value=value))


def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=value))

type_map = {
    "int": int32_feature,
    "int32": int32_feature,
    "int64": int64_feature,
    "float": float_feature,
    "float32": float_feature,
    "float64": double_feature,
    "double": double_feature,
    "byte": bytes_feature,
}


class OFRecordWriter:
    def __init__(self, filename, mode="wb"):
        self.f = open(filename, mode=mode)

    def write(self, sample):
        if not isinstance(sample, dict):
            raise ValueError("sample must be a dict")

        to_pack_dict = dict()
        for key, (value, type_) in sample.items():
            to_pack_dict[key] = type_map[type_](value)

        ofrecord_features = ofrecord.OFRecord(feature=to_pack_dict)
        serilizedBytes = ofrecord_features.SerializeToString()
        length = ofrecord_features.ByteSize()

        self.f.write(struct.pack("q", length))
        self.f.write(serilizedBytes)

    def close(self):
        self.f.close()

@lru_cache(maxsize=128)
def zero_bytes(legnth: int):
    return b'0'*legnth


def jsonpack(data: dict, maxlen: int):
    raw_bytes = json.dumps(data).encode(encoding=ENCODING)
    raw_len = len(raw_bytes)

    codes = struct.pack("<I", raw_len)
    codes_len = len(codes)

    if raw_len + codes_len > maxlen:
        raise ValueError(f"The length of raw data ({raw_len+codes_len}) is larger than limited value ({maxlen}).")

    out = codes + raw_bytes + zero_bytes(maxlen-raw_len-codes_len)

    return out



def make_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))
    if num_workers == 0:
        write_partial_samples(
            pattern,
            shard_ids,
            samples_per_shards,
            map_func,
            kwargs
        )
    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")

    writer = OFRecordWriter(fname)
    size = 0
    for i, item in enumerate(samples):
        raw_data = map_func(item)
        size += len(raw_data["image"][0])
        writer.write(raw_data)

        if i % 1000 == 0:
            print(f"[{datetime.datetime.now()}] complete to write {i:06d} samples to shard {fname}")
    writer.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}!!!")
    return size


def main(source, dest, num_shards, num_workers):
    root = source
    items = []
    dataset = ImageFolder(root=root,  loader=lambda x: x)
    for i in range(len(dataset)):
        items.append(dataset[i])

    def map_func(item):
        name, class_idx = item
        import filetype
        kind = filetype.guess(name)
        if kind.extension != "jpg":
            raise ValueError()
        # import ipdb; ipdb.set_trace()
        with open(os.path.join(name), "rb") as stream:
            image = stream.read()

        # print(name)
        # import ipdb; ipdb.set_trace()

        # print(os.path.relpath(name, root))

        sample = {
            # "fname": (bytes(os.path.splitext(os.path.basename(name))[0], "utf-8"), "byte"),
            "metadata": (
                jsonpack(dict(path=os.path.relpath(name, root)), maxlen=128),
                "byte"
            ),
            "image": (image, "byte"),
            "label": (class_idx, "int")
        }
        return sample
    make_shards(
        pattern=dest,
        num_shards=num_shards,  # 设置分片数量
        num_workers=num_workers,  # 设置创建wds数据集的进程数
        samples=items,
        map_func=map_func,
    )


if __name__ == "__main__":
    # source = "/gdata/imagenet2012/"
    source = "/home/chenyaofo/datasets/imagenet-c/snow/1/"
    dest = "/home/chenyaofo/datasets/imagenet-c-of/snow/1"
    os.makedirs(dest)
    main(
        source=source,
        dest=os.path.join(dest, "part-%06d"),
        num_shards=256,
        num_workers=8
    )
    # main(
    #     source=os.path.join(source, "train"),
    #     dest=os.path.join(dest, "train", "imagenet-1k-train-%06d.tfrecord"),
    #     num_shards=256,
    #     num_workers=8
    # )
    # main(
    #     source=os.path.join(source, "val"),
    #     dest=os.path.join(dest, "val", "imagenet-1k-val-%06d.tfrecord"),
    #     num_shards=256,
    #     num_workers=8
    # )
