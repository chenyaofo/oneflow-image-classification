import pathlib
import shutil
import typing

import oneflow as flow


def rank() -> int:
    """

    Returns:
        int: The rank of the current node in distributed system, return 0 if distributed 
        mode is not initialized.
    """
    return flow.env.get_rank()


def world_size() -> int:
    """

    Returns:
        int: The world size of the distributed system, return 1 if distributed mode is not 
        initialized.
    """
    return flow.env.get_world_size()


def is_master() -> bool:
    """

    Returns:
        int: True if the rank current node is euqal to 0. Thus it will always return True if 
        distributed mode is not initialized.
    """
    return rank() == 0


def is_dist_init() -> bool:
    """

    Returns:
        bool: True if distributed mode is initialized correctly, False otherwise.
    """
    return rank() > 1


def flowsave(obj: typing.Any, f: str) -> None:
    """A simple warp of flow.save. This function is only performed when the current node is the
    master. It will do nothing otherwise. 

    Args:
        obj (typing.Any): The object to save.
        f (str): The output file path.
    """
    if is_master():
        f: pathlib.Path = pathlib.Path(f)
        tmp_name = f.with_name("model.tmp")
        flow.save(obj, tmp_name)
        shutil.move(tmp_name, f)
