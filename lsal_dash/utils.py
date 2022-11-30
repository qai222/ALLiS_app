import gzip
import json
import os.path
import pickle
import time
from functools import wraps

import monty.json
from loguru import logger
from pandas._typing import FilePath


def json_load(fn: FilePath, timing=True, disable_monty=False):
    ts1 = time.perf_counter()
    if disable_monty:
        decoder = None
    else:
        decoder = monty.json.MontyDecoder

    if fn.endswith(".gz"):
        gz = True
    else:
        gz = False

    if gz:
        with gzip.open(fn, 'rt') as f:
            o = json.load(f, cls=decoder)
    else:
        with open(fn, "r") as f:
            o = json.load(f, cls=decoder)
    ts2 = time.perf_counter()
    if timing:
        logger.info("loaded {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))
    return o


def get_folder(path: FilePath):
    return os.path.dirname(os.path.abspath(path))


def get_basename(path: FilePath):
    return strip_extension(os.path.basename(os.path.abspath(path)))


def strip_extension(p: FilePath):
    return os.path.splitext(p)[0]


def pkl_dump(o, fn: FilePath, print_timing=True) -> None:
    ts1 = time.perf_counter()
    with open(fn, "wb") as f:
        pickle.dump(o, f)
    ts2 = time.perf_counter()
    if print_timing:
        logger.info("dumped {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))


def pkl_load(fn: FilePath, print_timing=True):
    ts1 = time.perf_counter()
    with open(fn, "rb") as f:
        d = pickle.load(f)
    ts2 = time.perf_counter()
    if print_timing:
        logger.info("loaded {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))
    return d


def json_dump(o, fn: FilePath, gz=False, timing=True):
    ts1 = time.perf_counter()
    if gz:
        with gzip.open(fn, 'wt') as f:
            json.dump(o, f, cls=monty.json.MontyEncoder)
    else:
        with open(fn, "w") as f:
            json.dump(o, f, cls=monty.json.MontyEncoder)
    ts2 = time.perf_counter()
    if timing:
        logger.info("dumped {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))


def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def file_exists(fn: FilePath):
    return os.path.isfile(fn) and os.path.getsize(fn) > 0


def critical_step(method):
    @wraps(method)
    def load_method(*args, **kwargs):
        name = method.__name__
        logger.critical(f"CALLED: {name}")
        return method(*args, **kwargs)

    return load_method


def rgb_to_rgba(rgb: str, a: float):
    rgba = rgb.strip().strip(")")
    rgba += ", {:.4f})".format(a)
    return rgba.replace("rgb", "rgba")
