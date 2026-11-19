"""PROTOTYPE (not ready to merge): a native pass for MEDS plain predicates.

``plain_predicates(path, predicates, prune)`` returns exactly what
``aces.predicates.generate_plain_predicates_from_meds`` returns, from one PyArrow read and one C++ pass per
shard (``csrc/aces_native.cc``):

- codes stay dictionary-encoded, and each predicate's code condition is evaluated once per distinct code, with
  ACES's own Polars expression (so regex / any / exact-match semantics are ACES's);
- the C++ pass drops null and MEDS_BIRTH codes, applies value bounds with Polars semantics (NaN above every
  number; a null value makes the predicate null, counted as 0), applies the prune filter (null timestamp, a
  subject's first or last timestamp, or a predicate holds), and sums per (subject, timestamp) in
  first-appearance order;
- the C++ call does not hold the GIL, so shards can run on threads (``plain_predicates_many``).

Nothing in ACES calls this yet: it is opt-in. It falls back to ACES when the extension is not built, a
predicate uses ``other_cols``, there are more than 31 plain predicates, or the columns have unexpected types.
"""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import meds
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from . import predicates as aces_predicates
from .config import PlainPredicateConfig
from .types import PRED_CNT_TYPE


class Column(ct.Structure):
    _fields_ = (("validity", ct.c_void_p), ("data", ct.c_void_p), ("offset", ct.c_int64))


class Batch(ct.Structure):
    _fields_ = (
        ("length", ct.c_int64),
        ("subjects", Column),
        ("times", Column),
        ("codes", Column),
        ("code_flags", ct.c_void_p),
        ("dictionary_length", ct.c_int64),
        ("values", Column),
        ("values_64", ct.c_int32),
    )


class BoundsT(ct.Structure):
    _fields_ = (
        ("has_min", ct.c_int32),
        ("min_inclusive", ct.c_int32),
        ("has_max", ct.c_int32),
        ("max_inclusive", ct.c_int32),
        ("min", ct.c_double),
        ("max", ct.c_double),
    )


def _library_path() -> Path | None:
    here = Path(__file__).parent
    return next(here.glob("_native*.pyd"), None) or next(here.glob("_native*.so"), None)


def available() -> bool:
    """Whether the native extension is built."""
    return _library_path() is not None


@cache
def lib() -> ct.CDLL:
    path = _library_path()
    if path is None:
        raise RuntimeError("The aces._native extension is not built.")
    library = ct.CDLL(str(path))
    library.acn_meds_predicates.restype = ct.c_void_p
    library.acn_meds_predicates.argtypes = [
        ct.POINTER(Batch),
        ct.c_int64,
        ct.c_int32,
        ct.POINTER(BoundsT),
        ct.c_int32,
        ct.c_int32,
    ]
    library.acn_last_error.restype = ct.c_char_p
    library.acn_result_rows.restype = ct.c_int64
    for name in ("subjects", "times", "counts"):
        getattr(library, f"acn_result_{name}").restype = ct.POINTER(ct.c_int64)
    for name in ("subject_valid", "time_valid"):
        getattr(library, f"acn_result_{name}").restype = ct.POINTER(ct.c_uint8)
    for name in ("rows", "subjects", "times", "counts", "subject_valid", "time_valid", "free"):
        getattr(library, f"acn_result_{name}").argtypes = [ct.c_void_p]
    return library


def _column(array: pa.Array | None) -> Column:
    if array is None:
        return Column(None, None, 0)
    buffers = array.buffers()
    validity = buffers[0].address if buffers[0] is not None and array.null_count else None
    data = buffers[1].address if len(buffers) > 1 and buffers[1] is not None else None
    return Column(validity, data, array.offset)


def supported(predicates: dict[str, PlainPredicateConfig]) -> bool:
    """Whether the native pass handles these predicates (otherwise plain_predicates calls ACES)."""
    return available() and len(predicates) <= 31 and not any(p.other_cols for p in predicates.values())


def _code_flags(dictionary: pa.Array, predicates: dict[str, PlainPredicateConfig]) -> np.ndarray:
    """Per dictionary entry: dropped (null or MEDS_BIRTH), then each predicate's code condition, evaluated
    with ACES's own expression on the distinct codes."""
    codes = pl.DataFrame({"code": pl.from_arrow(dictionary.cast(pa.string()))})
    exprs = [(pl.col("code").is_null() | (pl.col("code") == meds.birth_code)).alias("_drop")]
    exprs += [
        PlainPredicateConfig(code=p.code).MEDS_eval_expr().fill_null(False).alias(f"_{i}")
        for i, p in enumerate(predicates.values())
    ]
    return np.ascontiguousarray(codes.select(exprs).to_numpy().astype(np.uint8))


def plain_predicates(path, predicates: dict[str, PlainPredicateConfig], prune: bool = False) -> pl.DataFrame:
    """aces.predicates.generate_plain_predicates_from_meds(path, predicates, prune), natively."""
    if not supported(predicates):
        return aces_predicates.generate_plain_predicates_from_meds(path, predicates, prune=prune)
    names = list(predicates)
    bounded = any(p.value_min is not None or p.value_max is not None for p in predicates.values())
    columns = ["subject_id", "time", "code"] + (["numeric_value"] if bounded else [])
    table = pq.read_table(path, columns=columns, read_dictionary=["code"])
    if table["subject_id"].type != pa.int64() or not pa.types.is_timestamp(table["time"].type):
        return aces_predicates.generate_plain_predicates_from_meds(path, predicates, prune=prune)
    value_type = table["numeric_value"].type if bounded else pa.float64()
    if value_type not in (pa.float32(), pa.float64()):
        return aces_predicates.generate_plain_predicates_from_meds(path, predicates, prune=prune)
    code = table["code"]
    if not (pa.types.is_dictionary(code.type) and code.type.index_type == pa.int32()):
        code = code.cast(pa.string()).dictionary_encode()
        table = table.set_column(table.schema.get_field_index("code"), "code", code)
    round_value = (lambda x: float(np.float32(x))) if value_type == pa.float32() else float
    bounds = (BoundsT * max(len(names), 1))(
        *(
            BoundsT(
                p.value_min is not None,
                bool(p.value_min_inclusive),
                p.value_max is not None,
                bool(p.value_max_inclusive),
                round_value(p.value_min) if p.value_min is not None else 0.0,
                round_value(p.value_max) if p.value_max is not None else 0.0,
            )
            for p in predicates.values()
        )
    )
    batches, keep, last = [], [], None
    for batch in table.to_batches():
        dictionary = batch["code"].dictionary
        if last is None or not dictionary.equals(last[0]):
            last = (dictionary, _code_flags(dictionary, predicates))
        flags = last[1]
        keep += [batch, flags]
        times = batch["time"].cast(pa.int64())
        keep.append(times)
        batches.append(
            Batch(
                batch.num_rows,
                _column(batch["subject_id"]),
                _column(times),
                _column(batch["code"].indices),
                flags.ctypes.data,
                len(dictionary),
                _column(batch["numeric_value"]) if bounded else Column(None, None, 0),
                value_type == pa.float64(),
            )
        )
    array = (Batch * max(len(batches), 1))(*batches)
    result = lib().acn_meds_predicates(
        array, len(batches), len(names), bounds, int(prune), value_type == pa.float64()
    )
    if not result:
        raise ValueError(lib().acn_last_error().decode())
    try:
        n = lib().acn_result_rows(result)

        def take(name, dtype, count):
            pointer = getattr(lib(), f"acn_result_{name}")(result)
            return (
                np.ctypeslib.as_array(pointer, (count,)).astype(dtype, copy=True)
                if count
                else np.zeros(0, dtype)
            )

        subjects = take("subjects", np.int64, n)
        times = take("times", np.int64, n)
        subject_valid = take("subject_valid", bool, n)
        time_valid = take("time_valid", bool, n)
        counts = take("counts", np.int64, n * len(names)).reshape(n, len(names))
    finally:
        lib().acn_result_free(result)
    time_type = table["time"].type
    subject = pa.array(subjects, type=pa.int64(), mask=None if subject_valid.all() else ~subject_valid)
    timestamp = pa.array(times, type=pa.int64(), mask=None if time_valid.all() else ~time_valid).cast(
        time_type
    )
    frame = pl.from_arrow(pa.table({"subject_id": subject, "timestamp": timestamp}))
    if not names:
        return frame
    counts_frame = pl.from_numpy(counts, schema={n_: PRED_CNT_TYPE for n_ in names}, orient="row")
    frame = frame.hstack(counts_frame)
    return frame


def plain_predicates_many(paths, predicates, prune: bool = False, workers: int = 8) -> list[pl.DataFrame]:
    """plain_predicates of several shards, on threads (the native pass and the reads release the GIL)."""
    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(lambda p: plain_predicates(p, predicates, prune), paths))
