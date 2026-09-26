"""Image statistical characterization for cross-sensor lunar imagery.

Computes comprehensive first-order statistics, percentile distributions,
and zero/saturation fractions for raw sensor imagery.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..io.raster_reader import BaseRasterReader


@dataclass(frozen=True)
class ImageStatistics:
    """Comprehensive descriptive statistics of a raw sensor image raster."""

    sensor_name: str
    dtype: str
    shape: Tuple[int, ...]
    min: float
    max: float
    mean: float
    median: float
    std: float
    p1: float
    p5: float
    p50: float
    p95: float
    p99: float
    fraction_zero: float
    fraction_below_p5: float
    fraction_saturated: float
    saturation_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


def compute_image_statistics(
    arr: np.ndarray,
    sensor_name: str = "GENERIC",
    saturation_threshold: Optional[float] = None,
) -> ImageStatistics:
    """Compute exact statistical metrics for an in-memory 2D or 3D NumPy array.

    Does not modify or mutate the input array.

    Args:
        arr: Input numpy array (uint8, uint16, float, etc.).
        sensor_name: Sensor identifier string.
        saturation_threshold: Intensity value considered saturated. Defaults to max possible
                              value of the integer dtype (255 for uint8, 65535 for uint16).

    Returns:
        Populated ImageStatistics instance.
    """
    if arr.size == 0:
        raise ValueError("Cannot compute statistics on an empty array.")

    # Determine default saturation threshold based on dtype
    if saturation_threshold is None:
        if np.issubdtype(arr.dtype, np.integer):
            saturation_threshold = float(np.iinfo(arr.dtype).max)
        else:
            saturation_threshold = float(np.max(arr))

    # Compute percentiles
    p1, p5, p50, p95, p99 = [float(x) for x in np.percentile(arr, [1.0, 5.0, 50.0, 95.0, 99.0])]

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr))

    total_pixels = arr.size
    frac_zero = float(np.count_nonzero(arr == 0) / total_pixels)
    frac_p5 = float(np.count_nonzero(arr <= p5) / total_pixels)
    frac_sat = float(np.count_nonzero(arr >= saturation_threshold) / total_pixels)

    return ImageStatistics(
        sensor_name=sensor_name,
        dtype=str(arr.dtype),
        shape=tuple(arr.shape),
        min=min_val,
        max=max_val,
        mean=mean_val,
        median=median_val,
        std=std_val,
        p1=p1,
        p5=p5,
        p50=p50,
        p95=p95,
        p99=p99,
        fraction_zero=frac_zero,
        fraction_below_p5=frac_p5,
        fraction_saturated=frac_sat,
        saturation_threshold=float(saturation_threshold),
    )


def compute_streaming_image_statistics(
    reader: BaseRasterReader,
    sensor_name: str = "OHRC",
    chunk_lines: int = 10000,
    saturation_threshold: Optional[float] = None,
) -> ImageStatistics:
    """Compute exact full-image statistics by streaming chunks from disk.

    Avoids loading gigabyte-scale rasters into system RAM while calculating exact
    min, max, mean, standard deviation, and histogram-based percentiles.

    Args:
        reader: Instantiated BaseRasterReader (e.g. OHRCRasterReader).
        sensor_name: Sensor identifier string.
        chunk_lines: Number of image lines to stream per buffer.
        saturation_threshold: Saturation threshold intensity.

    Returns:
        Populated ImageStatistics instance representing the entire raster.
    """
    total_lines = reader.lines
    samples = reader.samples
    dtype = np.dtype(reader.dtype)

    if saturation_threshold is None:
        if np.issubdtype(dtype, np.integer):
            saturation_threshold = float(np.iinfo(dtype).max)
        else:
            saturation_threshold = 255.0

    # Determine histogram bins based on dtype
    is_uint8 = dtype == np.uint8
    hist_size = 256 if is_uint8 else 65536
    hist = np.zeros(hist_size, dtype=np.int64)

    total_sum: float = 0.0
    total_sq_sum: float = 0.0
    total_count: int = 0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    count_zero: int = 0
    count_sat: int = 0

    for start_line in range(0, total_lines, chunk_lines):
        end_line = min(start_line + chunk_lines, total_lines)
        chunk = reader.read_region(start_line, end_line, 0, samples)

        c_min = float(np.min(chunk))
        c_max = float(np.max(chunk))
        if c_min < min_val:
            min_val = c_min
        if c_max > max_val:
            max_val = c_max

        count_zero += int(np.count_nonzero(chunk == 0))
        count_sat += int(np.count_nonzero(chunk >= saturation_threshold))

        total_sum += float(np.sum(chunk, dtype=np.float64))
        total_sq_sum += float(np.sum(chunk.astype(np.float64) ** 2))
        total_count += chunk.size

        # Accumulate exact histogram
        if is_uint8:
            hist += np.bincount(chunk.ravel(), minlength=256)
        else:
            c_hist, _ = np.histogram(chunk, bins=hist_size, range=(0, hist_size))
            hist += c_hist

    if total_count == 0:
        raise ValueError("Reader returned zero pixels.")

    mean_val = total_sum / total_count
    var_val = max(0.0, (total_sq_sum / total_count) - (mean_val**2))
    std_val = float(np.sqrt(var_val))

    # Exact percentiles from cumulative histogram
    cum_hist = np.cumsum(hist)
    p1 = float(np.searchsorted(cum_hist, 0.01 * total_count))
    p5 = float(np.searchsorted(cum_hist, 0.05 * total_count))
    p50 = float(np.searchsorted(cum_hist, 0.50 * total_count))
    p95 = float(np.searchsorted(cum_hist, 0.95 * total_count))
    p99 = float(np.searchsorted(cum_hist, 0.99 * total_count))

    frac_zero = count_zero / total_count
    frac_p5 = float(cum_hist[int(p5)]) / total_count
    frac_sat = count_sat / total_count

    return ImageStatistics(
        sensor_name=sensor_name,
        dtype=str(dtype),
        shape=(total_lines, samples),
        min=min_val,
        max=max_val,
        mean=mean_val,
        median=p50,
        std=std_val,
        p1=p1,
        p5=p5,
        p50=p50,
        p95=p95,
        p99=p99,
        fraction_zero=frac_zero,
        fraction_below_p5=frac_p5,
        fraction_saturated=frac_sat,
        saturation_threshold=float(saturation_threshold),
    )
