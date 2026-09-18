"""LatticeRank edge-registration package.

OpenCV keeps its own thread pool and its own RNG, neither of which respects
the OMP/MKL environment variables the entry points set. Left alone, parallel
float reductions and unseeded RANSAC make the confidence column drift in the
last few decimal places between runs. Pinning both here -- at package import,
before any solver runs -- makes a scored run reproducible byte for byte.
"""

__version__ = "2.0.1"

try:  # pragma: no cover - exercised implicitly by every run
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
except Exception:  # OpenCV is optional for the pure-NumPy code paths
    pass
