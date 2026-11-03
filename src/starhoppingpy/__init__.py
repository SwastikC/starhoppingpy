"""starhoppingpy: VLT/SPHERE-IRDIS star-hopping reduction pipeline.

Modules
-------
sh_sort    Sort raw SPHERE frames into science/calibration sets (uses dfits/fitsort).
sh_funcs   Calibrations (flat, sky, bad pixels), centering, basic reduction, ASDI cubes.
basic      Low-level image utilities (centroiding, fixpix, robust statistics, ...).
functions  LOCI / RDI helper routines.
zfuncs     FITS I/O, plotting and photometry helpers.
"""

__version__ = "0.1.0"
