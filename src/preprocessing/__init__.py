"""Preprocessing package for the lumbar spine MRI segmentation project.

Sub-modules
-----------
extract      Reproducible extraction of the raw ZIP archives.
volume_io    Reading ``.mha`` volumes and their geometric metadata.
labels       SPIDER label semantics and label remapping.
pairing      Image <-> mask pairing driven by filename identifiers.
validate     Dataset-level integrity and quality checks.
transforms   Image/mask preprocessing operations.
visualize    Qualitative before/after figures.
splits       Patient-level train/validation/test splitting.
"""
