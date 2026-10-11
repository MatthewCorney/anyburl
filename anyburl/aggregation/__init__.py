"""Learned aggregation of rule evidence into candidate rankings."""

from .features import CandidateFeature, CandidateFeaturizer, build_features

__all__ = ["CandidateFeature", "CandidateFeaturizer", "build_features"]
