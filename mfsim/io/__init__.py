"""Entrada e saída de casos: JSON (`mfsim-case/1`) e conversão dos .mf2 legados."""
from .spec import SCHEMA, case_from_json, case_to_json, load_json, network_from_json, network_to_json, save_json

__all__ = ["SCHEMA", "case_from_json", "case_to_json", "load_json", "network_from_json", "network_to_json", "save_json"]
