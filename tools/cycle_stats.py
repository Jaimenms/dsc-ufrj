"""Estatísticas de ciclo de corridas gravadas do severe slug = `mfsim stats` (mfsim.analysis.cycle_stats).
uso: uv run python tools/cycle_stats.py nome1 nome2 ...   (nomes em notebooks/data/, runs/ ou caminhos .npz)"""
import sys
import warnings

warnings.filterwarnings("ignore")
from mfsim.cli import main  # noqa: E402

sys.exit(main(["stats", *sys.argv[1:]]))
