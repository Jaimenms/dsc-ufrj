"""Casos do capítulo 10 (Resultados de Simulações Dinâmicas).

Cada caso é carregado do arquivo .mf2 original (entrada + resultados do
MATLAB para comparação). As descrições resumem o texto da tese.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..io_legacy import load_case


@dataclass
class CaseInfo:
    key: str
    file: str
    section: str
    title: str
    description: str
    figures: str


CHAPTER10 = {
    "10.1.1": CaseInfo(
        "10.1.1", "CASO10-1-1.mf2", "10.1.1", "Escoamento horizontal estratificado",
        "Duto horizontal de 1000 m, D=0.1 m, 50 elementos, Galerkin, modelo a 4 equações com "
        "padrão estratificado fixo. qL=0.5 kg/s; qG sobe de 0.1 para 0.13 kg/s entre 150 e 180 s. "
        "Pressão de saída 100 bar.", "Figs. 10.2-10.8, Tabela 10.1"),
    "10.1.2": CaseInfo(
        "10.1.2", "CASO10-1-2.mf2", "10.1.2", "Escoamento horizontal anular",
        "Mesmo duto com padrão anular fixo. qL=0.5 kg/s; qG sobe de 5.0 para 5.5 kg/s entre 150 e 180 s.",
        "Figs. 10.10-10.13"),
    "10.1.3": CaseInfo(
        "10.1.3", "CASO10-1-3.mf2", "10.1.3", "Escoamento horizontal padrão bolhas",
        "Mesmo duto com padrão bolhas fixo. qG=0.1 kg/s; qL sobe de 50 para 55 kg/s entre 150 e 180 s.",
        "Figs. 10.15-10.16"),
    "10.1.4": CaseInfo(
        "10.1.4", "CASO10-1-4.mf2", "10.1.4", "Escoamento horizontal via modelo a 3 equações",
        "Cenário de 10.1.1 com o modelo de mistura (Beggs & Brill).", "Fig. 10.17"),
    "10.2.1A": CaseInfo(
        "10.2.1A", "CASO10-2-1.mf2", "10.2.1", "Dois trechos em série (formulação A: conexão)",
        "Dois dutos de 500 m (25 elementos cada) ligados por especificação de conexão; "
        "cenário de 10.1.1; diferenças finitas centrais.", "Figs. 10.20-10.24, Tabelas 10.2-10.3"),
    "10.2.1B": CaseInfo(
        "10.2.1B", "CASO10-2-1b.mf2", "10.2.1", "Dois trechos em série (formulação B: vértice dinâmico)",
        "Dois dutos de 500 m acoplados por vértice dinâmico (tipo A).", "Figs. 10.20-10.24"),
    "10.2.2": CaseInfo(
        "10.2.2", "CASO10-2-2.mf2", "10.2.2", "Rede com ciclo",
        "Cinco dutos de 200 m (10 elementos) formando um ciclo; dutos 4 e 5 com D=0.05 m; "
        "vértices dinâmicos nas junções.", "Figs. 10.26-10.30"),
    "10.3.1": CaseInfo(
        "10.3.1", "CASO10-3-1.mf2", "10.3.1", "Poço simples",
        "Poço vertical de 2048 m (5 in), modelo a 3 equações (intermitente), reservatório "
        "(170 bar, IPL 0.71, IPG 0.042 kg/s/bar) e choke (Cv=30) com rampa 100%->25% (60-90 s) "
        "e 25%->100% (500-530 s).", "Figs. 10.32-10.34"),
    "10.3.2a": CaseInfo(
        "10.3.2a", "CASO10-3-2.mf2", "10.3.2", "Poço com gas lift - cenário 1 (sem intermitência)",
        "Choke de gás 100%->75% entre 100 e 150 s.", "Figs. 10.36-10.40"),
    "10.3.2b": CaseInfo(
        "10.3.2b", "CASO10-3-2b.mf2", "10.3.2", "Poço com gas lift - cenário 2 (com intermitência)",
        "Choke de gás 75%->15% entre 100 e 150 s.", "Figs. 10.41-10.46"),
    "10.3.3": CaseInfo(
        "10.3.3", "CASO10-3-3b.mf2", "10.3.3", "Poço offshore",
        "Poço + linha de produção (3500 m horizontal + trecho descendente) + riser de 1500 m, "
        "gas lift com linha de gás e anular; choke de gás 100%->80% entre 100 e 200 s.",
        "Figs. 10.48-10.54"),
}


def load_chapter10_case(key, **kw):
    """Retorna (case, legacy_raw) do caso `key` (ex.: '10.1.1')."""
    info = CHAPTER10[key]
    if key == "10.3.2a":
        kw = dict(kw, ic_from="file")   # o arquivo original não monta (válvula de retenção em vértice estático)
    case, raw = load_case(info.file, **kw)
    case.name = f"{info.section} {info.title}"
    if key == "10.3.2a":
        # o arquivo do cenário 1 tem o vértice 8 como estático (editado após a
        # simulação); a rede da tese usa vértice dinâmico com válvula de retenção,
        # como no arquivo do cenário 2
        case_b, _ = load_case("CASO10-3-2b.mf2", ic_from="legacy_t0")
        valve_tables = {v.vertex: v for v in case.valves}
        case_b.valves = [valve_tables.get(v.vertex, v) for v in case_b.valves]
        case_b.name = case.name
        case = case_b
    return case, raw
