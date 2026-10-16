# 046 · Ponte do filme do anular avaliada com a fração de líquido local

| | |
|---|---|
| tipo | bugfix |
| área | identificação de padrão |
| estado | feito (2026-10-16); opção `regime.local_film` (com `local`), desligada por omissão; ligada no exemplo base do artigo 01 |
| onde | `mfsim/closures/regime.py` (`identify_regime(local_film=)`); `PipeParams.ipe_local_film`; `RegimeSpec.local_film`; `tests/test_local_film.py` |
| dados | sonda de 2026-10-16: estacionário do poço com padrão dinâmico a 15 bar e abertura 0,13 (`cases/severe_slug/well_choke_auto.json`) |

## Antes

O modo local da identificação automática (021) avalia no estado da célula o nível e a fração do
estratificado e a fração de bolhas, mas o critério de ponte do filme do anular (o filme faz ponte e o
escoamento deixa de ser anular quando a sua fração de líquido passa de 0,35; Barnea, 1987) continuava a usar
o filme de equilíbrio de Pots, calculado só a partir das velocidades superficiais. Com velocidades pequenas o
filme de equilíbrio é fino, e uma célula quase cheia de líquido que não seja estratificada saía **anular**:
no estacionário do caso do poço, a célula do fim do descendente onde o bolsão encontra o líquido acumulado no
ponto baixo (fração de gás 0,09, inclinação −3°) era anular nas aberturas 0,06 e 0,13 a 15 bar.

## Depois

Com `local_film`, a fração de líquido do filme no critério é o máximo entre a de equilíbrio e a da célula: uma
célula com mais líquido do que o filme que faz ponte não pode ser anular. Células com pouco líquido (riser
cheio de gás depois do blowout) continuam anulares. Sem `local`, a opção não tem efeito.

## Benefício

- A célula do ponto baixo passa a intermitente; o estacionário quase não muda (o Newton a partir do anterior
  converge em 3 iterações e as frações coincidem a duas casas decimais).
- O mapa de padrões ao longo do duto do artigo deixa de mostrar anular numa célula com 91 % de líquido.

## Referências

Issues 020 e 021 (identificação automática); Barnea (1987), *A unified model for predicting flow-pattern
transitions for the whole range of pipe inclinations*, IJMF 13; artigo 01.
