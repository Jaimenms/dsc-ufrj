# Primeiro caso: um duto, duas fronteiras (hello world)

*Documentação do mfsim (índice no [README](../README.md)). Vocabulário completo dos ficheiros
em [data_formats.md](data_formats.md).*

Um caso do mfsim é **um ficheiro JSON** com tudo o que a simulação precisa: fluido, rede
(vértices e arestas), tempo, integrador e estado inicial. O caso mais simples é um duto
horizontal com vazões impostas na entrada e pressão imposta na saída. Está em
[`cases/examples/hello_world.json`](../cases/examples/hello_world.json) e corre em menos de
um segundo:

```bash
uv run mfsim validate cases/examples/hello_world.json      # obedece ao schema mfsim-case/1?
uv run mfsim run cases/examples/hello_world.json --out runs/hello_world.npz
```

## O ficheiro, bloco a bloco

```json
{
 "schema": "mfsim-case/1",
 "name": "hello_world",
 "description": "Duto horizontal de 1000 m: vazões impostas na entrada, pressão imposta na saída.",
```
`schema` identifica o formato (é obrigatório e validado); `name` dá o nome à corrida.

```json
 "fluid": {"type": "polytropic", "rhoL0": 860.0, "rhoG0": 80.0, "muL": 0.008, "muG": 2e-05,
           "gammaL": 1000.0, "gammaG": 1.2, "P0": 10000000.0, "sigma": 0.03},
 "reference": {"PR": 10000000.0},
 "time": {"t0": 0.0, "tf": 600.0, "dt_out": 10.0},
```
Fluido politrópico (densidades em `P0`, expoentes `gammaL`/`gammaG`; unidades SI). `PR` é a
pressão de referência: no vetor de estados a pressão aparece adimensional, `PT = P/PR`, e por
isso 100 bar com `PR = 1e7` Pa é `PT = 1`. O tempo diz de `t0` a `tf` com saída a cada `dt_out` (e sempre em `tf`).

```json
 "network": {
  "vertices": [
   {"id": "inlet",  "kind": "boundary", "spec": {"qL": 0.5, "qG": {"ramp": [100.0, 130.0], "from": 0.1, "to": 0.13}}},
   {"id": "outlet", "kind": "boundary", "spec": {"P": 10000000.0}}
  ],
```
A rede é um grafo. Os vértices de fronteira (`boundary`) levam as **especificações** que os
dutos vão ler: na entrada as vazões mássicas de líquido e gás (kg/s; a de gás sobe de 0.10
para 0.13 kg/s entre 100 e 130 s, uma rampa), na saída a pressão (`P` em Pa, ou `PT`
adimensional). Uma especificação é um número, uma rampa `{"ramp": [t1, t2], "from", "to"}` ou
uma tabela `{"t": [...], "v": [...]}`. Há ainda vértices `junction` (algébricos, para ligar
dutos) e `tank` (com volume e estados próprios).

```json
  "edges": [
   {"id": "line", "type": "pipe", "from": "inlet", "to": "outlet", "D": 0.1, "roughness": 0.001,
    "geometry": {"points": [{"xyz": [0.0, 0.0, 0.0]}, {"xyz": [1000.0, 0.0, 0.0]}], "segments": [{"n": 20}]},
    "model":  {"type": "two_fluid", "regime": {"type": "fixed", "regime": "stratified_smooth"}},
    "scheme": {"type": "galerkin"},
    "ports": {"in":  {"alphaG": "free", "PT": "free",      "qL": "dirichlet", "qG": "dirichlet"},
              "out": {"alphaG": "free", "PT": "dirichlet", "qL": "free",      "qG": "free"}}}
  ]
 },
```
A aresta é um duto de `inlet` para `outlet`: diâmetro e rugosidade (m), geometria por pontos
`xyz` com o número de elementos de cada segmento (aqui 1000 m horizontais em 20 elementos),
o **modelo** (a dois fluidos, padrão estratificado fixo; há `drift_flux_bb`, `single_liquid`,
`single_gas` e identificação automática de padrão) e o **esquema** numérico (Galerkin; há
diferenças finitas e volumes finitos escalonados). As `ports` dizem, para cada extremidade e
cada variável de estado (`alphaG`, `PT`, `qL`, `qG`), se o valor é imposto pelo vértice
(`dirichlet`), se a derivada é imposta (`neumann`) ou se é calculado pelo duto (`free`): na
entrada impõem-se as vazões, na saída a pressão. Outras arestas: `connector`, `choke`,
`reservoir`, `check_valve` (comprimento zero, ligam vértices).

```json
 "integration": {"integrator": "bdf", "rtol": 0.001, "atol": 0.001, "max_step": 1.0},
 "initial_state": {"source": "guess"}
}
```
`bdf` é o BDF do SciPy; `implicit` é o BDF próprio do mfsim (o usado na intermitência severa,
com `"backend": "jax"` para o lado direito compilado), e há `radau` e `LSODA`. Os campos
omitidos ficam com os defaults do integrador. O estado inicial `guess` é uma estimativa de
escoamento estabelecido a partir das especificações; `edges` usa as condições iniciais
declaradas nos dutos (`ic`: arrays num npz ao lado ou uma receita).

## Correr e ler os resultados

`mfsim run` imprime o resumo (passos, chamadas do lado direito, tempo) e grava a corrida em
npz v2: trajetória `T`/`X`, `info_json`, proveniência e o caso completo em `case_json`.
Em Python:

```python
import matplotlib.pyplot as plt
from mfsim.runs import load_run

run = load_run("runs/hello_world.npz")
res = run.res                      # Results: saídas por duto e por vértice, nomes da tese
p = res.pipe(0)                    # duto 0: p["x"] (nós), p["out"][nome] com forma (nós, tempos)
P, aG = p["out"]["P_bar"], p["out"]["alphaG"]
print(f"pressão na entrada: {P[0, 0]:.3f} -> {P[0, -1]:.3f} bar; fração de gás na entrada {aG[0, -1]:.3f}")
plt.plot(run.T, P[0], label="entrada"); plt.plot(run.T, P[-1], label="saída")
plt.xlabel("t (s)"); plt.ylabel("P (bar)"); plt.legend(); plt.show()
```
`p["out"]` tem as saídas com os nomes da tese (`pressao`, `alfaG`, `vazaoMassaL`,
`velocidadeL`, ...) e os aliases `P_bar`, `alphaG`, `qL`, `qG`; `mfsim.plotting.pipe_surfaces`
e `pipe_timeseries` desenham as figuras no estilo da tese. `run.case` é o `Network` da corrida
e `run.network(tf=...)` uma cópia para continuar ou variar.

## O mesmo caso em Python

O JSON é a imagem de um grafo `Network`; dá para construir o caso em código, simular sem
gravar, e exportar para JSON quando estiver pronto:

```python
import numpy as np
from mfsim import BoundaryVertex, Fluid, Network, Pipe, specs
from mfsim.network.schedules import Ramp
from mfsim.simulation import simulate
from mfsim.io import case_to_json

fluid = Fluid(rhoL0=860.0, rhoG0=80.0, muL=0.008, muG=2e-5, gammaL=1000.0, gammaG=1.2, P0=100e5, sigma=0.03)
inlet = BoundaryVertex("inlet", spec=specs(qL=0.5, qG=Ramp(100.0, 130.0, 0.10, 0.13)))
outlet = BoundaryVertex("outlet", spec=specs(PT=100e5 / 1e7))
line = Pipe("line", "inlet", "outlet", points=np.array([[0.0, 0, 0], [1000.0, 0, 0]]), n_elements=np.array([20]),
            D=0.1, roughness=1e-3)   # modelo, esquema e portas por omissão = os do JSON acima
net = Network.build("hello_world", fluid, [inlet, outlet], [line], tf=600.0, dt_out=10.0, PR=1e7,
                    integrator="bdf", rtol=1e-3, atol=1e-3, max_step=1.0, initial="guess")
res = simulate(net, progress=False)
case_to_json(net, "hello_world.json")
```

## Variar sem copiar o ficheiro

Para experimentar outro integrador ou outra tolerância não é preciso um ficheiro novo:

```bash
uv run mfsim run cases/examples/hello_world.json --set integration.integrator=implicit --set integration.rtol=1e-4
uv run mfsim run cases/examples/hello_world.json --set network.edges.line.geometry.segments.0.n=50
```
`--set caminho=valor` altera qualquer campo (um segmento sobre uma lista é o `id` do elemento
ou o índice) e a corrida grava o caso efetivo. Uma variante que valha a pena guardar herda o
caso base e traz só o que muda:

```json
{"schema": "mfsim-case/1", "$extends": "hello_world.json", "name": "hello_world_implicit",
 "integration": {"integrator": "implicit", "rtol": 1e-4}}
```

## Para onde ir a seguir

- [data_formats.md](data_formats.md): todas as secções e opções (modelos, padrões, esquemas,
  tanques, chokes, reservatórios, receitas de condição inicial, `$extends`/`$ref`), o schema e
  o formato das corridas.
- [thesis_reproduction.md](thesis_reproduction.md): os casos do capítulo 10 da tese
  (`cases/chapter10/`, notebooks em `notebooks_src/`); o 10.1.1 é este caso com 50 elementos.
- [severe_slug.md](severe_slug.md) e [lab_validation.md](lab_validation.md): intermitência
  severa com o esquema conservativo, o BDF próprio e o backend JAX (`cases/severe_slug/`).
