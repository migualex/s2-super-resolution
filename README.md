# s2-super-resolution

<p align="center">
  <img src="https://img.shields.io/badge/License-GPLv3-blue" alt="GPLv3" />
  <img src="https://img.shields.io/badge/lifecycle-experimental-orange.svg" alt="experimental" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
</p>

Super-resolução de imagens Sentinel-2 de **10 m para 2,5 m** (fator 4) com modelos de difusão latente (**LDSR-S2** / **SEN2SR**), aplicada aos mosaicos de 16 dias do **Brazil Data Cube (BDC, coleção `S2-16D-2`)**. O produto final é a composição **PRODES** (R = B11, G = B08, B = B04) em 2,5 m, pensada para a interpretação visual de desmatamento.

## Sumário

- [Visão geral](#visão-geral)
- [Modos de super-resolução](#modos-de-super-resolução)
- [Instalação](#instalação)
- [Uso](#uso)
- [Fontes de dados](#fontes-de-dados)
- [Saídas](#saídas)
- [Visualização e comparação](#visualização-e-comparação)
- [Desempenho e hardware](#desempenho-e-hardware)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Referências](#referências)
- [Licença](#licença)

## Visão geral

```
BDC STAC (S2-16D-2, tile + data)
        │  download dos COGs (ou leitura local / remota / cena L2A)
        ▼
  Bandas 10 m (B04 B03 B02 B08) + 20 m (B11, ...)
        │  suavização gaussiana das bandas 20 m (remove os blocos 2×2 do BDC)
        ▼
  Janelas de 128 px com sobreposição ──► LDSR-S2 / SEN2SR na GPU
        │  casamento de histograma com a entrada, mistura das bordas
        ▼
  GeoTIFF 2,5 m (Int16, reflectância × 10 000)
        │
        ▼
  Composição PRODES  R=B11  G=B08  B=B04
```

Principais características:

- detecção automática de GPU (inclusive partições MIG), escolha de precisão (fp32 + TF32 ou bf16) e ajuste do tamanho do lote pela memória disponível;
- processamento do tile inteiro ou de um quadrado de N km em torno de um ponto;
- **retomada automática**: se a execução for interrompida, rodar de novo continua de onde parou;
- controle de qualidade do mosaico pela banda `PROVENANCE` (densidade de emendas entre cenas);
- log e relatório JSON por execução, com todos os parâmetros, o hardware e os tempos de cada etapa.

## Modos de super-resolução

Todos os modos geram a composição PRODES em 2,5 m. A diferença está em como a banda **B11** (SWIR, 20 m) é obtida.

| Modo | Modelo | Bandas | B11 | Tempo aprox. por janela |
|---|---|---|---|---|
| `fast` (padrão) | LDSR-S2 | B04 B03 B02 B08 | Injeção de detalhe guiada pela saída do LDSR-S2 | ~1 s |
| `combined` | LDSRS2-SEN2SR (OpenSR) | 10 bandas | Saída do SEN2SR | ~55 s |
| `separate` | LDSR-S2 + SEN2SR | 4 + B11 | SEN2SR rodado à parte | entre os dois |

No modo `fast`, a B11 em 2,5 m é estimada por regressão (pansharpening): em cada janela ajusta-se `B11 ≈ c0 + Σ ci·banda_i` com as bandas de 10 m, aplica-se o mesmo ajuste às bandas super-resolvidas para obter só as altas frequências, e corrige-se o resultado para que cada bloco 4×4 tenha a média da B11 original.

Há ainda `--engine bicubic`, uma interpolação bicúbica simples para servir de referência nas comparações.

## Instalação

Requisitos: Python 3.10 ou mais recente; GPU NVIDIA com CUDA é fortemente recomendada (funciona em CPU, mas muito devagar).

```bash
git clone https://github.com/migualex/s2-super-resolution.git
```

```bash
cd s2-super-resolution
```

1. Instale o PyTorch compatível com a sua versão de CUDA, seguindo https://pytorch.org/get-started/locally/.
2. Instale as demais dependências:

```bash
pip install -r requirements.txt
```

Para os modos `combined` e `separate`, instale também `mlstac` e `sen2sr` (o modelo SEN2SR completo depende de `mamba_ssm`, que exige CUDA 12). Se faltar alguma dependência, o script tenta instalá-la sozinho na primeira execução.

Os pesos dos modelos são baixados automaticamente na primeira execução para `work/model/` (o LDSRS2-SEN2SR ocupa cerca de 1,5 GB).

## Uso

### Como notebook (recomendado)

[`s2_super_resolution.py`](s2_super_resolution.py) está dividido em células `# %%`, que o VS Code, o Spyder e o Jupyter (via jupytext) reconhecem.

1. Edite o **BLOCO 1 — PARAMETERS**:

   ```python
   TILE = "034026"          # tile da grade BDC
   DATE = "2026-07-12"      # data de início do mosaico de 16 dias
   SOURCE = "bdc16d"        # baixa o mosaico do BDC
   MODE = "fast"            # "fast" | "combined" | "separate"
   ROI_LAT, ROI_LON = None, None   # None = tile inteiro; ou um ponto central
   ROI_SIZE_KM = 10                 # lado do quadrado em volta do ponto
   STEPS = 50                       # passos DDIM (100 = referência)
   ```

2. Execute o **BLOCO 2 — RUN**. Para retomar uma execução interrompida, basta executá-lo de novo.

Um tile inteiro tem cerca de 9 000 janelas; um quadrado de 10 km, cerca de 100. Para testar o modo `combined`, comece com um ROI pequeno.

### Pela linha de comando

Os argumentos substituem o BLOCO 1:

```bash
python s2_super_resolution.py --tile 034026 --date 2026-07-12 --source bdc16d --mode fast --steps 50 --prodes-composite
```

Quadrado de 10 km em torno de um ponto:

```bash
python s2_super_resolution.py --tile 034026 --date 2026-07-12 --source bdc16d --center -3.4653 -62.2159 --box-km 10 --prodes-composite
```

Só verificar o hardware e o tamanho de lote, sem processar:

```bash
python s2_super_resolution.py --tile 034026 --date 2026-07-12 --source bdc16d --probe-only
```

Argumentos mais usados (lista completa em `parse_args()`):

| Argumento | Padrão | Descrição |
|---|---|---|
| `--tile`, `--date` | obrigatórios | Tile BDC e data do mosaico (`YYYY-MM-DD`) |
| `--source` | `auto` | `bdc16d`, `local`, `16d`, `l2a` ou `auto` |
| `--mode` | `fast` | `fast`, `combined` ou `separate` |
| `--center LAT LON`, `--box-km` | tile inteiro, `10` | Recorte quadrado em volta de um ponto |
| `--steps` | `100` | Passos DDIM (50 é cerca de 2× mais rápido) |
| `--deblock-20m` | `1.0` | Sigma gaussiano das bandas de 20 m; `0` mantém os valores do BDC |
| `--uncertainty N` | `0` | Mapa de incerteza com N amostras (N > 3; N vezes mais lento) |
| `--precision` | `auto` | `auto`, `fp32`, `bf16`, `fp16` |
| `--batch-size` | `0` (automático) | Tamanho do lote na GPU |
| `--prodes-composite` | desligado | Gera a composição PRODES |
| `--out-dir` | `work/cube_resolution` | Pasta de resultados |
| `--overwrite` | desligado | Refaz produtos que já existem |

## Fontes de dados

| `SOURCE` | O que faz |
|---|---|
| `bdc16d` | Baixa os COGs do mosaico `S2-16D-2` do [STAC do BDC](https://data.inpe.br/bdc/stac/v1) para `work/cube_bdc_16d/` (arquivos já baixados são reaproveitados). **Recomendado.** |
| `local` | Usa arquivos existentes no padrão `LOCAL_PATTERN` (`SENTINEL-2_MSI_{tile}_{band}_{date}.tif`) |
| `16d` | Lê o mosaico remotamente (`/vsicurl/`), sem baixar |
| `l2a` | Usa a cena Sentinel-2 L2A original (`S2_L2A-1`) menos nublada em ±8 dias |
| `auto` | Tenta `local`, depois `16d`, depois `l2a` |

Se não houver mosaico começando exatamente em `DATE`, o mais próximo dentro de ±8 dias é usado (com aviso no log). Um token de acesso ao BDC, se necessário, pode ser passado por `--bdc-token` ou pela variável de ambiente `BDC_ACCESS_TOKEN`.

## Saídas

Tudo fica em `work/cube_resolution/` (configurável), com nomes curtos no formato:

```
{tile}_{data}_{modo}_{área}[_raw][_s{passos}][_{fonte}].tif
```

Exemplo: `034026_20260712_fast_10km_3.4653S_62.2159W.tif`. Os sufixos só aparecem quando a configuração difere da usual: `_raw` (bandas de 20 m sem suavização), `_s100` (passos DDIM diferentes de 50), `_l2a` (fonte diferente de `bdc16d`).

| Arquivo | Conteúdo |
|---|---|
| `{base}.tif` | Bandas super-resolvidas em 2,5 m |
| `{base}_PRODES.tif` | Composição R=B11 G=B08 B=B04 (Int16; o realce 2–98 % fica nas tags `STRETCH_LO`/`STRETCH_HI`) |
| `{base}_uncertainty.tif` | Mapa de incerteza (2 desvios-padrão), se `UNCERTAINTY > 3` |
| `{base}_B11.tif` | B11 do SEN2SR (só no modo `separate`) |
| `logs/{base}.log` | Log completo da execução |
| `logs/{base}.report.json` | Parâmetros, hardware, arquivos de entrada, saídas e tempos |

Os GeoTIFFs são *tiled*, comprimidos (`zstd` por padrão) e guardam reflectância × 10 000 em Int16, com nodata = 0. As tags registram modelo, modo, área, passos DDIM, precisão, GPU e versão do PyTorch.

## Visualização e comparação

Os dois scripts em [`tools/`](tools) podem ser rodados como célula, no mesmo kernel do notebook (aproveitam `TILE`, `DATE` e `OUT_DIR`), ou pela linha de comando.

**[`tools/preview_prodes.py`](tools/preview_prodes.py)**: visão geral da última execução e o mesmo recorte em 10 m (original) e 2,5 m (super-resolvido).

```bash
python tools/preview_prodes.py work/cube_resolution
```

**[`tools/compare_modes.py`](tools/compare_modes.py)**: compara o original de 10 m com todos os produtos `fast`/`combined` da mesma área, com o mesmo realce em todos os painéis (visível, PRODES e B11 isolada), e imprime métricas:

| Métrica | Significado |
|---|---|
| vis/NIR MAE | Diferença média entre a média 4×4 do produto e a banda original (fidelidade radiométrica) |
| B11 bias / MAE | O mesmo teste para a B11 |
| B11/B08 detail | Energia de detalhe fino da B11 relativa à B08; muito acima do original indica detalhe inventado na B11 |
| grid | ~1,0 = sem padrão de blocos de 20 m; acima de 1,3 = "escadinha" visível |

```bash
python tools/compare_modes.py 034026 2026-07-12 work/cube_resolution
```

## Desempenho e hardware

- `PRECISION = "auto"` testa bf16 contra fp32 e só usa bf16 se o erro for desprezível (≤ 0,003 de reflectância) e o ganho de velocidade for de pelo menos 15 %. No modo `combined`, o padrão é fp32 + TF32.
- `BATCH_SIZE = 0` mede o maior lote que cabe em ~85 % da memória da GPU (80 % em MIG).
- `PROBE_ONLY = True` (ou `--probe-only`) roda só a detecção de hardware e o ajuste de lote.
- `USE_ENV_TORCH = True` força o PyTorch do ambiente conda a ter prioridade sobre `~/.local`. Ao mudar, reinicie o kernel.
- Arquivos temporários (`*.partial*`, `*.progress.json`, `*.dl`) são apagados ao fim de uma execução bem-sucedida.

## Estrutura do repositório

```
s2-super-resolution/
├── s2_super_resolution.py   # Pipeline: parâmetros (bloco 1) e execução (bloco 2); também roda como CLI
├── tools/
│   ├── preview_prodes.py    # Pré-visualização da composição PRODES
│   └── compare_modes.py     # Comparação original × fast × combined, com métricas
├── requirements.txt
├── LICENSE
└── README.md
```

A pasta `work/` é criada na primeira execução e está no `.gitignore`:

```
work/
├── cube_bdc_16d/            # Mosaicos BDC baixados
├── model/                   # Pesos: LDSR_S2, LDSRS2-SEN2SR, SEN2SR_main
└── cube_resolution/         # Resultados
    └── logs/                # .log e .report.json de cada execução
```

## Referências

- **LDSR-S2 / opensr-model** (ESA OpenSR): https://github.com/ESAOpenSR/opensr-model, pesos em https://huggingface.co/simon-donike/RS-SR-LTDF
- **SEN2SR** (tacofoundation): https://huggingface.co/tacofoundation/sen2sr
- **LDSRS2-SEN2SR**: https://huggingface.co/tacofoundation/RS-SR-LTDF
- **Brazil Data Cube**, coleção Sentinel-2 16 dias (`S2-16D-2`): https://data.inpe.br/bdc/stac/v1
- **PRODES** (INPE), monitoramento do desmatamento por satélite: http://terrabrasilis.dpi.inpe.br

## Licença

Distribuído sob a [GNU General Public License v3.0](LICENSE). Os modelos e dados de terceiros (OpenSR, SEN2SR, BDC) seguem as licenças dos respectivos projetos.
