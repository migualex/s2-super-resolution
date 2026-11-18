# Sentinel-2 Super-Resolution for PRODES Composition

<p align="center">
  <img src="docs/banner.svg" width="640" alt="Sentinel-2 Super-Resolution: 10 m → 2.5 m, PRODES composition" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-GPLv3-blue" alt="GPLv3" />
  <img src="https://img.shields.io/badge/lifecycle-experimental-orange.svg" alt="experimental" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
</p>

Super-resolução de imagens Sentinel-2 de 10 m para 2,5 m com os modelos **LDSR-S2** e **SEN2SR**, aplicada aos mosaicos de 16 dias do **Brazil Data Cube** (`S2-16D-2`). O resultado é a composição **PRODES** (R = B11, G = B08, B = B04) em 2,5 m.

<!-- Exemplo: salve a figura de tools/preview_prodes.py como docs/example.png e descomente.
## Exemplo

<p align="center">
  <img src="docs/example.png" width="90%" alt="Original 10 m × super-resolução 2,5 m (composição PRODES)" />
</p>
-->

## Instalação

Requer Python 3.10+ e, de preferência, uma GPU NVIDIA com CUDA. Instale o [PyTorch](https://pytorch.org/get-started/locally/) para a sua versão de CUDA e depois:

```bash
pip install -r requirements.txt
```

Os pesos dos modelos são baixados automaticamente na primeira execução.

## Uso

Edite os parâmetros no início de [`s2_super_resolution.py`](s2_super_resolution.py) (tile, data, modo e área) e execute as células, ou use a linha de comando:

```bash
python s2_super_resolution.py --tile 034026 --date 2026-07-12 --source bdc16d --prodes-composite
```

| Modo | Descrição |
|---|---|
| `fast` | LDSR-S2 nas bandas de 10 m; B11 estimada a partir delas (padrão, mais rápido) |
| `combined` | Modelo LDSRS2-SEN2SR com 10 bandas (melhor qualidade, mais lento) |
| `separate` | LDSR-S2 com a B11 gerada pelo SEN2SR |

Os resultados são salvos em `work/cube_resolution/`, junto com um log de cada execução.

## Ferramentas

- [`tools/preview_prodes.py`](tools/preview_prodes.py): visualiza a composição PRODES da última execução.
- [`tools/compare_modes.py`](tools/compare_modes.py): compara a imagem original com os produtos de cada modo.

## Referências

- [ESA OpenSR / LDSR-S2](https://github.com/ESAOpenSR/opensr-model)
- [SEN2SR](https://huggingface.co/tacofoundation/sen2sr)
- [Brazil Data Cube](https://data.inpe.br/bdc/stac/v1)

## Licença

Distribuído sob a licença [GNU GPL v3.0](LICENSE).
