# Modelagem QUAL-UFMG — Rio Araçatuba

Modelagem matemática de qualidade da água e estudo de autodepuração
para o lançamento do efluente tratado da ETE do Residencial Imbituba
(M91 Incorporadora), situado em Imbituba/SC.

## Contexto

Estudo elaborado em atendimento ao Ofício de Documentação Complementar
nº 40859/2026 (Requerimento nº 102350, processo URB/69838) do IMA/SC.

- **Corpo receptor:** Rio Araçatuba (Classe 2 — CONAMA 357/2005)
- **Vazão de referência:** Q90 = 0,745 m³/s (Portaria IMA nº 192/2025)
- **Modelo:** QUAL-UFMG (Von Sperling, 2007)
- **Implementação:** Python 3.11

## Estrutura

### Scripts de modelagem (raiz)

- `modelagem_qualufmg.py` — script principal (perfis OD, DBO₅, N, P, coliformes)
- `modelagem_perfil_rio.py` — perfil longitudinal e isométrico
- `modelagem_figuras_avancadas.py` — figuras complementares

### Scripts de figuras profissionais (atualizado_grafico/)

- `gerar_todas_figuras.py` — 28 figuras (Cap 7 a 10 + Anexo A2)
- `zona_mistura_avancada.py` — 6 figuras da zona de mistura (2D, 3D, interativo)
- `perfil_rio_pluma.py` — 4 figuras de perfil cinematográfico

### Parâmetros e dados

- `dados/parametros_capitulo8.py` — parâmetros consolidados do estudo
- `requirements.txt` — dependências

### Saídas (atualizado_grafico/)

- `output_capitulo7/` — 5 figuras (batimetria, hidráulica)
- `output_capitulo8/` — 8 figuras (modelagem QUAL-UFMG + pluma 3D)
- `output_capitulo9/` — 5 figuras (emissário + isométrico 3D)
- `output_capitulo10/` — 4 figuras (análise integrada + pluviometria)
- `output_anexo_A2/` — 6 figuras (simulação estendida 1.500 m)
- `output_zona_mistura/` — 5 PNGs + 1 HTML interativo
- `output_perfil_rio/` — 4 figuras (perfil com pluma)

## Como executar

```bash
pip install -r requirements.txt

# Script principal da modelagem
python modelagem_qualufmg.py
python modelagem_perfil_rio.py

# Figuras profissionais
python atualizado_grafico/gerar_todas_figuras.py
python atualizado_grafico/zona_mistura_avancada.py
python atualizado_grafico/perfil_rio_pluma.py
