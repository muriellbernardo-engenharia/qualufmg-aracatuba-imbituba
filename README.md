# Modelagem QUAL-UFMG — Rio Araçatuba

Modelagem matemática de qualidade da água e estudo de autodepuração para o lançamento do efluente tratado da ETE do Residencial Imbituba (M91 Incorporadora), situado em Imbituba/SC.

## Contexto

Estudo elaborado em atendimento ao Ofício de Documentação Complementar nº 40859/2026 (Requerimento nº 102350, processo URB/69838) do IMA/SC.

- Corpo receptor: Rio Araçatuba (Classe 2 — CONAMA 357/2005)
- Vazão de referência: Q90 = 0,745 m³/s (Portaria IMA nº 192/2025)
- Modelo: QUAL-UFMG (Von Sperling, 2007)
- Implementação: Python 3.11

## Estrutura

modelagem_qualufmg.py        (script principal — 9 figs + 3 HTMLs)
modelagem_perfil_rio.py      (perfil longitudinal + isométrico — 3 figs)
dados/parametros_capitulo8.py (parâmetros consolidados)
requirements.txt

## Como executar

pip install -r requirements.txt
python modelagem_qualufmg.py
python modelagem_perfil_rio.py

## Resultados principais

| Parâmetro | Cenário A | Cenário B (Q90) | Padrão C2 |
|-----------|-----------|-----------------|-----------|
| OD (mg/L) | maior 6,45 | maior 6,39 | maior 5,0 |
| DBO5 (mg/L) | 20,31 | 20,28 | menor 5,0 |
| N-NH3 (mg/L) | 0,054 | 0,055 | menor 2,0 |
| N-NO3 (mg/L) | 0,419 | 0,476 | menor 10,0 |
| P-total (mg/L) | 0,082 | 0,089 | menor 0,10 |
| Coliformes (NMP/100 mL) | 77,1 | 77,2 | menor 1.000 |

Fator de diluição: 319:1 (Cenário B) a 1.270:1 (Cenário A).

## Autoria

Eng.ª Muriell Bernardo Macedo — Engenheira Sanitarista e Ambiental
CREA-SC 222578-3

## Licença

MIT