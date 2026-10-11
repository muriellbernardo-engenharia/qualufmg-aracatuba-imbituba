# -*- coding: utf-8 -*-
"""
=============================================================================
ANEXO I — SIMULAÇÃO ESTENDIDA DO QUAL-UFMG ATÉ 1.500 m A JUSANTE
Rio Araçatuba — ETE Residencial Imbituba
=============================================================================
Complementa o Capítulo 8 com simulação estendida, gerando 6 figuras.
Autor: Eng.ª Muriell Bernardo Macedo (CREA-SC 222578-3)
Licença: MIT
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dados.parametros_capitulo8 import (
    OD_SAT, PROF, PONTOS, CEN_A, CEN_B, EFLUENTE, RIO_M1,
    COEF, PADROES, F_DBO,
)

warnings.filterwarnings("ignore")

OUTPUT = Path(__file__).parent / "output_anexo_I"
OUTPUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.titlepad": 12,
    "axes.edgecolor": "#2B2B2B", "axes.linewidth": 1.0,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.7, "grid.linestyle": "-",
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "legend.fontsize": 9,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.dpi": 120,
})

COR_A = "#0D4A70"
COR_B = "#C1440E"
COR_PADRAO = "#8B2635"
COR_SAT = "#4A7C59"
COR_P0 = "#E5A54B"


def fn_(OD, k):
    return OD / (OD + k) if (OD + k) > 0 else 0.0


def ode(x, y, U, K2, coef, prof):
    L, OD, Norg, NNH3, NNO2, NNO3, Porg, Pinorg, Coli = y
    f = fn_(max(OD, 0), coef["k_nitr"])
    return [
        -(coef["Kd"] + coef["Ks"]) * L / U,
        (K2 * (OD_SAT - OD) - coef["Kd"] * L
         - coef["O2Namon"] * NNH3 * coef["Kan"] * f
         - coef["O2Nnitri"] * NNO2 * coef["Knn"] * f) / U,
        -(coef["Kso"] + coef["Koa"]) * Norg / U,
        (coef["Koa"] * Norg - coef["Kan"] * NNH3 * f
         + coef["S_NH3"] / prof) / U,
        (coef["Kan"] * NNH3 * f - coef["Knn"] * NNO2 * f) / U,
        coef["Knn"] * NNO2 * f / U,
        -(coef["Kspo"] + coef["Koi"]) * Porg / U,
        (coef["Koi"] * Porg - coef["Kspi"] * Pinorg) / U,
        -coef["Kb"] * Coli / U,
    ]


def resolver(cen, xf):
    Qr, Qe = cen["Q_rio"], EFLUENTE["Q"]
    Qt = Qr + Qe
    C0 = {k: (Qr * RIO_M1[k] + Qe * EFLUENTE[k]) / Qt for k in RIO_M1}
    U = cen["V"] * 86400.0
    y0 = [C0["DBO5"] * F_DBO, C0["OD"], C0["N_org"], C0["N_NH3"],
          C0["N_NO2"], C0["N_NO3"], C0["P_org"], C0["P_inorg"], C0["Coli"]]
    s = solve_ivp(lambda x, y: ode(x, y, U, cen["K2"], COEF, PROF),
                  (0, xf), y0, method="RK45",
                  rtol=1e-6, atol=1e-9, dense_output=True)
    xg = np.linspace(0, xf, 1500)
    Y = s.sol(xg)
    L, OD, Norg, NNH3, NNO2, NNO3, Porg, Pinorg, Coli = Y
    return {
        "x": xg,
        "DBO5": L / F_DBO,
        "OD": OD,
        "N_org": Norg,
        "N_NH3": NNH3,
        "N_NO2": NNO2,
        "N_NO3": NNO3,
        "P_org": Porg,
        "P_inorg": Pinorg,
        "P_total": Porg + Pinorg,
        "Coli": Coli,
        "Q_tot": Qt,
        "FD": Qr / Qe,
    }


def salvar(fig, nome):
    p = OUTPUT / nome
    fig.savefig(p, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[OK] {p.name}")
    plt.close(fig)


print("Simulando Cenário A (realista) até 1.500 m...")
resA = resolver(CEN_A, 1500)

print("Simulando Cenário B (Q90) até 1.500 m...")
resB = resolver(CEN_B, 1500)


# =============================================================================
# FIGURA I.1 — PERFIL DE OD
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(resA["x"], resA["OD"], color=COR_A, lw=2.5,
        label="Cenário A — Q = 2,973 m³/s")
ax.plot(resB["x"], resB["OD"], color=COR_B, lw=2.5, ls="--",
        label="Cenário B — Q90 = 0,745 m³/s")

ax.axhline(OD_SAT, color=COR_SAT, lw=1.6, ls=":",
           label=f"OD de saturação ({OD_SAT} mg/L)")
ax.axhline(PADROES["OD"], color=COR_PADRAO, lw=1.8, ls="-.",
           label=f"Padrão Classe 2 (≥ {PADROES['OD']} mg/L)")
ax.axvline(0, color=COR_P0, lw=2, ls="--", alpha=0.85,
           label="P0 (lançamento)")

for x in [310, 500, 1000]:
    ax.axvline(x, color="#CCCCCC", lw=0.8, ls=":", alpha=0.7)
    ax.text(x, 9.8, f"{x} m", ha="center", fontsize=8.5, color="#666")

ax.set_xlim(-20, 1500)
ax.set_ylim(5.5, 10.0)
ax.set_xlabel("Distância a partir de P0 (m)")
ax.set_ylabel("Oxigênio dissolvido (mg/L)")
ax.set_title("Anexo I — Perfil de OD — Simulação Estendida até 1.500 m\n"
             "Rio Araçatuba — Cenário Crítico (Q90)",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
salvar(fig, "fig_anexo_I_OD_1500m.png")


# =============================================================================
# FIGURA I.2 — PERFIL DE DBO₅
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(resA["x"], resA["DBO5"], color=COR_A, lw=2.5,
        label="Cenário A — Q = 2,973 m³/s")
ax.plot(resB["x"], resB["DBO5"], color=COR_B, lw=2.5, ls="--",
        label="Cenário B — Q90 = 0,745 m³/s")

ax.axhline(PADROES["DBO5"], color=COR_PADRAO, lw=1.8, ls="-.",
           label=f"Padrão Classe 2 (≤ {PADROES['DBO5']} mg/L)")
ax.axvline(0, color=COR_P0, lw=2, ls="--", alpha=0.85,
           label="P0 (lançamento)")

ax.set_xlim(-20, 1500)
ax.set_ylim(20, 21)
ax.set_xlabel("Distância a partir de P0 (m)")
ax.set_ylabel("DBO₅ (mg/L)")
ax.set_title("Anexo I — Perfil de DBO₅ — Simulação Estendida até 1.500 m\n"
             "Rio Araçatuba — Condição pré-existente acima do padrão",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
salvar(fig, "fig_anexo_I_DBO5_1500m.png")


# =============================================================================
# FIGURA I.3 — N-AMONIACAL
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(resA["x"], resA["N_NH3"], color=COR_A, lw=2.5,
        label="Cenário A — Q = 2,973 m³/s")
ax.plot(resB["x"], resB["N_NH3"], color=COR_B, lw=2.5, ls="--",
        label="Cenário B — Q90 = 0,745 m³/s")

ax.axhline(PADROES["N_NH3"], color=COR_PADRAO, lw=1.8, ls="-.",
           label=f"Padrão Classe 2 (≤ {PADROES['N_NH3']} mg/L)")
ax.axvline(0, color=COR_P0, lw=2, ls="--", alpha=0.85,
           label="P0 (lançamento)")

ax.set_xlim(-20, 1500)
ax.set_ylim(0, 2.2)
ax.set_xlabel("Distância a partir de P0 (m)")
ax.set_ylabel("N-amoniacal (mg/L)")
ax.set_title("Anexo I — Perfil de N-amoniacal — Simulação Estendida até 1.500 m\n"
             "Rio Araçatuba — Cenário Crítico (Q90)",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
salvar(fig, "fig_anexo_I_NH3_1500m.png")


# =============================================================================
# FIGURA I.4 — N-NITRATO
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(resA["x"], resA["N_NO3"], color=COR_A, lw=2.5,
        label="Cenário A — Q = 2,973 m³/s")
ax.plot(resB["x"], resB["N_NO3"], color=COR_B, lw=2.5, ls="--",
        label="Cenário B — Q90 = 0,745 m³/s")

ax.axhline(PADROES["N_NO3"], color=COR_PADRAO, lw=1.8, ls="-.",
           label=f"Padrão Classe 2 (≤ {PADROES['N_NO3']} mg/L)")
ax.axvline(0, color=COR_P0, lw=2, ls="--", alpha=0.85,
           label="P0 (lançamento)")

ax.set_xlim(-20, 1500)
ax.set_ylim(0, 11)
ax.set_xlabel("Distância a partir de P0 (m)")
ax.set_ylabel("N-nitrato (mg/L)")
ax.set_title("Anexo I — Perfil de N-nitrato — Simulação Estendida até 1.500 m\n"
             "Rio Araçatuba — Cenário Crítico (Q90)",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
salvar(fig, "fig_anexo_I_NO3_1500m.png")


# =============================================================================
# FIGURA I.5 — FÓSFORO TOTAL
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(resA["x"], resA["P_total"], color=COR_A, lw=2.5,
        label="Cenário A — Q = 2,973 m³/s")
ax.plot(resB["x"], resB["P_total"], color=COR_B, lw=2.5, ls="--",
        label="Cenário B — Q90 = 0,745 m³/s")

ax.axhline(PADROES["P_total"], color=COR_PADRAO, lw=1.8, ls="-.",
           label=f"Padrão Classe 2 (≤ {PADROES['P_total']} mg/L)")
ax.axvline(0, color=COR_P0, lw=2, ls="--", alpha=0.85,
           label="P0 (lançamento)")

ax.set_xlim(-20, 1500)
ax.set_ylim(0, 0.15)
ax.set_xlabel("Distância a partir de P0 (m)")
ax.set_ylabel("Fósforo total (mg/L)")
ax.set_title("Anexo I — Perfil de Fósforo Total — Simulação Estendida até 1.500 m\n"
             "Rio Araçatuba — Cenário Crítico (Q90)",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
salvar(fig, "fig_anexo_I_Ptotal_1500m.png")


# =============================================================================
# FIGURA I.6 — COLIFORMES
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

ax.semilogy(resA["x"], resA["Coli"], color=COR_A, lw=2.5,
            label="Cenário A — Q = 2,973 m³/s")
ax.semilogy(resB["x"], resB["Coli"], color=COR_B, lw=2.5, ls="--",
            label="Cenário B — Q90 = 0,745 m³/s")

ax.axhline(PADROES["Coli"], color=COR_PADRAO, lw=1.8, ls="-.",
           label=f"Padrão Classe 2 (≤ {int(PADROES['Coli'])} NMP/100 mL)")
ax.axvline(0, color=COR_P0, lw=2, ls="--", alpha=0.85,
           label="P0 (lançamento)")

ax.set_xlim(-20, 1500)
ax.set_ylim(50, 200)
ax.set_xlabel("Distância a partir de P0 (m)")
ax.set_ylabel("Coliformes termotolerantes (NMP/100 mL)")
ax.set_title("Anexo I — Perfil de Coliformes — Simulação Estendida até 1.500 m\n"
             "Rio Araçatuba — Cenário Crítico (Q90)",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
salvar(fig, "fig_anexo_I_coliformes_1500m.png")


print("\n" + "=" * 66)
print("  SIMULAÇÃO ESTENDIDA CONCLUÍDA")
print("=" * 66)
print(f"  Diretório: {OUTPUT}")
print(f"  6 figuras (PNG, 400 DPI)")
print("=" * 66)

print("\n  Estatísticas da simulação (até 1.500 m):")
print(f"  OD mínimo (Cenário B):      {resB['OD'].min():.2f} mg/L")
print(f"  OD mínimo (Cenário A):      {resA['OD'].min():.2f} mg/L")
print(f"  DBO₅ em J2 (Cenário B):     {resB['DBO5'][-1]:.2f} mg/L")
print(f"  P-total em J2 (Cenário B):  {resB['P_total'][-1]:.4f} mg/L")
print(f"  N-NH₃ em J2 (Cenário B):    {resB['N_NH3'][-1]:.4f} mg/L")
print("=" * 66)