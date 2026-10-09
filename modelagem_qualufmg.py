# -*- coding: utf-8 -*-
"""
=============================================================================
MODELAGEM QUAL-UFMG — RIO ARAÇATUBA
ETE Residencial Imbituba — M91 Incorporadora
=============================================================================
Script principal: gera 9 figuras técnicas (PNG, 400 DPI) + 3 figuras
interativas (HTML Plotly) + tabelas de resultados (CSV + XLSX).

Autor: Eng.ª Muriell Bernardo Macedo (CREA-SC 222578-3)
Repositório: https://github.com/muriellbernardo-engenharia/qualufmg-aracatuba-imbituba
Licença: MIT
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from dados.parametros_capitulo8 import (
    T_AGUA, OD_SAT, L_TRECHO, LARGURA, PROF, DECLIV,
    PONTOS, CEN_A, CEN_B, EFLUENTE, RIO_M1, COEF, PADROES, F_DBO,
    MEDICOES_CAMPO,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "0.5",
    "figure.dpi": 120, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.axisbelow": True, "axes.edgecolor": "0.35",
})

COLOR_A = "#0D4A70"
COLOR_B = "#B22222"
COLOR_OD_SAT = "#2E8B57"
COLOR_LIM = "#8B0000"
COLOR_P0 = "#FF8C00"


# =============================================================================
# NÚCLEO DO MODELO QUAL-UFMG
# =============================================================================
def fator_nitr(OD, k_nitr):
    return OD / (OD + k_nitr) if (OD + k_nitr) > 0 else 0.0


def qual_ufmg_ode(x, y, U, K2, OD_sat, coef, prof):
    L, OD, N_org, N_NH3, N_NO2, N_NO3, P_org, P_inorg, Coli = y
    fn = fator_nitr(max(OD, 0.0), coef["k_nitr"])

    dOD = (K2 * (OD_sat - OD)
           - coef["Kd"] * L
           - coef["O2Namon"] * N_NH3 * coef["Kan"] * fn
           - coef["O2Nnitri"] * N_NO2 * coef["Knn"] * fn) / U
    dL = -(coef["Kd"] + coef["Ks"]) * L / U
    dN_org = -(coef["Kso"] + coef["Koa"]) * N_org / U
    dN_NH3 = (coef["Koa"] * N_org - coef["Kan"] * N_NH3 * fn
              + coef["S_NH3"] / prof) / U
    dN_NO2 = (coef["Kan"] * N_NH3 * fn - coef["Knn"] * N_NO2 * fn) / U
    dN_NO3 = (coef["Knn"] * N_NO2 * fn) / U
    dP_org = -(coef["Kspo"] + coef["Koi"]) * P_org / U
    dP_inorg = (coef["Koi"] * P_org - coef["Kspi"] * P_inorg) / U
    dColi = -coef["Kb"] * Coli / U

    return [dL, dOD, dN_org, dN_NH3, dN_NO2, dN_NO3, dP_org, dP_inorg, dColi]


def resolver_cenario(cen, efluente, rio, coef):
    Q_rio, Q_eff = cen["Q_rio"], efluente["Q"]
    Q_tot = Q_rio + Q_eff

    def mistura(cr, ce):
        return (Q_rio * cr + Q_eff * ce) / Q_tot

    C0 = {k: mistura(rio[k], efluente[k]) for k in rio}

    U = cen["V"] * 86400.0
    xf = PONTOS["J2"]
    y0 = [C0["DBO5"] * F_DBO, C0["OD"], C0["N_org"], C0["N_NH3"],
          C0["N_NO2"], C0["N_NO3"], C0["P_org"], C0["P_inorg"], C0["Coli"]]

    sol = solve_ivp(
        fun=lambda x, y: qual_ufmg_ode(x, y, U, cen["K2"], OD_SAT, coef, PROF),
        t_span=(0.0, xf), y0=y0, method="RK45",
        rtol=1e-6, atol=1e-9, dense_output=True
    )

    x_grid = np.linspace(0.0, xf, 511)
    Y = sol.sol(x_grid)
    L, OD, N_org, N_NH3, N_NO2, N_NO3, P_org, P_inorg, Coli = Y

    DBO5 = L / F_DBO
    P_total = P_org + P_inorg

    def interp(x, arr):
        return float(np.interp(x, x_grid, arr))

    campos = {"DBO5": DBO5, "OD": OD, "N_org": N_org, "N_NH3": N_NH3,
              "N_NO2": N_NO2, "N_NO3": N_NO3, "P_org": P_org,
              "P_inorg": P_inorg, "P_total": P_total, "Coli": Coli}

    return {
        "x": x_grid, **campos, "C0": C0,
        "Q_tot": Q_tot, "FD": Q_rio / Q_eff,
        "P0": {k: interp(PONTOS["P0"], v) for k, v in campos.items()},
        "J1": {k: interp(PONTOS["J1"], v) for k, v in campos.items()},
        "J2": {k: interp(PONTOS["J2"], v) for k, v in campos.items()},
    }


resA = resolver_cenario(CEN_A, EFLUENTE, RIO_M1, COEF)
resB = resolver_cenario(CEN_B, EFLUENTE, RIO_M1, COEF)


# =============================================================================
# UTILITÁRIOS DE PLOTAGEM
# =============================================================================
def gradient_line(ax, x, y, cmap_name="Blues", lw=2.8, label=None):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = plt.get_cmap(cmap_name)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    if label:
        ax.plot([], [], color=cmap(0.6), lw=lw, label=label)


def gradient_fill(ax, x, y_base, y_curve, cmap_name="Blues", alpha=0.28):
    n = len(x)
    cmap = plt.get_cmap(cmap_name)
    for i in range(n - 1):
        c = cmap(np.linspace(0, 1, n)[i])
        ax.fill_between([x[i], x[i + 1]], [y_base[i], y_base[i + 1]],
                        [y_curve[i], y_curve[i + 1]],
                        color=c, alpha=alpha, linewidth=0)


def add_pontos_monitoramento(ax, ymin, ymax):
    for nome, x in PONTOS.items():
        if x < 0:
            continue
        ax.axvline(x, color="0.55", linestyle=":", lw=0.9, alpha=0.6)
        ax.text(x, ymax - (ymax - ymin) * 0.02, f" {nome}",
                va="top", ha="left", fontsize=8, color="0.4",
                fontweight="bold")


def add_lancamento(ax, ymin, ymax):
    ax.axvline(0, color=COLOR_P0, linestyle="--", lw=1.5, alpha=0.85)
    ax.text(3, ymax - (ymax - ymin) * 0.02, "P0",
            color=COLOR_P0, fontsize=9, fontweight="bold", va="top")


def save_fig(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[OK] {path.name}")
    plt.close(fig)


# =============================================================================
# FIGURA 01 — CURVA DE AUTODEPURAÇÃO (OD + déficit)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8),
                               gridspec_kw={"width_ratios": [1.2, 1]})

gradient_line(ax1, resA["x"], resA["OD"], "Blues", 3.0,
              "Cenário A — Q = 2,973 m³/s")
gradient_line(ax1, resB["x"], resB["OD"], "Reds", 3.0,
              "Cenário B — Q90 = 0,745 m³/s")
ax1.axhline(OD_SAT, color=COLOR_OD_SAT, linestyle="--", lw=1.7,
            label=f"Saturação ({OD_SAT} mg/L)")
ax1.axhline(PADROES["OD"], color=COLOR_LIM, linestyle="-.", lw=1.7,
            label=f"Padrão Classe 2 (≥ {PADROES['OD']} mg/L)")
ax1.axhspan(PADROES["OD"], OD_SAT, color="#e8f5e9", alpha=0.35, zorder=0)
ax1.scatter(list(MEDICOES_CAMPO["OD"].values()),
            [MEDICOES_CAMPO["OD"][k] for k in MEDICOES_CAMPO["OD"]],
            s=110, marker="o", facecolor="white", edgecolor="black",
            linewidth=1.7, zorder=10, label="Medições de campo (C2)")
add_lancamento(ax1, 5.5, 9.8)
add_pontos_monitoramento(ax1, 5.5, 9.8)
ax1.set_xlim(-50, PONTOS["J2"] + 20)
ax1.set_ylim(5.5, 9.8)
ax1.set_xlabel("Distância a partir do lançamento (m)")
ax1.set_ylabel("Oxigênio dissolvido (mg/L)")
ax1.set_title("Perfil de Oxigênio Dissolvido", pad=10)
ax1.legend(loc="lower right", fontsize=9)

D_A = OD_SAT - resA["OD"]
D_B = OD_SAT - resB["OD"]
gradient_line(ax2, resA["x"], D_A, "Blues", 3.0, "Cenário A")
gradient_line(ax2, resB["x"], D_B, "Reds", 3.0, "Cenário B")
ax2.axhline(OD_SAT - PADROES["OD"], color=COLOR_LIM, linestyle="-.",
            lw=1.7, label="Déficit crítico CONAMA")
add_lancamento(ax2, D_B.min() * 0.98, D_B.max() * 1.05)
add_pontos_monitoramento(ax2, D_B.min() * 0.98, D_B.max() * 1.05)
ax2.set_xlim(-50, PONTOS["J2"] + 20)
ax2.set_xlabel("Distância a partir do lançamento (m)")
ax2.set_ylabel("Déficit de OD (mg/L)")
ax2.set_title("Curva de Autodepuração", pad=10)
ax2.legend(loc="upper right", fontsize=9)

fig.suptitle("Autodepuração — Rio Araçatuba | ETE Residencial Imbituba",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, "fig_01_autodepuracao.png")


# =============================================================================
# FIGURA 02 — DBO₅
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
gradient_fill(ax, resA["x"], np.zeros_like(resA["x"]), resA["DBO5"], "Blues")
gradient_fill(ax, resB["x"], np.zeros_like(resB["x"]), resB["DBO5"], "Reds")
gradient_line(ax, resA["x"], resA["DBO5"], "Blues", 3.2,
              "Cenário A (Q = 2,973 m³/s)")
gradient_line(ax, resB["x"], resB["DBO5"], "Reds", 3.2,
              "Cenário B (Q90 = 0,745 m³/s)")
ax.axhline(PADROES["DBO5"], color=COLOR_LIM, linestyle="-.", lw=1.9,
           label=f"Padrão Classe 2 (≤ {PADROES['DBO5']} mg/L)")
ax.scatter(list(PONTOS[k] for k in MEDICOES_CAMPO["DBO5"]),
           list(MEDICOES_CAMPO["DBO5"].values()),
           s=110, marker="o", facecolor="white", edgecolor="black",
           linewidth=1.7, zorder=10, label="Medições laboratoriais (C1)")
add_lancamento(ax, 0, 45)
add_pontos_monitoramento(ax, 0, 45)
ax.set_xlim(-50, PONTOS["J2"] + 20)
ax.set_ylim(0, 46)
ax.set_xlabel("Distância a partir do lançamento (m)")
ax.set_ylabel("DBO₅ (mg/L)")
ax.set_title("Decaimento de DBO₅ ao longo do Rio Araçatuba", pad=10)
ax.legend(loc="upper right", fontsize=9.5)
fig.tight_layout()
save_fig(fig, "fig_02_DBO5.png")


# =============================================================================
# FIGURA 03 — CICLO DO NITROGÊNIO (4 painéis)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
paineis = [
    ("N_org", "Nitrogênio Orgânico"),
    ("N_NH3", "Nitrogênio Amoniacal (NH₃)"),
    ("N_NO2", "Nitrito (NO₂⁻)"),
    ("N_NO3", "Nitrato (NO₃⁻)"),
]
for ax, (key, nome) in zip(axes.flatten(), paineis):
    ax.plot(resA["x"], resA[key], lw=2.5, color=COLOR_A, label="Cenário A")
    ax.fill_between(resA["x"], 0, resA[key], color=COLOR_A, alpha=0.10)
    ax.plot(resB["x"], resB[key], lw=2.5, color=COLOR_B, label="Cenário B",
            linestyle="--")
    ax.fill_between(resB["x"], 0, resB[key], color=COLOR_B, alpha=0.08)
    if key in ("N_NH3", "N_NO2", "N_NO3"):
        ax.axhline(PADROES[key], color=COLOR_LIM, linestyle="-.", lw=1.6,
                   label=f"Padrão C2 (≤ {PADROES[key]})")
    add_lancamento(ax, 0, ax.get_ylim()[1] * 1.1)
    add_pontos_monitoramento(ax, 0, ax.get_ylim()[1] * 1.1)
    ax.set_xlim(-50, PONTOS["J2"] + 20)
    ax.set_ylabel("Concentração (mg/L)")
    ax.set_title(nome, pad=8)
    ax.legend(loc="best", fontsize=8.5)
axes[1, 0].set_xlabel("Distância a partir do lançamento (m)")
axes[1, 1].set_xlabel("Distância a partir do lançamento (m)")
fig.suptitle("Ciclo do Nitrogênio — Rio Araçatuba",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout()
save_fig(fig, "fig_03_nitrogenio.png")


# =============================================================================
# FIGURA 04 — CICLO DO FÓSFORO
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
for ax, (res, cen_nome, cmap_name) in zip(
        axes, [(resA, "Cenário A — Q = 2,973 m³/s", "Blues"),
               (resB, "Cenário B — Q90 = 0,745 m³/s", "Reds")]):
    ax.plot(res["x"], res["P_org"], lw=2.4, color="#7B1FA2",
            label="P-orgânico")
    ax.plot(res["x"], res["P_inorg"], lw=2.4, color="#E91E63",
            label="P-inorgânico", linestyle="--")
    gradient_line(ax, res["x"], res["P_total"], cmap_name, 3.2, "P-total")
    ax.axhline(PADROES["P_total"], color=COLOR_LIM, linestyle="-.",
               lw=1.8, label=f"Padrão C2 (≤ {PADROES['P_total']})")
    add_lancamento(ax, 0, 0.12)
    add_pontos_monitoramento(ax, 0, 0.12)
    ax.set_xlim(-50, PONTOS["J2"] + 20)
    ax.set_ylim(0, 0.12)
    ax.set_xlabel("Distância a partir do lançamento (m)")
    ax.set_ylabel("Concentração (mg/L)")
    ax.set_title(cen_nome, pad=8)
    ax.legend(loc="upper right", fontsize=9)
fig.suptitle("Ciclo do Fósforo — Rio Araçatuba",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, "fig_04_fosforo.png")


# =============================================================================
# FIGURA 05 — COLIFORMES
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogy(resA["x"], resA["Coli"], lw=3.0, color=COLOR_A,
            label="Cenário A")
ax.semilogy(resB["x"], resB["Coli"], lw=3.0, color=COLOR_B,
            label="Cenário B", linestyle="--")
ax.axhline(PADROES["Coli"], color=COLOR_LIM, linestyle="-.",
           lw=1.9, label=f"Padrão C2 (≤ {int(PADROES['Coli'])})")
ax.axhspan(50, PADROES["Coli"], color="#e8f5e9", alpha=0.35, zorder=0)
add_lancamento(ax, 50, 4000)
add_pontos_monitoramento(ax, 50, 4000)
ax.set_xlim(-50, PONTOS["J2"] + 20)
ax.set_ylim(50, 4000)
ax.set_xlabel("Distância a partir do lançamento (m)")
ax.set_ylabel("Coliformes termotolerantes (NMP/100 mL)")
ax.set_title("Decaimento de Coliformes Termotolerantes", pad=10)
ax.legend(loc="upper right", fontsize=9.5)
fig.tight_layout()
save_fig(fig, "fig_05_coliformes.png")


# =============================================================================
# FIGURA 06 — PAINEL COMPARATIVO
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
params = [
    ("OD",      "Oxigênio Dissolvido (mg/L)",  PADROES["OD"],      "≥"),
    ("DBO5",    "DBO₅ (mg/L)",                 PADROES["DBO5"],    "≤"),
    ("N_NH3",   "N-amoniacal (mg/L)",          PADROES["N_NH3"],   "≤"),
    ("N_NO3",   "N-nitrato (mg/L)",            PADROES["N_NO3"],   "≤"),
    ("P_total", "Fósforo total (mg/L)",        PADROES["P_total"], "≤"),
    ("Coli",    "Coliformes (NMP/100mL)",      PADROES["Coli"],    "≤"),
]
for ax, (key, ylabel, padrao, op) in zip(axes.flatten(), params):
    ax.plot(resA["x"], resA[key], lw=2.4, color=COLOR_A, label="Cenário A")
    ax.plot(resB["x"], resB[key], lw=2.4, color=COLOR_B, label="Cenário B",
            linestyle="--")
    ax.axhline(padrao, color=COLOR_LIM, linestyle="-.", lw=1.5,
               label=f"Padrão C2 ({op} {padrao})")
    ax.axvline(0, color=COLOR_P0, linestyle="--", lw=1.2, alpha=0.75)
    ax.set_title(ylabel, fontsize=10.5)
    ax.set_xlabel("Distância (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlim(-50, PONTOS["J2"] + 20)
    if key == "Coli":
        ax.set_yscale("log")
fig.suptitle("Painel Comparativo entre Cenários — Rio Araçatuba",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "fig_06_painel_comparativo.png")


# =============================================================================
# FIGURA 07 — ZONA DE MISTURA (PLANTA)
# =============================================================================
h, S = PROF, DECLIV
u_star = np.sqrt(9.81 * h * S)
eps_y = 0.15 * h * u_star
U_B = CEN_B["V"]
sigma_y0 = 0.5

x_plume = np.linspace(-15, 150, 600)
y_plume = np.linspace(-LARGURA / 2, LARGURA / 2, 220)
X, Y = np.meshgrid(x_plume, y_plume)

sigma_y = np.sqrt(sigma_y0**2 + 2 * eps_y * np.maximum(x_plume, 0) / U_B)
P_center = np.interp(np.maximum(x_plume, 0), resB["x"], resB["P_total"])
P_bg = resB["P0"]["P_total"]

C_field = np.zeros_like(X)
for j, (sig, cval) in enumerate(zip(sigma_y, P_center)):
    lateral = np.exp(-Y[:, j]**2 / (2 * sig**2))
    lateral_norm = lateral / max(lateral.max(), 1e-12)
    C_field[:, j] = P_bg + (cval - P_bg) * lateral_norm

mask = np.abs(Y) > (LARGURA / 2)
C_field = np.ma.array(C_field, mask=mask)

fig, ax = plt.subplots(figsize=(14, 6))
cmap_pluma = LinearSegmentedColormap.from_list(
    "pluma", ["#e8f4f8", "#a8d8ea", "#7ec9e0", "#f4c471", "#ef7a3d",
              "#c0392b"], N=256)
im = ax.pcolormesh(X, Y, C_field, cmap=cmap_pluma, shading="auto",
                   vmin=0.078, vmax=0.092)
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Fósforo total (mg/L)", fontsize=11)
cs = ax.contour(X, Y, C_field, levels=[PADROES["P_total"]],
                colors="black", linewidths=1.9, linestyles="--")
ax.clabel(cs, fmt="%.2f mg/L", fontsize=10)
ax.axhline(+LARGURA / 2, color="0.30", lw=1.5)
ax.axhline(-LARGURA / 2, color="0.30", lw=1.5)
ax.plot(0, 0, marker="*", markersize=24, color="#FFD700",
        markeredgecolor="black", markeredgewidth=1.5, zorder=10,
        label="Ponto de lançamento (P0)")
ax.axhspan(LARGURA / 2, LARGURA / 2 + 0.9, color="#c8c8c8", alpha=0.75)
ax.axhspan(-LARGURA / 2 - 0.9, -LARGURA / 2, color="#c8c8c8", alpha=0.75)
ax.annotate("", xy=(130, 0), xytext=(85, 0),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#0D4A70"))
ax.text(105, 0.75, "Escoamento", fontsize=10, color="#0D4A70",
        fontweight="bold", ha="center")
for x_sec, sinal in [(PONTOS["J1"], "J1"), (PONTOS["J2"], "J2")]:
    ax.axvline(x_sec, color="0.4", lw=0.8, linestyle=":", alpha=0.5)
    ax.text(x_sec, -LARGURA / 2 - 0.65, sinal, ha="center",
            fontsize=8.5, color="0.4", fontweight="bold")
ax.set_xlim(-15, 150)
ax.set_ylim(-LARGURA / 2 - 0.9, LARGURA / 2 + 0.9)
ax.set_xlabel("Distância a partir do lançamento (m)")
ax.set_ylabel("Posição lateral no canal (m)")
ax.set_title("Zona de Mistura — Vista em Planta "
             "(Fósforo Total, Cenário B, Q90)", pad=12)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
save_fig(fig, "fig_07_zona_mistura.png")


# =============================================================================
# FIGURA 08 — SEÇÕES TRANSVERSAIS
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6.5))
secoes = [1, 5, 15, 30, 60]
cmap_secoes = plt.get_cmap("plasma")
cores = [cmap_secoes(i / (len(secoes) - 1)) for i in range(len(secoes))]
y_sec = np.linspace(-LARGURA / 2, LARGURA / 2, 400)

for x_sec, cor in zip(secoes, cores):
    sig = np.sqrt(sigma_y0**2 + 2 * eps_y * x_sec / U_B)
    c_center = float(np.interp(x_sec, resB["x"], resB["P_total"]))
    lateral = np.exp(-y_sec**2 / (2 * sig**2))
    lateral_norm = lateral / lateral.max()
    c_profile = P_bg + (c_center - P_bg) * lateral_norm
    ax.plot(y_sec, c_profile, lw=2.6, color=cor,
            label=f"x = {x_sec} m  (σ_y = {sig:.2f} m)")

ax.axhline(PADROES["P_total"], color=COLOR_LIM, linestyle="-.",
           lw=1.9, label=f"Padrão C2 (≤ {PADROES['P_total']})")
ax.set_xlim(-LARGURA / 2, LARGURA / 2)
ax.set_ylim(0.078, 0.093)
ax.set_xlabel("Posição lateral no canal (m)")
ax.set_ylabel("Fósforo total (mg/L)")
ax.set_title("Evolução Transversal da Pluma de Fósforo Total\n"
             "Abertura lateral em função da distância — Cenário B", pad=10)
ax.legend(loc="upper right", fontsize=9.5, ncol=2)
fig.tight_layout()
save_fig(fig, "fig_08_secoes_transversais.png")


# =============================================================================
# FIGURA 09 — DASHBOARD RESUMO
# =============================================================================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

# Painel 1 — Conformidade em J2
ax = fig.add_subplot(gs[0, :2])
nomes = ["OD", "DBO₅", "N-NH₃", "N-NO₂", "N-NO₃", "P-total", "Colif."]
lim_norm = [PADROES["OD"], PADROES["DBO5"], PADROES["N_NH3"],
            PADROES["N_NO2"], PADROES["N_NO3"], PADROES["P_total"],
            PADROES["Coli"] / 1000.0]
vA = [resA["J2"][k] for k in ["OD", "DBO5", "N_NH3", "N_NO2", "N_NO3",
                              "P_total"]] + [resA["J2"]["Coli"] / 1000.0]
vB = [resB["J2"][k] for k in ["OD", "DBO5", "N_NH3", "N_NO2", "N_NO3",
                              "P_total"]] + [resB["J2"]["Coli"] / 1000.0]
rA = [v / l for v, l in zip(vA, lim_norm)]
rB = [v / l for v, l in zip(vB, lim_norm)]
y_pos = np.arange(len(nomes))
ax.barh(y_pos - 0.20, rA, height=0.38, color=COLOR_A, alpha=0.9,
        label="Cenário A")
ax.barh(y_pos + 0.20, rB, height=0.38, color=COLOR_B, alpha=0.9,
        label="Cenário B")
ax.axvline(1.0, color=COLOR_LIM, linestyle="--", lw=2.0,
           label="Limite (= 1,0)")
ax.axvspan(0, 1.0, color="#c8e6c9", alpha=0.25)
ax.axvspan(1.0, max(max(rA), max(rB)) + 0.4, color="#ffcdd2", alpha=0.20)
ax.set_yticks(y_pos); ax.set_yticklabels(nomes)
ax.set_xlabel("Razão (concentração em J2 / padrão CONAMA 357/2005)")
ax.set_title("Conformidade em J2", pad=10)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, max(max(rA), max(rB)) + 0.4)

# Painel 2 — Fator de diluição
ax = fig.add_subplot(gs[0, 2])
ax.bar(["A", "B"], [resA["FD"], resB["FD"]],
       color=[COLOR_A, COLOR_B], alpha=0.9, width=0.55)
for i, fd in enumerate([resA["FD"], resB["FD"]]):
    ax.text(i, fd * 1.04, f"{fd:.0f}:1", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylabel("Q_rio / Q_efluente")
ax.set_title("Fator de Diluição", pad=10)
ax.set_ylim(0, resA["FD"] * 1.20)

# Painel 3 — Especiação N
ax = fig.add_subplot(gs[1, 0])
valores = [resB["J2"]["N_org"], resB["J2"]["N_NH3"],
           resB["J2"]["N_NO2"], resB["J2"]["N_NO3"]]
cores = ["#6D4C41", "#2E7D32", "#EF6C00", "#1565C0"]
b = ax.bar(["N-org", "N-NH₃", "N-NO₂", "N-NO₃"], valores,
           color=cores, alpha=0.9, width=0.6)
for rect, v in zip(b, valores):
    ax.text(rect.get_x() + rect.get_width() / 2, v * 1.05,
            f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_ylabel("Concentração em J2 (mg/L)")
ax.set_title("Especiação do N (Cenário B)", pad=10)

# Painel 4 — Especiação P
ax = fig.add_subplot(gs[1, 1])
valores = [resB["J2"]["P_org"], resB["J2"]["P_inorg"], resB["J2"]["P_total"]]
b = ax.bar(["P-org", "P-inorg", "P-total"], valores,
           color=["#7B1FA2", "#E91E63", "#B22222"], alpha=0.9, width=0.6)
for rect, v in zip(b, valores):
    ax.text(rect.get_x() + rect.get_width() / 2, v * 1.05,
            f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
ax.axhline(PADROES["P_total"], color=COLOR_LIM, linestyle="-.",
           lw=1.5, label=f"Padrão C2 (≤ {PADROES['P_total']})")
ax.set_ylabel("Concentração em J2 (mg/L)")
ax.set_title("Especiação do P (Cenário B)", pad=10)
ax.legend(fontsize=8)

# Painel 5 — OD vs DBO5
ax = fig.add_subplot(gs[1, 2])
x = np.arange(2)
valsA = [resA["J2"]["OD"], resA["J2"]["DBO5"]]
valsB = [resB["J2"]["OD"], resB["J2"]["DBO5"]]
ax.bar(x - 0.18, valsA, 0.34, color=COLOR_A, alpha=0.9, label="Cenário A")
ax.bar(x + 0.18, valsB, 0.34, color=COLOR_B, alpha=0.9, label="Cenário B")
for i, (a, b) in enumerate(zip(valsA, valsB)):
    ax.text(i - 0.18, a + 0.4, f"{a:.2f}", ha="center", fontsize=8.5)
    ax.text(i + 0.18, b + 0.4, f"{b:.2f}", ha="center", fontsize=8.5)
ax.axhline(5.0, color="black", linestyle="--", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(["OD", "DBO₅"])
ax.set_ylabel("Concentração em J2 (mg/L)")
ax.set_title("Parâmetros Críticos em J2", pad=10)
ax.legend(fontsize=8)

fig.suptitle("Dashboard Resumo — Modelagem QUAL-UFMG\n"
             "Rio Araçatuba | ETE Residencial Imbituba",
             fontsize=13, fontweight="bold", y=0.995)
save_fig(fig, "fig_09_dashboard.png")


# =============================================================================
# EXPORTAÇÃO DE TABELAS
# =============================================================================
cols = ["x", "DBO5", "OD", "N_org", "N_NH3", "N_NO2", "N_NO3",
        "P_org", "P_inorg", "P_total", "Coli"]
df_A = pd.DataFrame({c: resA[c] for c in cols})
df_B = pd.DataFrame({c: resB[c] for c in cols})
df_A.to_csv(OUTPUT_DIR / "resultados_cenario_A.csv", index=False,
            float_format="%.6f")
df_B.to_csv(OUTPUT_DIR / "resultados_cenario_B.csv", index=False,
            float_format="%.6f")

with pd.ExcelWriter(OUTPUT_DIR / "resultados_qualufmg.xlsx",
                    engine="openpyxl") as xls:
    df_A.to_excel(xls, sheet_name="Cenario A completo", index=False)
    df_B.to_excel(xls, sheet_name="Cenario B completo", index=False)
    linhas = []
    for cen, res in [("A", resA), ("B", resB)]:
        for pt in ["P0", "J1", "J2"]:
            linhas.append({"Cenario": cen, "Ponto": pt, **res[pt]})
    pd.DataFrame(linhas).to_excel(xls, sheet_name="Consolidado",
                                  index=False)
print("[OK] resultados_qualufmg.xlsx")


# =============================================================================
# FIGURAS INTERATIVAS (HTML)
# =============================================================================
pio.templates["tecnico"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Arial, sans-serif", size=12, color="#222"),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False),
    margin=dict(l=70, r=40, t=80, b=60),
    hovermode="x unified",
))
pio.templates.default = "tecnico"


def html_sag():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resA["x"], y=resA["OD"], name="Cenário A",
                             line=dict(color="#0D4A70", width=3),
                             hovertemplate="x=%{x:.1f} m<br>OD=%{y:.3f} mg/L<extra></extra>"))
    fig.add_trace(go.Scatter(x=resB["x"], y=resB["OD"], name="Cenário B (Q90)",
                             line=dict(color="#B22222", width=3, dash="dash"),
                             hovertemplate="x=%{x:.1f} m<br>OD=%{y:.3f} mg/L<extra></extra>"))
    fig.add_hline(y=OD_SAT, line=dict(color="#2E8B57", dash="dot", width=2),
                  annotation_text=f"Saturação ({OD_SAT} mg/L)")
    fig.add_hline(y=PADROES["OD"], line=dict(color="#8B0000",
                  dash="dashdot", width=2),
                  annotation_text="Padrão Classe 2 (≥ 5,0 mg/L)")
    for nome, x in PONTOS.items():
        if x >= 0:
            fig.add_vline(x=x, line=dict(color="#888", width=1,
                          dash="dot"), annotation_text=nome)
    fig.update_layout(
        title=dict(text="Curva de Autodepuração — Oxigênio Dissolvido",
                   x=0.5, xanchor="center"),
        xaxis_title="Distância a partir do lançamento (m)",
        yaxis_title="Oxigênio dissolvido (mg/L)", height=550)
    fig.write_html(OUTPUT_DIR / "interativo_01_OD.html",
                   include_plotlyjs="cdn")
    print("[OK] interativo_01_OD.html")


def html_painel():
    fig = make_subplots(rows=3, cols=2,
        subplot_titles=("Oxigênio Dissolvido (mg/L)", "DBO₅ (mg/L)",
                        "N-amoniacal (mg/L)", "N-nitrato (mg/L)",
                        "Fósforo total (mg/L)", "Coliformes (NMP/100 mL)"),
        vertical_spacing=0.10, horizontal_spacing=0.10)
    series = [("OD", 1, 1, PADROES["OD"], "≥"),
              ("DBO5", 1, 2, PADROES["DBO5"], "≤"),
              ("N_NH3", 2, 1, PADROES["N_NH3"], "≤"),
              ("N_NO3", 2, 2, PADROES["N_NO3"], "≤"),
              ("P_total", 3, 1, PADROES["P_total"], "≤"),
              ("Coli", 3, 2, PADROES["Coli"], "≤")]
    for key, r, c, padrao, op in series:
        fig.add_trace(go.Scatter(x=resA["x"], y=resA[key], name="Cenário A",
                                 legendgroup="A",
                                 showlegend=(r == 1 and c == 1),
                                 line=dict(color="#0D4A70", width=2.4)),
                      row=r, col=c)
        fig.add_trace(go.Scatter(x=resB["x"], y=resB[key], name="Cenário B",
                                 legendgroup="B",
                                 showlegend=(r == 1 and c == 1),
                                 line=dict(color="#B22222", width=2.4,
                                           dash="dash")),
                      row=r, col=c)
        fig.add_hline(y=padrao, line=dict(color="#8B0000",
                      dash="dashdot", width=1.8), row=r, col=c,
                      annotation_text=f"{op} {padrao}")
        fig.add_vline(x=0, line=dict(color="#FF8C00", dash="dash",
                      width=1.4), row=r, col=c)
    fig.update_yaxes(type="log", row=3, col=2)
    fig.update_layout(title=dict(text="Painel Comparativo — Rio Araçatuba",
                                 x=0.5, xanchor="center"),
                      height=900, hovermode="x unified",
                      legend=dict(orientation="h", y=1.05, x=0.5,
                                  xanchor="center"))
    fig.write_html(OUTPUT_DIR / "interativo_02_painel.html",
                   include_plotlyjs="cdn")
    print("[OK] interativo_02_painel.html")


def html_zona():
    fig = go.Figure(data=go.Heatmap(
        x=x_plume, y=y_plume, z=np.ma.filled(C_field, np.nan),
        colorscale=[[0, "#e8f4f8"], [0.2, "#a8d8ea"], [0.4, "#7ec9e0"],
                    [0.6, "#f4c471"], [0.8, "#ef7a3d"], [1, "#c0392b"]],
        zmin=0.078, zmax=0.092,
        colorbar=dict(title="P-total (mg/L)"),
        hovertemplate="x=%{x:.1f} m<br>y=%{y:.2f} m<br>P=%{z:.4f} mg/L<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                             marker=dict(symbol="star", size=22,
                                         color="#FFD700",
                                         line=dict(color="black", width=1.5)),
                             name="Ponto de lançamento"))
    fig.update_layout(title=dict(text="Zona de Mistura — Planta (P-total)",
                                 x=0.5, xanchor="center"),
                      xaxis_title="Distância a partir do lançamento (m)",
                      yaxis_title="Posição lateral (m)", height=550)
    fig.write_html(OUTPUT_DIR / "interativo_03_zona_mistura.html",
                   include_plotlyjs="cdn")
    print("[OK] interativo_03_zona_mistura.html")


html_sag()
html_painel()
html_zona()


# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
print("\n" + "=" * 62)
print("  MODELAGEM CONCLUÍDA")
print("=" * 62)
print(f"  Diretório de saída: {OUTPUT_DIR}")
print(f"  Figuras estáticas:  9 (PNG, 400 DPI)")
print(f"  Figuras interativas: 3 (HTML Plotly)")
print(f"  Tabelas: 2 CSV + 1 XLSX")
print("=" * 62)