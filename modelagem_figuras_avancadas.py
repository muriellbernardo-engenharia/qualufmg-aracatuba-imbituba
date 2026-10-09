# -*- coding: utf-8 -*-
"""
=============================================================================
FIGURAS AVANÇADAS COMPLEMENTARES — MODELAGEM QUAL-UFMG
Rio Araçatuba — ETE Residencial Imbituba
=============================================================================
Gera 9 figuras técnicas avançadas (2D, 3D, gradientes, interativas)
para complementar os Capítulos 8, 9 e 10 do relatório.

Autor: Eng.ª Muriell Bernardo Macedo (CREA-SC 222578-3)
Licença: MIT
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from dados.parametros_capitulo8 import (
    OD_SAT, PROF, DECLIV, PONTOS, CEN_A, CEN_B, EFLUENTE, RIO_M1,
    COEF, PADROES, F_DBO, SECOES, COTAS_TERRENO,
)

warnings.filterwarnings("ignore")

OUTPUT = Path(__file__).parent / "output_extras"
OUTPUT.mkdir(exist_ok=True)

# =============================================================================
# PALETA PROFISSIONAL
# =============================================================================
PAL = {
    "A":       "#0D4A70",   # Cenário A (azul-petróleo)
    "B":       "#C1440E",   # Cenário B (laranja-terra)
    "padrao":  "#8B2635",   # Padrão CONAMA (bordô)
    "sat":     "#4A7C59",   # Saturação OD (verde-musgo)
    "p0":      "#E5A54B",   # Ponto de lançamento (dourado)
    "ok":      "#2E8B57",   # Conforme (verde)
    "nok":     "#B22222",   # Não conforme (vermelho)
    "neutro":  "#7A7A7A",
    "fundo":   "#FAFAFA",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.titlepad": 14,
    "axes.edgecolor": "#2B2B2B", "axes.linewidth": 1.0,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.7, "grid.linestyle": "-",
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "legend.fontsize": 9,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "figure.dpi": 120, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# =============================================================================
# DADOS ADICIONAIS (extraídos do relatório)
# =============================================================================
# Planos do emissário (V1 saída ETE, V2 ponto intermediário ou final, V3 lançamento)
PLANOS = {
    "A": {
        "V1": {"d": 0.00,    "cota": 7.4388, "crown": 6.439},
        "V2": {"d": 406.02,  "cota": 2.8933, "crown": 4.542},
        "V3": {"d": 588.02,  "cota": 4.6918, "crown": 3.692},
        "L":  588.02,
        "aterro_max": 2.649,
        "custo_aterro": 340800,
        "custo_bomb": 214626,
    },
    "B": {
        "V1": {"d": 0.00,    "cota": 7.4388, "crown": 6.439},
        "V2": {"d": 418.73,  "cota": 2.8933, "crown": 4.486},
        "V3": {"d": 588.887, "cota": 4.6918, "crown": 3.692},
        "L":  588.887,
        "aterro_max": 2.592,
        "custo_aterro": 338463,
        "custo_bomb": 214569,
    },
    "C": {
        "V1": {"d": 0.00,    "cota": 7.4388, "crown": 6.439},
        "V2": {"d": 617.127, "cota": 3.9316, "crown": 2.932},
        "L":  617.127,
        "aterro_max": 2.238,
        "custo_aterro": 337112,
        "custo_bomb": 214124,
    },
}

# Precipitação mensal (Estação Campo D'Una — CEMADEN)
PLUV = {
    "Set/25": 58.4, "Out/25": 22.0, "Nov/25": 5.2, "Dez/25": 45.4,
    "Jan/26": 79.2, "Fev/26": 94.8, "Mar/26": 45.0, "Abr/26": 28.2,
    "Mai/26": 31.6, "Jun/26": 34.0, "Jul/26": 27.4, "Ago/26": 18.0,
    "Set/26": 12.0,
}
PLUV_MEDIA_HISTORICA = 1450.0  # mm/ano

# Comparação de diâmetros comerciais
DIAMETROS = {
    "DN (mm)":  [75, 100, 150, 200],
    "D int":    [0.072, 0.096, 0.144, 0.192],
    "A (m²)":   [0.004072, 0.007238, 0.016286, 0.028953],
    "V (m/s)":  [1.024, 0.576, 0.256, 0.144],
    "J (m/m)":  [0.01630, 0.00400, 0.00056, 0.00014],
}

# Soluções técnicas (aterro × bombeamento)
SOLUCOES = {
    "A1 (Aterro)":  {"inicial": 340800, "op20": 0,      "total": 340800},
    "B1 (Aterro)":  {"inicial": 338463, "op20": 0,      "total": 338463},
    "C1 (Aterro)":  {"inicial": 337112, "op20": 0,      "total": 337112},
    "A2 (Bomb.)":   {"inicial": 182250, "op20": 32376,  "total": 214626},
    "B2 (Bomb.)":   {"inicial": 182375, "op20": 32194,  "total": 214569},
    "C2 (Bomb.)":   {"inicial": 185875, "op20": 28249,  "total": 214124},
}


# =============================================================================
# MODELO QUAL-UFMG (compacto)
# =============================================================================
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


def resolver(cen):
    from scipy.integrate import solve_ivp
    Qr, Qe = cen["Q_rio"], EFLUENTE["Q"]
    Qt = Qr + Qe
    C0 = {k: (Qr * RIO_M1[k] + Qe * EFLUENTE[k]) / Qt for k in RIO_M1}
    U = cen["V"] * 86400.0
    xf = PONTOS["J2"]
    y0 = [C0["DBO5"] * F_DBO, C0["OD"], C0["N_org"], C0["N_NH3"],
          C0["N_NO2"], C0["N_NO3"], C0["P_org"], C0["P_inorg"], C0["Coli"]]
    s = solve_ivp(lambda x, y: ode(x, y, U, cen["K2"], COEF, PROF),
                  (0, xf), y0, method="RK45", rtol=1e-6, atol=1e-9,
                  dense_output=True)
    xg = np.linspace(0, xf, 511)
    Y = s.sol(xg)
    L, OD, Norg, NNH3, NNO2, NNO3, Porg, Pinorg, Coli = Y
    return {"x": xg, "DBO5": L / F_DBO, "OD": OD,
            "N_org": Norg, "N_NH3": NNH3, "N_NO2": NNO2, "N_NO3": NNO3,
            "P_org": Porg, "P_inorg": Pinorg, "P_total": Porg + Pinorg,
            "Coli": Coli,
            "FD": Qr / Qe}


resA = resolver(CEN_A)
resB = resolver(CEN_B)


def salvar(fig, nome):
    p = OUTPUT / nome
    fig.savefig(p, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[OK] {p.name}")
    plt.close(fig)


def gradient_line(ax, x, y, cmap, lw=3.0, label=None, alpha=1.0):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap_obj = plt.get_cmap(cmap)
    lc = LineCollection(segs, cmap=cmap_obj, linewidth=lw,
                        alpha=alpha, capstyle="round")
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    if label:
        ax.plot([], [], color=cmap_obj(0.6), lw=lw, label=label)


# =============================================================================
# FIGURA 01 — HEATMAP SEMÁFORO DE CONFORMIDADE
# =============================================================================
# Matriz razão concentração / padrão (valores > 1 = não conforme)
def matriz_razao(res):
    parametros = ["OD", "DBO5", "N_NH3", "N_NO2", "N_NO3", "P_total", "Coli"]
    padroes = [PADROES["OD"], PADROES["DBO5"], PADROES["N_NH3"],
               PADROES["N_NO2"], PADROES["N_NO3"], PADROES["P_total"],
               PADROES["Coli"]]
    pts_nomes = ["P0", "J1", "J2"]
    M = np.zeros((len(parametros), len(pts_nomes)))
    for i, p in enumerate(parametros):
        for j, pt in enumerate(pts_nomes):
            idx = np.argmin(np.abs(res["x"] - PONTOS[pt]))
            v = res[p][idx]
            # OD é inverso (maior é melhor)
            if p == "OD":
                M[i, j] = padroes[i] / v
            else:
                M[i, j] = v / padroes[i]
    return M, parametros, pts_nomes


MA, parametros, pts_nomes = matriz_razao(resA)
MB, _, _ = matriz_razao(resB)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, M, titulo in [(axes[0], MA, "Cenário A"),
                      (axes[1], MB, "Cenário B (Q90)")]:
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=2.5)

    ax.set_xticks(range(len(pts_nomes)))
    ax.set_xticklabels(pts_nomes, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(parametros)))
    ax.set_yticklabels(["OD", "DBO₅", "N-NH₃", "N-NO₂", "N-NO₃",
                        "P-total", "Coliformes"], fontsize=10.5)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            razao = M[i, j]
            if razao <= 0.5:
                txt = "OK"
                cor = "white"
            elif razao <= 1.0:
                txt = f"{razao:.2f}\nOK"
                cor = "black"
            elif razao <= 2.0:
                txt = f"{razao:.2f}\n⚠"
                cor = "white"
            else:
                txt = f"{razao:.1f}×\n✗"
                cor = "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=cor, fontweight="bold")

    ax.set_title(titulo, fontsize=12, pad=10)
    ax.grid(False)

cbar = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.025)
cbar.set_label("Razão (concentração / padrão CONAMA 357/2005)\n"
               "Verde = conforme  |  Vermelho = não conforme",
               fontsize=9.5)
cbar.ax.axhline(1.0, color="black", lw=2)

fig.suptitle("Semáforo de Conformidade Ambiental — Rio Araçatuba",
             fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout()
salvar(fig, "fig_x01_semaforo_conformidade.png")


# =============================================================================
# FIGURA 02 — BALANÇO DE CARGAS (barras empilhadas)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

params_carga = [
    ("DBO₅", RIO_M1["DBO5"], EFLUENTE["DBO5"]),
    ("NTK", RIO_M1["N_NH3"] + RIO_M1["N_org"], EFLUENTE["N_NH3"] + EFLUENTE["N_org"]),
    ("P-total", RIO_M1["P_org"] + RIO_M1["P_inorg"],
     EFLUENTE["P_org"] + EFLUENTE["P_inorg"]),
]

Qr = CEN_B["Q_rio"]  # Q90 = 0,745 m³/s (cenário crítico)
Qe = EFLUENTE["Q"]

for ax, (nome, cr, ce) in zip(axes, params_carga):
    carga_rio = Qr * cr * 86.4    # kg/dia
    carga_eff = Qe * ce * 86.4
    total = carga_rio + carga_eff

    ax.barh([1], [carga_rio], color=PAL["A"], alpha=0.9,
            height=0.5, label="Rio a montante")
    ax.barh([1], [carga_eff], left=[carga_rio], color=PAL["p0"],
            alpha=0.95, height=0.5, label="Efluente ETE")

    ax.text(carga_rio / 2, 1, f"{carga_rio:.1f} kg/d\n({100*carga_rio/total:.2f}%)",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    if carga_eff / total > 0.02:
        ax.text(carga_rio + carga_eff / 2, 1,
                f"{carga_eff:.2f} kg/d",
                ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")

    ax.annotate(f"Efluente:\n{carga_eff:.2f} kg/d ({100*carga_eff/total:.3f}%)",
                xy=(carga_rio + carga_eff/2, 1.3),
                xytext=(carga_rio * 0.75, 1.75),
                fontsize=9.5, color=PAL["B"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=PAL["B"], lw=1.2),
                arrowprops=dict(arrowstyle="->", color=PAL["B"], lw=1.3))

    ax.set_yticks([])
    ax.set_xlim(0, total * 1.15)
    ax.set_xlabel("Carga transportada (kg/dia)", fontsize=10)
    ax.set_title(f"{nome}", fontsize=12, pad=10)
    ax.set_ylim(0.5, 2.0)
    ax.legend(loc="upper right", fontsize=8.5)

fig.suptitle("Balanço de Cargas — Rio a Montante × Efluente Tratado (Q90)\n"
             "Contribuição do empreendimento inferior a 0,4% da carga total",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
salvar(fig, "fig_x02_balanco_cargas.png")


# =============================================================================
# FIGURA 03 — CONFORMIDADE NORMALIZADA AO LONGO DO RIO
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 6.5))

x_full = np.concatenate([[-200], [-100], resB["x"] + 0])

# Valores medidos em M2 e M1 (a montante do P0)
valores_M2 = {"DBO5": 18.36, "N_NH3": 0.05, "N_NO3": 0.40, "P_total": 0.11}
valores_M1 = {"DBO5": 20.40, "N_NH3": 0.05, "N_NO3": 0.40, "P_total": 0.08}

series = [
    ("DBO5",    "DBO₅",       "DBO5",    valores_M2["DBO5"],   valores_M1["DBO5"],   "#B22222"),
    ("N_NH3",   "N-NH₃",      "N_NH3",   valores_M2["N_NH3"],  valores_M1["N_NH3"],  "#2E7D32"),
    ("N_NO3",   "N-NO₃",      "N_NO3",   valores_M2["N_NO3"],  valores_M1["N_NO3"],  "#1565C0"),
    ("P_total", "P-total",    "P_total", valores_M2["P_total"], valores_M1["P_total"], "#7B1FA2"),
]

for key_pad, label, key_res, v_m2, v_m1, cor in series:
    padrao = PADROES[key_pad]
    y_m2 = 100 * v_m2 / padrao
    y_m1 = 100 * v_m1 / padrao
    y_p0 = 100 * resB[key_res][0] / padrao
    y_jus = 100 * resB[key_res] / padrao

    # Linha de M2 → M1 (medido) e M1 → P0 (medido) e P0 → J2 (modelado)
    ax.plot([-200, -100], [y_m2, y_m1], color=cor, lw=2.5, ls="--",
            alpha=0.7, marker="o", markersize=7, markeredgecolor="white",
            markeredgewidth=1.5)
    ax.plot([-100, 0], [y_m1, y_p0], color=cor, lw=2.5, ls="--",
            alpha=0.7)
    ax.plot(resB["x"], y_jus, color=cor, lw=3.0, label=label)

# Linha do padrão (100%)
ax.axhline(100, color=PAL["padrao"], lw=2.5, ls="-.",
           label="Padrão CONAMA (100%)", zorder=1)

# Faixas de fundo
ax.axhspan(0, 100, color="#E8F5E9", alpha=0.35, zorder=0)
ax.axhspan(100, 500, color="#FFEBEE", alpha=0.35, zorder=0)

# Marcadores verticais
for nome, x in PONTOS.items():
    ax.axvline(x, color="#B0B0B0", lw=0.9, ls=":", zorder=0)
    ax.text(x, 470, nome, ha="center", fontsize=10, fontweight="bold",
            color="#444",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#CCC", lw=0.7))

# Destacar P0
ax.axvline(0, color=PAL["p0"], lw=2.2, ls="--", alpha=0.9, zorder=2)

# Anotação dos valores em J2
ax.annotate("DBO₅ permanece\nacima do padrão\nem TODO o trecho",
            xy=(150, 100 * resB["DBO5"][np.argmin(np.abs(resB["x"]-150))] / 5.0),
            xytext=(230, 280),
            fontsize=10, color="#B22222", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#B22222", lw=1.3),
            arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.5))

ax.set_xlim(-220, 340)
ax.set_ylim(0, 500)
ax.set_xlabel("Distância a partir do lançamento P0 (m)", fontsize=11)
ax.set_ylabel("Concentração em relação ao padrão CONAMA (%)", fontsize=11)
ax.set_title("Conformidade Ambiental ao Longo do Rio Araçatuba\n"
             "Linhas acima de 100% indicam não conformidade",
             fontsize=12.5, pad=12)
ax.legend(loc="upper right", fontsize=9.5, ncol=2)
fig.tight_layout()
salvar(fig, "fig_x03_conformidade_rio.png")


# =============================================================================
# FIGURA 04 — PERFIL LONGITUDINAL DO EMISSÁRIO (com terreno e aterro)
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

for ax, (nome_plano, dados) in zip(axes, PLANOS.items()):
    L = dados["L"]
    V1 = dados["V1"]
    V2 = dados.get("V2")
    V3 = dados.get("V3", V2)

    # Perfil do terreno
    if V2:
        d_terreno = [V1["d"], V2["d"], V3["d"]]
        c_terreno = [V1["cota"], V2["cota"], V3["cota"]]
    else:
        d_terreno = [V1["d"], V3["d"]]
        c_terreno = [V1["cota"], V3["cota"]]

    # Perfil do crown
    d_crown = [V1["d"], V2["d"], V3["d"]] if V2 else [V1["d"], V3["d"]]
    c_crown = [V1["crown"], V2["crown"], V3["crown"]] if V2 else [V1["crown"], V3["crown"]]

    # Interpolação fina para suavizar
    f_terr = interp1d(d_terreno, c_terreno, kind="linear")
    f_crow = interp1d(d_crown, c_crown, kind="linear")
    d_fino = np.linspace(0, L, 300)
    c_terr_fino = f_terr(d_fino)
    c_crow_fino = f_crow(d_fino)

    # Fill do aterro (área entre terreno e crown + 1m recobrimento)
    recobrimento = c_crow_fino + 1.0
    mask = recobrimento > c_terr_fino
    ax.fill_between(d_fino, c_terr_fino, recobrimento,
                    where=mask, color="#A0826D", alpha=0.45,
                    label="Aterro necessário")

    # Terreno
    ax.plot(d_fino, c_terr_fino, color="#4A3B2C", lw=2.5,
            label="Terreno natural", zorder=3)

    # Crown
    ax.plot(d_fino, c_crow_fino, color=PAL["A"], lw=2.8,
            label="Geratriz superior do tubo (crown)", zorder=4)

    # Vértices
    for v_nome, v in [("V1", V1), ("V2", V2) if V2 else None,
                      ("V3", V3)]:
        if v is None:
            continue
        ax.scatter(v["d"], v["cota"], s=110, marker="o",
                   facecolor="white", edgecolor="#4A3B2C",
                   linewidth=2, zorder=5)
        ax.annotate(f"{v_nome}\n{v['cota']:.2f} m",
                    xy=(v["d"], v["cota"]),
                    xytext=(0, 15), textcoords="offset points",
                    fontsize=9, fontweight="bold", ha="center",
                    color="#333")

    # Poço de sucção em V2 (se existir)
    if V2:
        ax.scatter(V2["d"], V2["cota"], s=350, marker="s",
                   facecolor=PAL["B"], edgecolor="black",
                   linewidth=1.5, zorder=6,
                   label="Poço de sucção (bombeamento)")
        ax.annotate("Vale natural",
                    xy=(V2["d"], V2["cota"]),
                    xytext=(V2["d"] + 60, V2["cota"] + 2),
                    fontsize=10, color=PAL["B"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=PAL["B"], lw=1.2),
                    arrowprops=dict(arrowstyle="->", color=PAL["B"], lw=1.3))

    ax.set_xlim(-10, L + 10)
    ax.set_ylim(min(c_terr_fino) - 0.5, max(c_terr_fino) + 2.5)
    ax.set_xlabel("Distância a partir da ETE (m)", fontsize=10)
    ax.set_ylabel("Cota (m)", fontsize=10)
    ax.set_title(f"Plano {nome_plano}  —  Comprimento total: {L:.1f} m",
                 fontsize=11, pad=8)
    ax.legend(loc="upper right", fontsize=8.5)

fig.suptitle("Perfil Longitudinal do Emissário — Comparação dos 3 Traçados\n"
             "Aterro necessário para regularização topográfica",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout()
salvar(fig, "fig_x04_perfil_emissario.png")


# =============================================================================
# FIGURA 05 — ZONA DE MISTURA 3D (matplotlib)
# =============================================================================
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection="3d")

# Grid
x3 = np.linspace(0, 120, 80)
y3 = np.linspace(-4.09, 4.09, 40)
X3, Y3 = np.meshgrid(x3, y3)

# Modelo de pluma gaussiana
h, S = PROF, DECLIV
u_star = np.sqrt(9.81 * h * S)
eps_y = 0.15 * h * u_star
U_B = CEN_B["V"]
sigma_y0 = 0.5

# Concentração central ao longo de x
centro_x = np.linspace(-200, 310.51, 511)
P_center = np.interp(x3, resB["x"], resB["P_total"])
P_bg = resB["x"][0] * 0 + resB["x"][0] * 0 + 0.082  # P_bg ~ 0,082

# Sigma y em função de x
sig = np.sqrt(sigma_y0**2 + 2 * eps_y * np.maximum(x3, 0) / U_B)

# Concentração 2D
C3 = np.zeros_like(X3)
for j, (s_, cv) in enumerate(zip(sig, P_center)):
    lat = np.exp(-Y3[:, j]**2 / (2 * s_**2))
    lat_n = lat / max(lat.max(), 1e-12)
    C3[:, j] = P_bg + (cv - P_bg) * lat_n

# Superfície
cmap_3d = plt.get_cmap("plasma")
surf = ax.plot_surface(X3, Y3, C3, cmap=cmap_3d,
                        edgecolor="none", alpha=0.92, rstride=1, cstride=1,
                        linewidth=0, antialiased=True)

# Padrão de classe (100% do padrão) - superfície em 0,10 mg/L
Z_padrao = np.full_like(X3, PADROES["P_total"])
ax.plot_wireframe(X3, Y3, Z_padrao, color=PAL["padrao"],
                  linewidth=1.0, alpha=0.5, rstride=4, cstride=4)

# Ponto de lançamento
ax.scatter([0], [0], [P_bg + 0.01], s=300, marker="*",
           color=PAL["p0"], edgecolor="black", linewidth=1.5,
           zorder=10, label="P0 (lançamento)")

# Rótulos
ax.set_xlabel("Distância a partir do P0 (m)", fontsize=10, labelpad=10)
ax.set_ylabel("Posição lateral no canal (m)", fontsize=10, labelpad=10)
ax.set_zlabel("Fósforo total (mg/L)", fontsize=10, labelpad=10)
ax.set_title("Zona de Mistura 3D — Distribuição Espaço-Lateral da Pluma de P-total\n"
             "Cenário B (Q90 = 0,745 m³/s)  —  plano branco: padrão CONAMA (0,10 mg/L)",
             fontsize=11.5, pad=15)

ax.view_init(elev=25, azim=-55)
ax.legend(loc="upper right", fontsize=9)

cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, aspect=20)
cbar.set_label("Fósforo total (mg/L)", fontsize=10)

fig.tight_layout()
salvar(fig, "fig_x05_zona_mistura_3d.png")


# =============================================================================
# FIGURA 06 — ZONA DE MISTURA 3D INTERATIVA (HTML Plotly)
# =============================================================================
fig3d = go.Figure(data=[
    go.Surface(
        x=X3, y=Y3, z=C3,
        colorscale="Plasma",
        colorbar=dict(title="P-total (mg/L)", thickness=20),
        hovertemplate="x=%{x:.1f} m<br>y=%{y:.2f} m<br>P=%{z:.4f} mg/L<extra></extra>",
        opacity=0.95,
        showscale=True,
    )
])

# Superfície do padrão
fig3d.add_trace(go.Surface(
    x=X3, y=Y3, z=np.full_like(X3, PADROES["P_total"]),
    colorscale=[[0, "rgba(139,38,53,0.3)"], [1, "rgba(139,38,53,0.3)"]],
    showscale=False,
    name="Padrão CONAMA (0,10 mg/L)",
    hovertemplate="Padrão: 0,10 mg/L<extra></extra>",
))

# Ponto P0
fig3d.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[P_bg + 0.01],
    mode="markers",
    marker=dict(symbol="diamond", size=12, color="#E5A54B",
                line=dict(color="black", width=2)),
    name="P0 (lançamento)",
))

fig3d.update_layout(
    title=dict(
        text="Zona de Mistura 3D Interativa — Pluma de Fósforo Total<br>"
             "<sub>Arraste para rotacionar  |  Scroll para zoom  |  "
             "Passe o mouse para ver valores</sub>",
        x=0.5, xanchor="center", font=dict(size=16),
    ),
    scene=dict(
        xaxis_title="Distância do P0 (m)",
        yaxis_title="Posição lateral (m)",
        zaxis_title="P-total (mg/L)",
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)),
        aspectratio=dict(x=1.5, y=0.8, z=0.5),
    ),
    height=700,
    margin=dict(l=20, r=20, t=100, b=20),
    paper_bgcolor="white",
)

fig3d.write_html(OUTPUT / "fig_x06_zona_mistura_3d_interativo.html",
                 include_plotlyjs="cdn")
print("[OK] fig_x06_zona_mistura_3d_interativo.html")


# =============================================================================
# FIGURA 07 — COMPARAÇÃO DE SOLUÇÕES (barras empilhadas + ranking)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                gridspec_kw={"width_ratios": [1.2, 1]})

# --- Painel esquerdo: custos empilhados ---
nomes = list(SOLUCOES.keys())
y_pos = np.arange(len(nomes))
inicial = [SOLUCOES[n]["inicial"] for n in nomes]
op20 = [SOLUCOES[n]["op20"] for n in nomes]
total = [SOLUCOES[n]["total"] for n in nomes]

cores_barras = [PAL["B"] if "Bomb" in n else "#8B7355" for n in nomes]

ax1.barh(y_pos, inicial, color=cores_barras, alpha=0.85,
         height=0.6, label="Custo inicial")
ax1.barh(y_pos, op20, left=inicial, color="#F4DCCF", alpha=0.95,
         height=0.6, label="Custo operacional (20 anos)")

for i, (ini, op, tot) in enumerate(zip(inicial, op20, total)):
    ax1.text(tot + 5000, i, f"R$ {tot:,.0f}",
             va="center", fontsize=9.5, fontweight="bold")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(nomes, fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel("Custo total em 20 anos (R$)", fontsize=10.5)
ax1.set_title("Comparação de Custos por Solução", fontsize=11.5, pad=10)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_xlim(0, max(total) * 1.25)

# --- Painel direito: ranking ---
ranking = sorted(SOLUCOES.items(), key=lambda x: x[1]["total"])
nomes_rank = [r[0] for r in ranking]
valores_rank = [r[1]["total"] for r in ranking]

cores_rank = []
for n in nomes_rank:
    if "Bomb" in n:
        cores_rank.append(PAL["A"])
    else:
        cores_rank.append("#8B7355")

bars = ax2.barh(range(len(nomes_rank)), valores_rank,
                color=cores_rank, alpha=0.9, height=0.6)

for i, (bar, v) in enumerate(zip(bars, valores_rank)):
    ax2.text(v + 3000, i, f"{i+1}º   R$ {v:,.0f}",
             va="center", fontsize=10, fontweight="bold")

ax2.set_yticks(range(len(nomes_rank)))
ax2.set_yticklabels(nomes_rank, fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("Custo total (R$)", fontsize=10.5)
ax2.set_title("Ranking (menor custo primeiro)", fontsize=11.5, pad=10)
ax2.set_xlim(0, max(valores_rank) * 1.35)

# Anotação do mais econômico
ax2.annotate("Solução recomendada\n(menor custo)",
             xy=(valores_rank[0], 0),
             xytext=(valores_rank[0] * 0.4, 1.6),
             fontsize=10, color=PAL["A"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec=PAL["A"], lw=1.3),
             arrowprops=dict(arrowstyle="->", color=PAL["A"], lw=1.5))

fig.suptitle("Análise Comparativa Técnico-Econômica das Soluções\n"
             "Aterro topográfico × Estação elevatória de pequeno porte",
             fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout()
salvar(fig, "fig_x07_comparacao_solucoes.png")


# =============================================================================
# FIGURA 08 — REGIME PLUVIOMÉTRICO MENSAL
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 6))

meses = list(PLUV.keys())
valores = list(PLUV.values())
total = sum(valores)
media_mensal_hist = PLUV_MEDIA_HISTORICA / 12

cores_meses = ["#4A90E2" if v < media_mensal_hist else "#E5A54B"
               for v in valores]

bars = ax.bar(meses, valores, color=cores_meses, alpha=0.85,
              edgecolor="black", linewidth=1.0, width=0.7)

# Linha da média histórica mensal
ax.axhline(media_mensal_hist, color=PAL["B"], lw=2.5, ls="--",
           label=f"Média histórica mensal ({media_mensal_hist:.0f} mm/mês)")

# Valores acima das barras
for bar, v in zip(bars, valores):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
            f"{v:.1f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333")

# Média do período
media_periodo = total / len(valores)
ax.axhline(media_periodo, color=PAL["A"], lw=2.5, ls=":",
           label=f"Média do período ({media_periodo:.1f} mm/mês)")

# Destaque do período mais seco
idx_seco = np.argmin(valores)
ax.annotate(f"Mês mais seco\n({valores[idx_seco]:.1f} mm)",
            xy=(idx_seco, valores[idx_seco]),
            xytext=(idx_seco + 0.7, valores[idx_seco] + 25),
            fontsize=10, color=PAL["B"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=PAL["B"], lw=1.3),
            arrowprops=dict(arrowstyle="->", color=PAL["B"], lw=1.5))

# Anotação do total
ax.text(0.98, 0.95, f"Total do período: {total:.1f} mm",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5",
                  ec="#999", lw=1.2))

ax.set_ylim(0, max(valores) * 1.35)
ax.set_ylabel("Precipitação mensal (mm)", fontsize=11)
ax.set_xlabel("Mês / Ano", fontsize=11)
ax.set_title("Regime Pluviométrico — Estação Campo D'Una (CEMADEN)\n"
             "Período: Setembro/2025 a Setembro/2026",
             fontsize=12.5, pad=12)
ax.legend(loc="upper right", fontsize=9.5)
ax.tick_params(axis="x", rotation=45)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")

fig.tight_layout()
salvar(fig, "fig_x08_pluviometria.png")


# =============================================================================
# FIGURA 09 — SANKEY DE CARGAS (interativo, plotly)
# =============================================================================
# Carga em kg/dia no cenário Q90
carga_rio_dbo = CEN_B["Q_rio"] * RIO_M1["DBO5"] * 86.4      # 1.313 kg/dia
carga_eff_dbo = EFLUENTE["Q"] * EFLUENTE["DBO5"] * 86.4     # 4,53 kg/dia
carga_total = carga_rio_dbo + carga_eff_dbo                 # 1.317,6 kg/dia

fig_sankey = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=25,
        thickness=35,
        line=dict(color="black", width=0.8),
        label=[
            f"Rio a montante (M1)<br>{carga_rio_dbo:.1f} kg/dia DBO₅",
            f"Efluente ETE<br>{carga_eff_dbo:.2f} kg/dia DBO₅",
            f"Rio a jusante (J2)<br>{carga_total:.1f} kg/dia DBO₅",
        ],
        color=[PAL["A"], PAL["p0"], PAL["B"]],
    ),
    link=dict(
        source=[0, 1],
        target=[2, 2],
        value=[carga_rio_dbo, carga_eff_dbo],
        color=["rgba(13,74,112,0.55)", "rgba(229,165,75,0.75)"],
        hovertemplate="Contribuição: %{value:.2f} kg/dia DBO₅<br>"
                      "Percentual: %{customdata:.3f}%<extra></extra>",
        customdata=[
            100 * carga_rio_dbo / carga_total,
            100 * carga_eff_dbo / carga_total,
        ],
    ),
)])

fig_sankey.update_layout(
    title=dict(
        text="Balanço de Carga de DBO₅ no Rio Araçatuba — Cenário Q90<br>"
             "<sub>O efluente tratado representa apenas "
             f"{100*carga_eff_dbo/carga_total:.3f}% da carga total transportada</sub>",
        x=0.5, xanchor="center", font=dict(size=15),
    ),
    font=dict(size=12, family="Arial"),
    height=450,
    margin=dict(l=40, r=40, t=100, b=40),
    paper_bgcolor="white",
)

fig_sankey.write_html(OUTPUT / "fig_x09_sankey_cargas.html",
                       include_plotlyjs="cdn")
print("[OK] fig_x09_sankey_cargas.html")


# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
print("\n" + "=" * 66)
print("  FIGURAS COMPLEMENTARES GERADAS COM SUCESSO")
print("=" * 66)
print(f"  Diretório: {OUTPUT}")
print(f"  8 figuras PNG (400 DPI)")
print(f"  2 figuras HTML interativas (Plotly)")
print("=" * 66)
print("\n  Lista de arquivos:")
for f in sorted(OUTPUT.glob("*")):
    print(f"    - {f.name}")