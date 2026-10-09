# -*- coding: utf-8 -*-
"""
PERFIL LONGITUDINAL E VISTA ISOMETRICA DO RIO ARACATUBA
Com pluma de dispersao do efluente tratado

Autor: Eng. Muriell Bernardo Macedo (CREA-SC 222578-3)
Licenca: MIT
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from dados.parametros_capitulo8 import (
    OD_SAT, PROF, DECLIV, PONTOS, CEN_A, CEN_B, EFLUENTE, RIO_M1,
    COEF, PADROES, F_DBO, SECOES, COTAS_TERRENO,
)

warnings.filterwarnings("ignore")

OUTPUT = Path(__file__).parent / "output_v3"
OUTPUT.mkdir(exist_ok=True)

PAL = {
    "agua":   "#CFE4EE",
    "leito":  "#4A3B2C",
    "margem": "#A0826D",
    "sup":    "#1F5C7A",
    "P0":     "#E5A54B",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#2B2B2B",
    "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#EEEEEE",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.dpi": 120,
})


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


def solve(cen):
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
            "Coli": Coli}


resA = solve(CEN_A)
resB = solve(CEN_B)


def geo_interp(x_target):
    nome = list(SECOES.keys())
    xs = np.array([SECOES[n]["dist"] for n in nome])
    L = np.array([SECOES[n]["L"] for n in nome])
    he = np.array([SECOES[n]["h_esq"] for n in nome])
    hc = np.array([SECOES[n]["h_centro"] for n in nome])
    hd = np.array([SECOES[n]["h_dir"] for n in nome])
    ct = np.array([COTAS_TERRENO[n] for n in nome])

    fL = interp1d(xs, L, kind="linear", fill_value="extrapolate")
    fe = interp1d(xs, he, kind="linear", fill_value="extrapolate")
    fc = interp1d(xs, hc, kind="linear", fill_value="extrapolate")
    fd = interp1d(xs, hd, kind="linear", fill_value="extrapolate")
    ft = interp1d(xs, ct, kind="linear", fill_value="extrapolate")
    return (float(fL(x_target)), float(fe(x_target)), float(fc(x_target)),
            float(fd(x_target)), float(ft(x_target)))


fig, ax = plt.subplots(figsize=(14, 6))

x = np.linspace(-200, 310.51, 800)
hc_arr = np.zeros_like(x)
ct_arr = np.zeros_like(x)

for i, xi in enumerate(x):
    _, _, hc_arr[i], _, ct_arr[i] = geo_interp(xi)

fundo = ct_arr - hc_arr
superf = ct_arr

ax.fill_between(x, superf, superf + 1.5, color=PAL["margem"], alpha=0.55,
                label="Margem (terreno)")
ax.plot(x, superf, color=PAL["sup"], lw=1.5, label="Superficie da agua")
ax.fill_between(x, fundo, superf, color=PAL["agua"], alpha=0.85, zorder=1)

conc_plume = np.full_like(x, np.nan)
for i, xi in enumerate(x):
    if xi >= 0:
        conc_plume[i] = np.interp(xi, resB["x"], resB["P_total"])

norm = Normalize(vmin=0.078, vmax=0.095)
cmap_pluma = LinearSegmentedColormap.from_list(
    "pluma", ["#CFE4EE", "#7EC9E0", "#F4C471", "#EF7A3D", "#B8432B"], N=256)

for i in range(len(x) - 1):
    if np.isnan(conc_plume[i]) or np.isnan(conc_plume[i + 1]):
        continue
    c = cmap_pluma(norm(0.5 * (conc_plume[i] + conc_plume[i + 1])))
    ax.fill_between([x[i], x[i + 1]],
                    [fundo[i], fundo[i + 1]],
                    [superf[i], superf[i + 1]],
                    color=c, alpha=0.85, linewidth=0, zorder=2)

ax.plot(x, fundo, color=PAL["leito"], lw=2.2, label="Leito do rio")

for nome, xi in PONTOS.items():
    ax.axvline(xi, color="#B0B0B0", lw=0.9, linestyle=":", zorder=0)
    ax.text(xi, superf.max() + 1.7, nome, ha="center",
            fontsize=9, fontweight="bold", color="#444")

ax.axvline(0, color=PAL["P0"], lw=1.8, linestyle="--", alpha=0.9, zorder=3)
ax.annotate("P0\nLancamento", xy=(0, superf.max()),
            xytext=(20, 40), textcoords="offset points",
            fontsize=9.5, color=PAL["P0"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=PAL["P0"], lw=1.2),
            arrowprops=dict(arrowstyle="->", color=PAL["P0"], lw=1.3))

sm = plt.cm.ScalarMappable(cmap=cmap_pluma, norm=norm)
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label("Fosforo total na pluma (mg/L)", fontsize=10)

ax.set_xlim(-220, 330)
ax.set_ylim(1.4, superf.max() + 2.2)
ax.set_xlabel("Distancia a partir do lancamento P0 (m)")
ax.set_ylabel("Cota (m)")
ax.set_title("Perfil Longitudinal do Rio Aracatuba - com Pluma de Dispersao\n"
             "Cenario B (Q90 = 0,745 m3/s)", pad=12)
ax.legend(loc="lower right", fontsize=9, ncol=2)

fig.tight_layout()
fig.savefig(OUTPUT / "fig_v3_A_perfil_longitudinal.png", dpi=400,
            bbox_inches="tight", facecolor="white")
print("[OK] fig_v3_A_perfil_longitudinal.png")
plt.close(fig)


fig, ax = plt.subplots(figsize=(15, 7))

DX = 0.9
DY = -0.55
DEPTH_SCALE = 8.0

xs_iso = np.linspace(-200, 310.51, 60)
concs_iso = np.full_like(xs_iso, np.nan)
for i, xi in enumerate(xs_iso):
    if xi >= 0:
        concs_iso[i] = np.interp(xi, resB["x"], resB["P_total"])

norm_iso = Normalize(vmin=0.078, vmax=0.095)
cmap_iso = LinearSegmentedColormap.from_list(
    "pluma_iso",
    ["#E8F4F8", "#8FCBDB", "#7EC9E0", "#F4C471", "#EF7A3D", "#B8432B"],
    N=256)

for i in reversed(range(len(xs_iso))):
    xi = xs_iso[i]
    L, he, hc, hd, _ = geo_interp(xi)

    pts_local = [
        (-L / 2, 0.0),
        (-L / 2, -he * DEPTH_SCALE),
        (0.0, -hc * DEPTH_SCALE),
        (+L / 2, -hd * DEPTH_SCALE),
        (+L / 2, 0.0),
    ]

    x_screen = xi * DX
    y_screen = xi * DY
    pts = [(y + x_screen, z + y_screen) for y, z in pts_local]

    if np.isnan(concs_iso[i]):
        cor, alpha = "#E8F4F8", 0.55
    else:
        cor, alpha = cmap_iso(norm_iso(concs_iso[i])), 0.95

    poly = Polygon(pts, closed=True, facecolor=cor, edgecolor="#3A3A3A",
                   linewidth=0.7, alpha=alpha, zorder=100 + i)
    ax.add_patch(poly)

for i in range(len(xs_iso) - 1):
    xi = xs_iso[i]
    x_screen = xi * DX
    y_screen = xi * DY
    xa = x_screen
    ya = y_screen - geo_interp(xi)[2] * DEPTH_SCALE
    xb = xs_iso[i + 1] * DX
    yb = xs_iso[i + 1] * DY - geo_interp(xs_iso[i + 1])[2] * DEPTH_SCALE
    ax.plot([xa, xb], [ya, yb], color="#3A3A3A", lw=0.7, zorder=150 + i)

xp = 0 * DX
yp = 0 * DY
ax.plot(xp, yp + 0.3, marker="*", markersize=26, color=PAL["P0"],
        markeredgecolor="black", markeredgewidth=1.4, zorder=1000)
ax.annotate("P0\nLancamento", xy=(xp, yp + 0.3),
            xytext=(15, 35), textcoords="offset points",
            fontsize=10, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=PAL["P0"], lw=1.4),
            arrowprops=dict(arrowstyle="->", color=PAL["P0"], lw=1.5),
            zorder=1001)

for nome, xi in PONTOS.items():
    xs_ = xi * DX
    ys_ = xi * DY + 0.5
    ax.text(xs_, ys_, nome, ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#2B2B2B",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#B0B0B0", lw=0.8), zorder=1002)

ax.annotate("", xy=(250 * DX + 3, 250 * DY - 0.5),
            xytext=(-180 * DX, -180 * DY + 0.5),
            arrowprops=dict(arrowstyle="->", color="#333",
                            lw=2, alpha=0.5))

sm = plt.cm.ScalarMappable(cmap=cmap_iso, norm=norm_iso)
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Fosforo total (mg/L)", fontsize=10.5)

ax.set_aspect("equal")
ax.set_xlim(-220 * DX - 6, 330 * DX + 8)
ax.set_ylim(-180 * 0.55 - 3, 15)
ax.set_xlabel("Posicao longitudinal (m)")
ax.set_ylabel("Profundidade e posicao lateral (unidades visuais)")
ax.set_title("Vista Isometrica do Rio Aracatuba - Pluma de Dispersao\n"
             "Cada secao mostra a profundidade real medida em campo "
             "(Campanha C3)", pad=12)
ax.grid(False)

fig.tight_layout()
fig.savefig(OUTPUT / "fig_v3_B_rio_isometrico.png", dpi=400,
            bbox_inches="tight", facecolor="white")
print("[OK] fig_v3_B_rio_isometrico.png")
plt.close(fig)


fig, ax = plt.subplots(figsize=(14, 5.5))

x_grid = np.linspace(-200, 310.51, 600)
z_grid = np.linspace(-2.7, 0.0, 180)
X, Z = np.meshgrid(x_grid, z_grid)


def depth_max(x_target):
    _, _, hc, _, _ = geo_interp(x_target)
    return hc


C = np.full_like(X, 0.078)
for j, xi in enumerate(x_grid):
    if xi >= 0:
        C[:, j] = np.interp(xi, resB["x"], resB["P_total"])

dmax = np.array([depth_max(xi) for xi in x_grid])
mask = Z < -np.maximum(dmax, 0.1)
C = np.ma.array(C, mask=mask)

cmap_h = LinearSegmentedColormap.from_list(
    "pluma_h",
    ["#CFE4EE", "#8FCBDB", "#7EC9E0", "#F4C471", "#EF7A3D", "#B8432B"],
    N=256)
im = ax.pcolormesh(X, Z, C, cmap=cmap_h, shading="auto",
                   vmin=0.078, vmax=0.095)

cs = ax.contour(X, Z, C, levels=[PADROES["P_total"]],
                colors=PAL["P0"], linewidths=2, linestyles="--")
try:
    ax.clabel(cs, fmt="<= %.2f mg/L", fontsize=9)
except Exception:
    pass

z_bed = -dmax
ax.plot(x_grid, z_bed, color=PAL["leito"], lw=2, label="Leito do rio")
ax.plot(x_grid, np.zeros_like(x_grid), color=PAL["sup"], lw=1.5,
        alpha=0.7, label="Superficie da agua")
ax.fill_between(x_grid, 0, 0.5, color=PAL["margem"], alpha=0.5)
ax.fill_between(x_grid, z_bed, -3, color="#6B5B4A", alpha=0.5)

for nome, xi in PONTOS.items():
    ax.axvline(xi, color="white", lw=1.0, linestyle=":", alpha=0.7)
    ax.text(xi, 0.35, nome, ha="center", fontsize=9,
            fontweight="bold", color="#2B2B2B")

ax.axvline(0, color=PAL["P0"], lw=2, linestyle="--", alpha=0.9)

cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label("Fosforo total (mg/L)", fontsize=10.5)

ax.set_xlim(-220, 330)
ax.set_ylim(-3, 0.6)
ax.set_xlabel("Distancia a partir do lancamento P0 (m)")
ax.set_ylabel("Profundidade relativa a superficie (m)")
ax.set_title("Mapa de Calor Longitudinal - Pluma de Fosforo Total\n"
             "Concentracao x Distancia x Profundidade (Cenario B, Q90)",
             pad=12)
ax.legend(loc="lower left", fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT / "fig_v3_C_heatmap_longitudinal.png", dpi=400,
            bbox_inches="tight", facecolor="white")
print("[OK] fig_v3_C_heatmap_longitudinal.png")
plt.close(fig)


print("\n" + "=" * 62)
print("  PERFIL DO RIO E PLUMA GERADOS COM SUCESSO")
print("=" * 62)
print(f"  Diretorio: {OUTPUT}")
print(f"  Figuras: 3 (PNG, 400 DPI)")
print("=" * 62)