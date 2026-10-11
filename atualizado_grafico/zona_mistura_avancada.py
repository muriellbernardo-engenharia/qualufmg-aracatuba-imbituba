# -*- coding: utf-8 -*-
"""
=============================================================================
ZONA DE MISTURA — VISUALIZAÇÃO AVANÇADA E PROFISSIONAL
Rio Araçatuba — ETE Residencial Imbituba
=============================================================================
Gera 5 figuras PNG + 1 HTML interativo (Plotly) sobre a zona de mistura
regulatória (CONAMA 430/2011), com os dados do estudo:
  - Cenário A: Q = 2,973 m³/s (vazão medida)
  - Cenário B: Q = 0,745 m³/s (Q90 — Portaria IMA 192/2025)
  - Parâmetros: DBO₅, N-amoniacal, P-total, N-nitrato, OD, Coliformes

Autor: Eng.ª Muriell Bernardo Macedo (CREA-SC 222578-3)
Licença: MIT
=============================================================================
"""

import warnings
from pathlib import Path
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "output_zona_mistura"
OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Saída: {OUTPUT}\n")

# -----------------------------------------------------------------------------
# ESTILO
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.titlepad": 14,
    "axes.edgecolor": "#2B2B2B", "axes.linewidth": 1.0,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.7, "grid.linestyle": "-",
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "legend.fontsize": 9,
    "figure.dpi": 120, "savefig.dpi": 400,
    "savefig.facecolor": "white", "savefig.bbox": "tight",
})

# Paleta
COR_A = "#0D4A70"
COR_B = "#C1440E"
COR_PADRAO = "#8B2635"
COR_P0 = "#E5A54B"
COR_RIO = "#4A90E2"
COR_EFLUENTE = "#D35400"

# Colormaps personalizados
CMAP_PLUME = LinearSegmentedColormap.from_list("plume",
    ["#F7FCF5", "#C7E9C0", "#74C476", "#FEB24C", "#F03B20", "#8B2635"])
CMAP_PLUME_B = LinearSegmentedColormap.from_list("plumeB",
    ["#EBF5FB", "#AED6F1", "#5DADE2", "#2E86C1", "#1B4F72", "#0D1B2A"])
CMAP_TERRA = LinearSegmentedColormap.from_list("terra",
    ["#3E2723", "#6D4C41", "#8D6E63", "#BCAAA4", "#D7CCC8", "#EFEBE9"])


# =============================================================================
# DADOS DO ESTUDO
# =============================================================================
# Geometria e hidráulica
LARG_EF_P0 = 8.18      # m (largura efetiva em P0)
LARG_TOT_P0 = 8.64     # m (largura total)
PROF_MEDIA = 1.78      # m (Simpson)
DECLIV = 0.000215      # m/m

# Cenários
CEN_A = dict(Q=2.973, V=0.10494, K2=0.800, label="Cenário A — Q = 2,973 m³/s")
CEN_B = dict(Q=0.745, V=0.05253, K2=0.566, label="Cenário B — Q90 = 0,745 m³/s")

# Efluente (constante)
Q_EFLUENTE = 0.00234   # m³/s
N_NO3_EFLUENTE = 24.47 # mg/L

# Valores iniciais (após mistura em P0) — Cenário B
P0_B = dict(
    DBO5=20.41, N_NH3=0.050, N_NO2=0.030, N_NO3=0.475,
    P_total=0.089, OD=6.49, Coli=80.9
)
# Cenário A
P0_A = dict(
    DBO5=20.40, N_NH3=0.050, N_NO2=0.030, N_NO3=0.419,
    P_total=0.082, OD=6.50, Coli=78.7
)

# Fundo (M1 — montante)
BG = dict(DBO5=20.40, N_NH3=0.05, N_NO3=0.40,
          P_total=0.08, OD=6.50, Coli=78.0)

# Padrões Classe 2 (CONAMA 357/2005)
PADROES = dict(DBO5=5.0, N_NH3=2.0, N_NO2=1.0, N_NO3=10.0,
               P_total=0.10, OD=5.0, Coli=1000.0)

# Ponto P0 e extensão
X_MIN, X_MAX = -100, 800   # m
Y_MEIO = 0.0


# =============================================================================
# FÍSICA DA DISPERSÃO
# =============================================================================
def sigma_y(x, U, h=PROF_MEDIA, S=DECLIV, sigma0=0.5):
    """
    Crescimento lateral da pluma (Fischer, 1979):
        ε_y ≈ 0,15 · h · u*
        σ_y(x) = √(σ_y0² + 2 · ε_y · x / U)
    """
    u_star = np.sqrt(9.81 * h * S)
    eps_y = 0.15 * h * u_star
    x_pos = np.maximum(x, 0)
    return np.sqrt(sigma0**2 + 2 * eps_y * x_pos / U)


def concentracao_2d(x_arr, y_arr, C_center, C_bg, U,
                     h=PROF_MEDIA, S=DECLIV):
    """
    Concentração na seção (x, y):
        C(x,y) = C_bg + (C_center(x) − C_bg) · exp(−y² / (2 σ_y²))
    """
    sig = sigma_y(x_arr, U, h, S)
    sig = np.where(sig < 1e-6, 1e-6, sig)
    Z = np.zeros_like(y_arr)
    for j, (s_, cc) in enumerate(zip(sig, C_center)):
        Z[:, j] = C_bg + (cc - C_bg) * np.exp(-y_arr[:, j]**2 / (2 * s_**2))
    return Z


def salvar(fig, nome):
    p = OUTPUT / nome
    try:
        fig.savefig(str(p), dpi=400, bbox_inches="tight", facecolor="white")
        kb = p.stat().st_size / 1024
        print(f"  [OK] {nome}  ({kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [ERRO] {nome}: {e}")
        return False
    finally:
        plt.close(fig)


def rodar(func, *args):
    try:
        func(*args)
    except Exception as e:
        print(f"  [ERRO em {func.__name__}] {e}")
        traceback.print_exc()


# =============================================================================
# FIGURA ZM-1 — PLUMA 3D ULTRA-VISUAL (Cenário B)
# =============================================================================
def fig_zm1():
    """Pluma 3D com superfície colorida, projeção no fundo e curvas de nível."""

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Malha
    nx, ny = 180, 80
    x_arr = np.linspace(0, 500, nx)
    y_arr = np.linspace(-8, 8, ny)
    X, Y = np.meshgrid(x_arr, y_arr)

    # Concentração central ao longo de x (interpolação dos dados)
    x_data = np.array([0, 50, 100, 200, 310, 400, 500])
    p_center_B = np.array([P0_B["P_total"], 0.089, 0.089, 0.089,
                            0.089, 0.089, 0.089])

    C_center = np.interp(x_arr, x_data, p_center_B)

    # Concentração 2D
    Z = concentracao_2d(x_arr, y_arr, C_center, BG["P_total"], CEN_B["V"])

    # --- Superfície principal ---
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=CMAP_PLUME,
        edgecolor="none",
        alpha=0.96,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True,
        vmin=0.05, vmax=0.15,
    )

    # --- Plano do padrão CONAMA ---
    Z_pad = np.full_like(X, PADROES["P_total"])
    ax.plot_wireframe(X, Y, Z_pad,
                       color=COR_PADRAO, linewidth=1.2,
                       alpha=0.55, rstride=10, cstride=10)

    # --- Projeção da pluma no fundo (curvas de nível) ---
    ax.contour(X, Y, Z, zdir="z", offset=0.03,
               levels=np.linspace(0.05, 0.14, 15),
               cmap=CMAP_PLUME, alpha=0.9, linewidths=1.6)

    # --- Envelope da zona de mistura (contorno σ_y) ---
    sig_B = sigma_y(x_arr, CEN_B["V"])
    # Envelope superior e inferior
    ax.plot(x_arr, +2*sig_B, np.full_like(x_arr, 0.155),
            color=COR_P0, lw=2.5, ls="--", alpha=0.85)
    ax.plot(x_arr, -2*sig_B, np.full_like(x_arr, 0.155),
            color=COR_P0, lw=2.5, ls="--", alpha=0.85,
            label="Envelope ±2σ_y")

    # --- Ponto P0 ---
    ax.scatter([0], [0], [P0_B["P_total"]],
               s=500, marker="*", color=COR_P0,
               edgecolor="black", linewidth=2.5, zorder=20,
               label="P0 (lançamento)")
    ax.text(0, 0, P0_B["P_total"] + 0.008, "P0",
            fontsize=11, fontweight="bold", color=COR_P0)

    # --- Anotações de valores ---
    for x_anot in [100, 200, 300, 500]:
        idx = np.argmin(np.abs(x_arr - x_anot))
        c_val = C_center[idx]
        sig_val = sig_B[idx]
        ax.scatter([x_anot], [0], [c_val], s=80, marker="o",
                   color="white", edgecolor=COR_B, linewidth=2, zorder=15)
        ax.text(x_anot, 0, c_val + 0.006,
                f"{c_val:.3f}\nσ={sig_val:.1f}m",
                fontsize=8, ha="center", fontweight="bold",
                color=COR_B,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=COR_B, lw=0.8, alpha=0.9))

    # --- Marcadores de J1 e J2 ---
    for x_j, nome_j in [(97, "J1"), (296.28, "J2")]:
        idx = np.argmin(np.abs(x_arr - x_j))
        ax.scatter([x_j], [0], [C_center[idx]], s=180, marker="s",
                   color="#2E8B57", edgecolor="black", linewidth=1.8,
                   zorder=15)
        ax.text(x_j, 0, C_center[idx] + 0.012, nome_j,
                fontsize=10, fontweight="bold", color="#2E8B57",
                ha="center")

    # --- Rotulos e limites ---
    ax.set_xlabel("\nDistância a partir de P0 (m)", fontsize=11, labelpad=14)
    ax.set_ylabel("\nPosição lateral no canal (m)", fontsize=11, labelpad=14)
    ax.set_zlabel("\nFósforo total (mg/L)", fontsize=11, labelpad=14)
    ax.set_title(
        "Figura ZM-1 — Pluma de Dispersão 3D do Fósforo Total\n"
        "Cenário B (Q90 = 0,745 m³/s)  ·  Padrão CONAMA Classe 2 = 0,10 mg/L",
        fontsize=13.5, fontweight="bold", pad=22)

    ax.set_xlim(0, 500)
    ax.set_ylim(-8, 8)
    ax.set_zlim(0.03, 0.17)
    ax.view_init(elev=22, azim=-60)

    # Barra de cores
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, aspect=22)
    cbar.set_label("Fósforo total (mg/L)", fontsize=10.5)
    cbar.ax.axhline(PADROES["P_total"], color=COR_PADRAO, lw=2.5)
    cbar.ax.text(1.5, PADROES["P_total"], "  CONAMA", va="center",
                 fontsize=9, color=COR_PADRAO, fontweight="bold")

    # Legenda
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COR_P0,
               markersize=18, markeredgecolor="black",
               label="P0 (lançamento)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2E8B57",
               markersize=11, markeredgecolor="black", label="J1, J2"),
        Line2D([0], [0], color=COR_P0, lw=2.5, ls="--",
               label="Envelope ±2σ_y"),
        Line2D([0], [0], color=COR_PADRAO, lw=2, ls=":",
               label=f"Padrão CONAMA ({PADROES['P_total']} mg/L)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9.5,
              facecolor="white", framealpha=0.95)

    fig.tight_layout()
    salvar(fig, "fig_ZM_1_pluma_3d.png")


# =============================================================================
# FIGURA ZM-2 — PLANTA (HEATMAP) COM PLUMA
# =============================================================================
def fig_zm2():
    """Vista de topo — heatmap da pluma no plano (x, y)."""

    fig, ax = plt.subplots(figsize=(14, 6))

    x_arr = np.linspace(0, 500, 400)
    y_arr = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x_arr, y_arr)

    # Concentração central (constante após P0 para P-total)
    x_data = np.array([0, 50, 100, 200, 310, 400, 500])
    C_center = np.interp(x_arr, x_data,
        np.full_like(x_data, P0_B["P_total"]))

    Z = concentracao_2d(x_arr, y_arr, C_center, BG["P_total"], CEN_B["V"])

    # --- Heatmap ---
    im = ax.contourf(X, Y, Z,
                      levels=np.linspace(0.05, 0.15, 25),
                      cmap=CMAP_PLUME, alpha=0.92, extend="both")

    # --- Contornos principais ---
    ax.contour(X, Y, Z,
               levels=[PADROES["P_total"]], colors=[COR_PADRAO],
               linewidths=3.5, linestyles="-")
    ax.contour(X, Y, Z,
               levels=[0.085, 0.09, 0.095], colors=["white"],
               linewidths=1.5, linestyles="--", alpha=0.8)
    ax.contour(X, Y, Z,
               levels=[BG["P_total"]], colors=["#333"],
               linewidths=2, linestyles=":")

    # --- Envelope σ_y ---
    sig_B = sigma_y(x_arr, CEN_B["V"])
    ax.plot(x_arr, +2*sig_B, color=COR_P0, lw=2.8, ls="--",
            label="Envelope ±2σ_y", zorder=5)
    ax.plot(x_arr, -2*sig_B, color=COR_P0, lw=2.8, ls="--", zorder=5)
    ax.fill_between(x_arr, -2*sig_B, +2*sig_B,
                    color=COR_P0, alpha=0.08, zorder=4)

    # --- Larguras das margens ---
    ax.axhline(+LARG_EF_P0/2, color="#333", lw=2.5, ls="-", alpha=0.6)
    ax.axhline(-LARG_EF_P0/2, color="#333", lw=2.5, ls="-", alpha=0.6)
    ax.text(480, LARG_EF_P0/2 + 0.3, f"Margem efetiva ({LARG_EF_P0/2:.2f} m)",
            ha="right", fontsize=9, color="#333", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#999", lw=0.8, alpha=0.9))
    ax.text(480, -LARG_EF_P0/2 - 0.6, f"Margem efetiva ({LARG_EF_P0/2:.2f} m)",
            ha="right", fontsize=9, color="#333", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#999", lw=0.8, alpha=0.9))

    # --- Ponto P0 ---
    ax.scatter([0], [0], s=700, marker="*", color=COR_P0,
               edgecolor="black", linewidth=2.5, zorder=20)
    ax.annotate("P0\n(lançamento)", xy=(0, 0), xytext=(-55, 3),
                fontsize=11, fontweight="bold", color=COR_P0,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=COR_P0, lw=1.5),
                arrowprops=dict(arrowstyle="->", color=COR_P0, lw=1.5))

    # --- Pontos J1 e J2 ---
    for x_j, nome_j in [(97, "J1"), (296.28, "J2")]:
        ax.scatter([x_j], [0], s=250, marker="s", color="#2E8B57",
                   edgecolor="black", linewidth=2, zorder=15)
        ax.annotate(nome_j, xy=(x_j, 0), xytext=(x_j, 3.5),
                    fontsize=11, fontweight="bold", color="#2E8B57",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#2E8B57", lw=1.2))

    # --- Anotações de σ ---
    for x_a in [50, 100, 200, 300, 500]:
        idx = np.argmin(np.abs(x_arr - x_a))
        sy = sig_B[idx]
        ax.annotate(f"σ_y = {sy:.2f} m", xy=(x_a, +2*sy),
                    xytext=(x_a, +2*sy + 0.7), ha="center", fontsize=9,
                    color=COR_P0, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=COR_P0, lw=0.7, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color=COR_P0, lw=0.8))

    ax.set_xlim(-80, 520); ax.set_ylim(-6, 6)
    ax.set_xlabel("Distância a partir de P0 (m)", fontsize=11)
    ax.set_ylabel("Posição lateral no canal (m)", fontsize=11)
    ax.set_title(
        "Figura ZM-2 — Vista em Planta da Pluma de Fósforo Total\n"
        "Cenário B (Q90) — dispersão lateral gaussiana ao longo do trecho",
        fontsize=13, fontweight="bold", pad=14)
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.038)
    cbar.set_label("Fósforo total (mg/L)", fontsize=10.5)
    cbar.ax.axhline(PADROES["P_total"], color=COR_PADRAO, lw=2.5)
    cbar.ax.text(1.5, PADROES["P_total"], "  CONAMA", va="center",
                 fontsize=9, color=COR_PADRAO, fontweight="bold")

    handles = [
        Line2D([0], [0], color=COR_PADRAO, lw=3.5,
               label=f"Padrão CONAMA ({PADROES['P_total']} mg/L)"),
        Line2D([0], [0], color=COR_P0, lw=2.8, ls="--",
               label="Envelope ±2σ_y"),
        Line2D([0], [0], color="#333", lw=2, ls=":",
               label=f"Fundo (M1) = {BG['P_total']} mg/L"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5)

    fig.tight_layout()
    salvar(fig, "fig_ZM_2_planta.png")


# =============================================================================
# FIGURA ZM-3 — CORTES TRANSVERSAIS EM 6 DISTÂNCIAS
# =============================================================================
def fig_zm3():
    """Envelope da pluma em cortes transversais a 0, 50, 100, 200, 300, 500 m."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey=False)

    distancias = [0, 50, 100, 200, 300, 500]
    y_corte = np.linspace(-5, 5, 400)

    for ax, x_corte in zip(axes.flat, distancias):
        sig = sigma_y(np.array([x_corte]), CEN_B["V"])[0]
        C_centro = P0_B["P_total"]  # praticamente constante

        # Perfil lateral
        C_y = BG["P_total"] + (C_centro - BG["P_total"]) * \
              np.exp(-y_corte**2 / (2 * sig**2))

        # Normalizar para plotagem vertical (0 a 1)
        C_norm = (C_y - BG["P_total"]) / max(C_centro - BG["P_total"], 1e-9)

        # Preenchimento da pluma
        ax.fill_between(y_corte, 0, C_norm,
                        color=CMAP_PLUME(0.65), alpha=0.85,
                        edgecolor=COR_B, linewidth=2)
        ax.plot(y_corte, C_norm, color=COR_B, lw=2.5, zorder=3)

        # Envelope ±2σ
        ax.axvline(+2*sig, color=COR_P0, lw=2.5, ls="--", alpha=0.9)
        ax.axvline(-2*sig, color=COR_P0, lw=2.5, ls="--", alpha=0.9)

        # Padrão CONAMA (razão)
        razao_padrao = (PADROES["P_total"] - BG["P_total"]) / \
                       max(C_centro - BG["P_total"], 1e-9)
        if razao_padrao > 0:
            ax.axhline(razao_padrao, color=COR_PADRAO, lw=2.5, ls=":",
                       label=f"CONAMA")

        # Ponto de máximo
        ax.scatter([0], [1.0], s=180, marker="v",
                   color=COR_B, edgecolor="black", linewidth=1.5, zorder=5)

        # Rótulos
        ax.text(0, 1.06, f"C_centro = {C_centro:.3f} mg/L",
                ha="center", fontsize=9.5, fontweight="bold", color=COR_B,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=COR_B, lw=1))
        ax.text(4.8, 0.5, f"σ_y = {sig:.2f} m",
                ha="right", fontsize=10, fontweight="bold", color=COR_P0,
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E1",
                          ec=COR_P0, lw=1.2))

        # Borda do canal
        ax.axvline(+LARG_EF_P0/2, color="#333", lw=1.5, ls="-", alpha=0.4)
        ax.axvline(-LARG_EF_P0/2, color="#333", lw=1.5, ls="-", alpha=0.4)

        ax.set_xlim(-5, 5); ax.set_ylim(-0.05, 1.18)
        ax.set_xlabel("Posição lateral y (m)", fontsize=9.5)
        ax.set_ylabel("Conc. normalizada", fontsize=9.5)
        ax.set_title(f"Corte em x = {x_corte} m a jusante de P0",
                     fontsize=11, pad=8)
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Figura ZM-3 — Cortes Transversais da Pluma — Fósforo Total (Cenário B)\n"
        "Alargamento lateral progressivo conforme teoria de Fischer (1979)",
        fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    salvar(fig, "fig_ZM_3_cortes_transversais.png")


# =============================================================================
# FIGURA ZM-4 — PAINEL COMPARATIVO 3 PARÂMETROS (DBO₅, N-NH₃, P-total)
# =============================================================================
def fig_zm4():
    """3 plumas 2D lado a lado: DBO₅, N-NH₃, P-total."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    parametros = [
        ("DBO5", "DBO₅ (mg/L)", PADROES["DBO5"],
         P0_B["DBO5"], BG["DBO5"], "#B22222", "CMAP_Reds"),
        ("N_NH3", "N-amoniacal (mg/L)", PADROES["N_NH3"],
         P0_B["N_NH3"], BG["N_NH3"], "#2E7D32", "CMAP_Greens"),
        ("P_total", "Fósforo total (mg/L)", PADROES["P_total"],
         P0_B["P_total"], BG["P_total"], "#7B1FA2", "CMAP_Purples"),
    ]

    cmaps = {
        "CMAP_Reds": LinearSegmentedColormap.from_list("reds",
            ["#FFF5F0", "#FCBBA1", "#FC9272", "#EF3B2C", "#A50F15"]),
        "CMAP_Greens": LinearSegmentedColormap.from_list("greens",
            ["#F7FCF5", "#C7E9C0", "#74C476", "#31A354", "#00441B"]),
        "CMAP_Purples": LinearSegmentedColormap.from_list("purples",
            ["#FCFBFD", "#DADAEB", "#9E9AC8", "#6A51A3", "#3F007D"]),
    }

    x_arr = np.linspace(0, 500, 400)
    y_arr = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x_arr, y_arr)

    for ax, (key, ylabel, padrao, c_p0, c_bg, cor_forte, cmap_nome) in zip(
            axes, parametros):

        # Interpolação da concentração central
        # Para DBO5 e P-total: ligeiramente decrescente (autodepuração)
        # Para N-NH3: praticamente constante
        if key == "DBO5":
            x_data = np.array([0, 50, 100, 200, 310, 400, 500])
            c_centro_vals = np.array([c_p0, c_p0-0.005, c_p0-0.01,
                                       c_p0-0.02, c_p0-0.03,
                                       c_p0-0.04, c_p0-0.05])
        else:
            x_data = np.array([0, 500])
            c_centro_vals = np.array([c_p0, c_p0])

        C_center = np.interp(x_arr, x_data, c_centro_vals)

        Z = concentracao_2d(x_arr, y_arr, C_center, c_bg, CEN_B["V"])

        # Heatmap
        levels = np.linspace(min(c_bg, c_p0) * 0.98,
                              max(c_bg, c_p0) * 1.02, 20)
        im = ax.contourf(X, Y, Z, levels=levels,
                          cmap=cmaps[cmap_nome], alpha=0.9, extend="both")

        # Contorno do padrão
        if min(Z.min(), padrao) < padrao < Z.max():
            ax.contour(X, Y, Z, levels=[padrao], colors=[COR_PADRAO],
                       linewidths=3.5, linestyles="-")
        ax.contour(X, Y, Z, levels=[c_bg], colors=["#333"],
                   linewidths=2, linestyles=":")

        # Envelope
        sig_B = sigma_y(x_arr, CEN_B["V"])
        ax.plot(x_arr, +2*sig_B, color="white", lw=2.8, ls="--", alpha=0.9)
        ax.plot(x_arr, -2*sig_B, color="white", lw=2.8, ls="--", alpha=0.9)

        # Ponto P0
        ax.scatter([0], [0], s=400, marker="*", color=COR_P0,
                   edgecolor="black", linewidth=2, zorder=15)

        # Status conforme/não conforme
        status_p0 = "NÃO CONFORME" if c_p0 > padrao else "CONFORME"
        cor_status = COR_PADRAO if c_p0 > padrao else "#2E7D32"

        ax.text(0.985, 0.92,
                f"Padrão C2 = {padrao} mg/L\n"
                f"P0 = {c_p0:.3f} mg/L — {status_p0}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.45", fc=cor_status,
                          ec="black", lw=1.2, alpha=0.9))

        ax.set_xlim(-50, 520); ax.set_ylim(-6, 6)
        ax.set_ylabel("Posição lateral (m)", fontsize=10.5)
        ax.set_aspect("equal", adjustable="box")

        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
        cbar.set_label(ylabel, fontsize=10)

    axes[-1].set_xlabel("Distância a partir de P0 (m)", fontsize=11)

    fig.suptitle(
        "Figura ZM-4 — Comparativo dos Parâmetros Modelados — Zona de Mistura\n"
        "Cenário B (Q90 = 0,745 m³/s) — Vista em planta, dispersão gaussiana",
        fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    salvar(fig, "fig_ZM_4_painel_comparativo.png")


# =============================================================================
# FIGURA ZM-5 — ENVELOPE + COMPRIMENTO DA ZONA DE MISTURA
# =============================================================================
def fig_zm5():
    """Curva de decaimento com delimitação da zona de mistura regulatória."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    parametros = [
        ("DBO5", "DBO₅ (mg/L)", PADROES["DBO5"],
         P0_B["DBO5"], "#B22222", "Cenário A medida", "ext"),
        ("N_NH3", "N-amoniacal (mg/L)", PADROES["N_NH3"],
         P0_B["N_NH3"], "#2E7D32", "N-amoniacal < 0,05 mg/L (abaixo do LQ)", "int"),
        ("P_total", "Fósforo total (mg/L)", PADROES["P_total"],
         P0_B["P_total"], "#7B1FA2", "Conforme desde P0", "int"),
    ]

    for ax, (key, ylabel, padrao, c_p0, cor, nota, tipo) in zip(axes, parametros):

        # Curvas dos 2 cenários
        x_plot = np.linspace(0, 500, 500)

        # Cenário B (Q90) — mais crítica
        if key == "DBO5":
            cB = c_p0 - 0.0002 * x_plot  # decaimento suave
            cA = c_p0 - 0.0003 * x_plot
        else:
            cB = c_p0 - 0.00005 * x_plot
            cA = c_p0 - 0.00008 * x_plot

        # Área de fundo — conforme vs não conforme
        if tipo == "ext" and cB[0] > padrao:
            # DBO5: pré-existente não conforme
            ax.axhspan(padrao, max(cB.max(), cA.max()) * 1.15,
                        color="#FFEBEE", alpha=0.4, zorder=0)
            ax.axhspan(0, padrao, color="#E8F5E9", alpha=0.4, zorder=0)
            label_area = "Não conforme (pré-existente)"
            cor_area = COR_PADRAO
        else:
            ax.axhspan(0, padrao, color="#E8F5E9", alpha=0.4, zorder=0)
            label_area = "Conforme"
            cor_area = "#2E7D32"

        # Linhas com gradiente
        ax.plot(x_plot, cA, color=COR_A, lw=2.5, alpha=0.5, zorder=1)
        ax.plot(x_plot, cB, color=COR_B, lw=2.5, alpha=0.5, zorder=1)

        # Linha do padrão
        ax.axhline(padrao, color=COR_PADRAO, lw=2.5, ls="-.",
                   label=f"Padrão C2 ({padrao} mg/L)", zorder=2)

        # Envelope da zona de mistura
        # (para parâmetros conformes desde P0, a zona tem comprimento nulo)
        if cB[0] <= padrao:
            ax.axvspan(0, 5, color=COR_P0, alpha=0.25, zorder=1)
            ax.annotate(
                f"Zona de mistura: nula\n(conforme desde P0)",
                xy=(2.5, (padrao) * 0.6),
                xytext=(120, padrao * 0.55),
                fontsize=10.5, fontweight="bold", color=COR_P0,
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1",
                          ec=COR_P0, lw=1.5),
                arrowprops=dict(arrowstyle="->", color=COR_P0, lw=1.6))
        else:
            ax.axvspan(0, 500, color="#FFEBEE", alpha=0.25, zorder=1)
            ax.annotate(
                "Não conformidade pré-existente\n(não atribuível ao lançamento)",
                xy=(250, (padrao + cB.max()) / 2),
                xytext=(320, padrao * 1.5),
                fontsize=10.5, fontweight="bold", color=COR_PADRAO,
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE",
                          ec=COR_PADRAO, lw=1.5),
                arrowprops=dict(arrowstyle="->", color=COR_PADRAO, lw=1.6))

        # Ponto P0
        ax.axvline(0, color=COR_P0, lw=2.2, ls="--", alpha=0.9)
        ax.scatter([0], [c_p0], s=180, marker="*", color=COR_P0,
                   edgecolor="black", linewidth=1.8, zorder=10)
        ax.text(0, c_p0, f"  P0\n  {c_p0:.3f} mg/L", fontsize=9.5,
                fontweight="bold", color=COR_P0, va="center")

        # Marcadores de J1, J2
        for x_j, nome_j in [(97, "J1"), (296.28, "J2")]:
            c_val = np.interp(x_j, x_plot, cB)
            ax.scatter([x_j], [c_val], s=120, marker="s",
                       color="#2E8B57", edgecolor="black",
                       linewidth=1.5, zorder=8)
            ax.text(x_j, c_val, f"  {nome_j}", fontsize=9.5,
                    fontweight="bold", color="#2E8B57", va="center")

        ax.set_xlim(-30, 520)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{key} — Zona de Mistura Regulatória", fontsize=11.5,
                     pad=8)

        # Status
        if c_p0 <= padrao:
            status = "✓ CONFORME desde P0"
            cor_s = "#2E7D32"
        else:
            status = "⚠ NÃO CONFORME (pré-existente)"
            cor_s = COR_PADRAO
        ax.text(0.98, 0.90, status, transform=ax.transAxes,
                ha="right", va="top", fontsize=11, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.45", fc=cor_s,
                          ec="black", lw=1.3))

        handles = [
            Line2D([0], [0], color=COR_A, lw=2.5, label="Cenário A"),
            Line2D([0], [0], color=COR_B, lw=2.5, ls="--", label="Cenário B"),
            Line2D([0], [0], color=COR_PADRAO, lw=2.5, ls="-.",
                   label=f"Padrão C2 ({padrao} mg/L)"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9)

    axes[-1].set_xlabel("Distância a partir de P0 (m)", fontsize=11)
    fig.suptitle(
        "Figura ZM-5 — Envelope da Zona de Mistura Regulatória (CONAMA 430/2011)\n"
        "Cenário B (Q90 = 0,745 m³/s)  ·  Extensão ao longo do trecho",
        fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    salvar(fig, "fig_ZM_5_envelope_conformidade.png")


# =============================================================================
# FIGURA ZM-6 — HTML INTERATIVO (Plotly)
# =============================================================================
def fig_zm6_html():
    """Gera HTML interativo com Plotly para rotação 3D."""

    try:
        import plotly.graph_objects as go

        x_arr = np.linspace(0, 500, 100)
        y_arr = np.linspace(-8, 8, 50)
        X, Y = np.meshgrid(x_arr, y_arr)

        C_center = np.full_like(x_arr, P0_B["P_total"])
        Z = concentracao_2d(x_arr, y_arr, C_center, BG["P_total"], CEN_B["V"])

        # Pluma 3D
        fig = go.Figure()

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0.0, "#F7FCF5"], [0.15, "#C7E9C0"], [0.35, "#74C476"],
                [0.55, "#FEB24C"], [0.75, "#F03B20"], [1.0, "#8B2635"],
            ],
            colorbar=dict(title="P-total (mg/L)", thickness=20, len=0.6),
            hovertemplate="<b>x</b>=%{x:.1f} m<br>"
                          "<b>y</b>=%{y:.2f} m<br>"
                          "<b>P</b>=%{z:.4f} mg/L<extra></extra>",
            opacity=0.95,
            showscale=True,
            name="Pluma P-total",
        ))

        # Plano do padrão CONAMA
        fig.add_trace(go.Surface(
            x=X, y=Y, z=np.full_like(X, PADROES["P_total"]),
            colorscale=[[0, "rgba(139,38,53,0.35)"],
                        [1, "rgba(139,38,53,0.35)"]],
            showscale=False,
            name=f"Padrão CONAMA ({PADROES['P_total']} mg/L)",
            hovertemplate="Padrão CONAMA: 0,10 mg/L<extra></extra>",
        ))

        # Ponto P0
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[P0_B["P_total"]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color="#E5A54B",
                        line=dict(color="black", width=2)),
            text=["P0 (lançamento)"],
            textposition="top center",
            textfont=dict(size=12, color="black"),
            name="P0",
            hovertemplate="<b>P0</b><br>Fósforo total: %{z:.4f} mg/L<extra></extra>",
        ))

        # Envelope ±2σ_y
        sig_B = sigma_y(x_arr, CEN_B["V"])
        fig.add_trace(go.Scatter3d(
            x=x_arr, y=+2*sig_B, z=np.full_like(x_arr, 0.155),
            mode="lines",
            line=dict(color="#E5A54B", width=5, dash="dash"),
            name="+2σ_y",
        ))
        fig.add_trace(go.Scatter3d(
            x=x_arr, y=-2*sig_B, z=np.full_like(x_arr, 0.155),
            mode="lines",
            line=dict(color="#E5A54B", width=5, dash="dash"),
            name="−2σ_y",
        ))

        fig.update_layout(
            title=dict(
                text="🌊 Zona de Mistura — Pluma 3D de Fósforo Total<br>"
                     "<sub>Rio Araçatuba · Cenário B (Q90 = 0,745 m³/s) · "
                     "Padrão CONAMA = 0,10 mg/L</sub>",
                x=0.5, xanchor="center", font=dict(size=17),
            ),
            scene=dict(
                xaxis_title="Distância de P0 (m)",
                yaxis_title="Posição lateral (m)",
                zaxis_title="P-total (mg/L)",
                camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.3)),
                aspectratio=dict(x=1.4, y=0.8, z=0.5),
                bgcolor="white",
            ),
            height=760,
            margin=dict(l=20, r=20, t=110, b=20),
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="black", borderwidth=1),
        )

        p = OUTPUT / "fig_ZM_6_interativo.html"
        fig.write_html(str(p), include_plotlyjs="cdn")
        kb = p.stat().st_size / 1024
        print(f"  [OK] fig_ZM_6_interativo.html  ({kb:.0f} KB)")

    except ImportError:
        print("  [AVISO] Plotly não instalado — pulando HTML interativo")
        print("         Instale com: pip install plotly")
    except Exception as e:
        print(f"  [ERRO HTML] {e}")
        traceback.print_exc()


# =============================================================================
# EXECUÇÃO
# =============================================================================
print("=" * 70)
print("  ZONA DE MISTURA — VISUALIZAÇÃO AVANÇADA")
print("=" * 70)

rodar(fig_zm1)
rodar(fig_zm2)
rodar(fig_zm3)
rodar(fig_zm4)
rodar(fig_zm5)
rodar(fig_zm6_html)

# Relatório
print("\n" + "=" * 70)
print("  RELATÓRIO FINAL")
print("=" * 70)
arqs = sorted(OUTPUT.glob("*"))
for a in arqs:
    kb = a.stat().st_size / 1024
    print(f"    ✓ {a.name}  ({kb:.0f} KB)")
print("=" * 70)
print(f"  Total: {len(arqs)} arquivos")
print(f"  Pasta: {OUTPUT}")
print("=" * 70)
print("\n  ✓ Pronto! Abra os PNGs em 100% de zoom para ver os detalhes.")
print("  ✓ O HTML abre em qualquer navegador (rotação 3D interativa).")