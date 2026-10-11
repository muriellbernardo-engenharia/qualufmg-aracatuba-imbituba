# -*- coding: utf-8 -*-
"""
=============================================================================
PERFIL DO RIO COM PLUMA DE MISTURA — VISUALIZAÇÃO CINEMATOGRÁFICA
Rio Araçatuba — ETE Residencial Imbituba
=============================================================================
Gera 4 figuras que mostram o rio visto de lado, com a pluma do efluente
se misturando em tempo real, usando gradiente de cor para representar
a intensidade da concentração.

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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Polygon, FancyArrowPatch, Ellipse
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "output_perfil_rio"
OUTPUT.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Saída: {OUTPUT}\n")

# =============================================================================
# ESTILO
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.titlepad": 14,
    "axes.edgecolor": "#2B2B2B", "axes.linewidth": 1.0,
    "axes.grid": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "legend.fontsize": 9,
    "figure.dpi": 120, "savefig.dpi": 400,
    "savefig.facecolor": "white", "savefig.bbox": "tight",
})

# Paletas
COR_RIO_AGUA = "#4A90E2"
COR_RIO_FUNDO = "#1B3A5C"
COR_LEITO = "#3E2723"
COR_MARGEM = "#6D4C41"
COR_VEG = "#2E7D32"
COR_P0 = "#E5A54B"
COR_PADRAO = "#8B2635"
COR_EFLUENTE = "#D35400"

# Colormaps ricos em gradiente
CMAP_MISTURA = LinearSegmentedColormap.from_list("mistura", [
    "#F0F9FF",   # água limpa
    "#BAE6FD",   # leve turbidez
    "#7DD3FC",
    "#FDE68A",   # transição
    "#FB923C",   # pluma ativa
    "#EA580C",
    "#991B1B",   # concentração alta
])
CMAP_TERMICO = LinearSegmentedColormap.from_list("termico", [
    "#0A1F44", "#1E40AF", "#3B82F6", "#93C5FD",
    "#FCD34D", "#FB923C", "#DC2626", "#7F1D1D",
])
CMAP_PLUVIA = LinearSegmentedColormap.from_list("pluvia", [
    "#F0FDF4", "#DCFCE7", "#86EFAC", "#4ADE80",
    "#FACC15", "#FB923C", "#EF4444", "#7F1D1D",
])


# =============================================================================
# DADOS DO ESTUDO
# =============================================================================
# Geometria do rio (dados de campo — Tabelas 7.2A, 7.2C)
PERFIL = dict(
    dist=[0.00, 100.73, 214.23, 311.23, 510.51],
    cota_terreno=[4.22, 4.20, 4.18, 4.16, 4.11],
    cota_leito=[1.70, 1.55, 2.03, 1.61, 1.89],
)

# Seções transversais
SECOES = dict(
    M2=dict(dist=-214.23, L=9.60, h_max=2.52),
    M1=dict(dist=-113.50, L=13.60, h_max=2.65),
    P0=dict(dist=0.0, L=8.18, h_max=2.15),
    J1=dict(dist=97.00, L=16.00, h_max=2.55),
    J2=dict(dist=296.28, L=7.00, h_max=2.22),
)

# Parâmetros (Cenário B — Q90)
CEN_B = dict(Q=0.745, V=0.05253, label="Q90 = 0,745 m³/s")

# Concentrações após mistura em P0 (Cenário B)
C_P0 = dict(
    P_total=0.089,     # mg/L (conforme desde P0)
    N_NH3=0.050,       # mg/L (conforme desde P0)
    N_NO3=0.475,       # mg/L (conforme desde P0)
    DBO5=20.41,        # mg/L (pré-existente)
    OD=6.49,           # mg/L
    Coli=80.9,         # NMP/100 mL
)

# Concentrações de fundo (montante M1)
BG = dict(P_total=0.08, N_NH3=0.05, N_NO3=0.40, DBO5=20.40,
          OD=6.50, Coli=78.0)

# Padrões CONAMA 357/2005 — Classe 2
PADROES = dict(DBO5=5.0, N_NH3=2.0, N_NO2=1.0, N_NO3=10.0,
               P_total=0.10, OD=5.0, Coli=1000.0)

# Parâmetro principal da análise
PARAM_FOCO = "P_total"
PARAM_LABEL = "Fósforo total (mg/L)"
PARAM_PADRAO = PADROES[PARAM_FOCO]


# =============================================================================
# MODELO DE DISPERSÃO
# =============================================================================
def sigma_y(x, U, h=1.78, S=0.000215, sigma0=0.5):
    """Crescimento lateral da pluma (Fischer, 1979)."""
    u_star = np.sqrt(9.81 * h * S)
    eps_y = 0.15 * h * u_star
    x_pos = np.maximum(x, 0)
    return np.sqrt(sigma0**2 + 2 * eps_y * x_pos / U)


def sigma_z(x, U, h=1.78, S=0.000215, sigma_z0=0.3):
    """Crescimento vertical da pluma (dobro do lateral em canais rasos)."""
    return sigma_y(x, U, h, S, sigma_z0) * 1.8


def concentracao(x, y, z, C_centro, C_bg, U):
    """Concentração 3D em (x, y, z) — gaussiana lateral + vertical."""
    sig_y = sigma_y(x, U)
    sig_z = sigma_z(x, U)
    sig_y = np.where(sig_y < 1e-6, 1e-6, sig_y)
    sig_z = np.where(sig_z < 1e-6, 1e-6, sig_z)
    lateral = np.exp(-y**2 / (2 * sig_y**2))
    vertical = np.exp(-z**2 / (2 * sig_z**2))
    return C_bg + (C_centro - C_bg) * lateral * vertical


def leito_do_rio(x_arr):
    """Cota do leito interpolada (Tabela 7.2C) — usa coordenadas do perfil."""
    return np.interp(x_arr, PERFIL["dist"], PERFIL["cota_leito"])


def terreno(x_arr):
    """Cota do terreno interpolada (Tabela 7.2C)."""
    return np.interp(x_arr, PERFIL["dist"], PERFIL["cota_terreno"])


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
# FIGURA RP-1 — PERFIL LONGITUDINAL DO RIO COM PLUMA COLORIDA
# =============================================================================
def fig_rp1():
    """
    Vista lateral do rio (corte longitudinal). O leito e o terreno são
    desenhados com curvas reais. Dentro da coluna d'água, uma malha
    colorida mostra a intensidade da pluma de fósforo total.
    """

    fig, ax = plt.subplots(figsize=(16, 7))

    # --- Coordenadas: usar distância a partir de M2 (0 a 510,51 m) ---
    x = np.linspace(0, 510.51, 500)
    cota_terr = terreno(x)
    cota_leito = leito_do_rio(x)

    # --- Preenchimento do terreno acima do leito (com gradiente) ---
    for i in range(15):
        t0 = i / 15
        t1 = (i + 1) / 15
        y0 = cota_leito + (cota_terr - cota_leito) * (1 - t0)
        y1 = cota_leito + (cota_terr - cota_leito) * (1 - t1)
        cor = plt.cm.copper(0.3 + 0.5 * t0)
        ax.fill_between(x, y0, y1, color=cor, alpha=0.5, linewidth=0, zorder=1)

    # --- Leito ---
    ax.plot(x, cota_leito, color=COR_LEITO, lw=4, zorder=5,
            label="Leito do rio")
    ax.plot(x, cota_terr, color="#6D4C41", lw=2.5, zorder=4,
            label="Terreno", alpha=0.8)

    # --- PLUMA dentro da água (malha 2D) ---
    # Para cada x, distribuir verticalmente a concentração
    nz = 60
    Z_linhas = np.linspace(0, 1, nz)  # fração de profundidade (0=superfície, 1=leito)
    profundidade_max = 2.5
    z_valores = Z_linhas * profundidade_max

    # Malha de concentração
    X_grid, Z_grid = np.meshgrid(x, z_valores)

    # Concentração central ao longo de x (constante após P0, decaindo levemente)
    x_relativo = x - 214.23  # distância a partir de P0
    C_centro_x = np.where(
        x_relativo < 0,
        BG[PARAM_FOCO],
        C_P0[PARAM_FOCO] - 0.00002 * np.maximum(x_relativo, 0)
    )

    # Pluma: concentração cai da superfície para o fundo
    C_grid = np.zeros_like(X_grid)
    for j, xv in enumerate(x):
        xr = x_relativo[j]
        if xr < 0:
            C_grid[:, j] = BG[PARAM_FOCO]
        else:
            sig_z_local = sigma_z(np.array([xr]), CEN_B["V"])[0]
            decay_vertical = np.exp(-(z_valores**2) / (2 * sig_z_local**2))
            C_grid[:, j] = BG[PARAM_FOCO] + (C_centro_x[j] - BG[PARAM_FOCO]) * decay_vertical

    # Cota da água em cada x (superfície)
    cota_sup = cota_leito + profundidade_max * 0.95

    # Y das linhas na água (para o contourf — precisa ser grid)
    Y_grid = cota_sup[np.newaxis, :] - Z_grid  # cota absoluta

    # --- Heatmap da pluma ---
    levels = np.linspace(BG[PARAM_FOCO], PARAM_PADRAO * 1.05, 22)
    cf = ax.contourf(X_grid, Y_grid, C_grid, levels=levels,
                     cmap=CMAP_MISTURA, alpha=0.92, zorder=3,
                     extend="both")

    # --- Linhas de contorno (iso-concentração) ---
    ax.contour(X_grid, Y_grid, C_grid,
               levels=[BG[PARAM_FOCO], 0.085, PARAM_PADRAO],
               colors=["#333", "white", COR_PADRAO],
               linewidths=[1.5, 2, 3.5], linestyles=[":", "--", "-"],
               zorder=6)

    # --- Superfície da água (linha) ---
    ax.plot(x, cota_sup, color=COR_RIO_AGUA, lw=2.5,
            zorder=4, alpha=0.9)

    # --- Ponto P0 (lançamento) ---
    x_p0 = 214.23
    cota_p0_sup = cota_leito[np.argmin(np.abs(x - x_p0))] + profundidade_max * 0.95
    ax.scatter([x_p0], [cota_p0_sup], s=500, marker="*",
               color=COR_P0, edgecolor="black", linewidth=2.5, zorder=20,
               label="P0 (lançamento)")
    ax.annotate("P0\n(lançamento)", xy=(x_p0, cota_p0_sup),
                xytext=(x_p0 - 60, cota_p0_sup + 1.2),
                fontsize=11, fontweight="bold", color=COR_P0,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=COR_P0, lw=1.5),
                arrowprops=dict(arrowstyle="->", color=COR_P0, lw=1.5))

    # --- Efluente entrando (seta laranja) ---
    ax.annotate("", xy=(x_p0, cota_p0_sup - 0.3),
                xytext=(x_p0 - 30, cota_p0_sup + 0.8),
                arrowprops=dict(arrowstyle="->", color=COR_EFLUENTE,
                                lw=3.5, mutation_scale=25),
                zorder=15)
    ax.text(x_p0 - 35, cota_p0_sup + 0.9, "Efluente ETE\n22,4 mg/L DBO₅",
            fontsize=9.5, fontweight="bold", color=COR_EFLUENTE,
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0",
                      ec=COR_EFLUENTE, lw=1.2))

    # --- Marcadores M2, M1, J1, J2 ---
    for nome, dist_rel in [("M2", -214.23), ("M1", -113.50),
                            ("J1", 97.00), ("J2", 296.28)]:
        x_pt = dist_rel + 214.23  # posição no gráfico
        cota_pt = cota_leito[np.argmin(np.abs(x - x_pt))]
        ax.scatter([x_pt], [cota_pt + 0.15], s=140, marker="o",
                   color="white", edgecolor="#333", linewidth=2, zorder=15)
        ax.text(x_pt, cota_pt + 0.15, nome, fontsize=9.5,
                fontweight="bold", color="#333", ha="center", va="center",
                zorder=16)
        # Linha vertical tracejada
        ax.axvline(x_pt, color="#999", lw=0.8, ls=":", alpha=0.5,
                   zorder=2, ymin=0.0, ymax=1.0)

    # --- Texto descritivo ---
    ax.text(0.02, 0.97,
            f"Perfil Longitudinal do Rio Araçatuba\n"
            f"Pluma de {PARAM_LABEL.split('(')[0].strip()} — Cenário B ({CEN_B['label']})",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=13, fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.6", fc="white",
                      ec="#999", lw=1.3, alpha=0.95))

    # --- Legenda ---
    handles = [
        Line2D([0], [0], color=COR_LEITO, lw=3, label="Leito do rio"),
        Line2D([0], [0], color=COR_RIO_AGUA, lw=2.5, label="Superfície da água"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COR_P0,
               markersize=20, markeredgecolor="black",
               label="P0 (lançamento)"),
        Line2D([0], [0], color=COR_PADRAO, lw=3, ls="-",
               label=f"Padrão CONAMA ({PARAM_PADRAO} mg/L)"),
        Line2D([0], [0], color="#333", lw=2, ls=":",
               label=f"Fundo (M1) = {BG[PARAM_FOCO]} mg/L"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10,
              facecolor="white", framealpha=0.95)

    # --- Barra de cores ---
    cbar = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.028, aspect=22)
    cbar.set_label(PARAM_LABEL, fontsize=10.5)
    cbar.ax.axhline(PARAM_PADRAO, color=COR_PADRAO, lw=3)
    cbar.ax.text(1.5, PARAM_PADRAO, "  Padrão\n  CONAMA",
                 va="center", fontsize=9, color=COR_PADRAO,
                 fontweight="bold")

    ax.set_xlim(-5, 520)
    ax.set_ylim(1.1, 5.0)
    ax.set_xlabel("Distância a partir de M2 (m)", fontsize=11.5)
    ax.set_ylabel("Cota (m)", fontsize=11.5)
    ax.set_title(
        "Figura RP-1 — Perfil Longitudinal do Rio com Pluma de Mistura\n"
        "Intensidade da concentração do efluente ao longo do trecho",
        fontsize=13.5, fontweight="bold", pad=14)

    fig.tight_layout()
    salvar(fig, "fig_RP_1_perfil_pluma.png")


# =============================================================================
# FIGURA RP-2 — VISTA COMBINADA: PLANTA + PERFIL (2 vistas)
# =============================================================================
def fig_rp2():
    """
    Duas vistas lado a lado:
      (a) Planta (topo) — pluma se espalhando lateralmente
      (b) Perfil (lado) — pluma se dispersando verticalmente
    """

    fig = plt.figure(figsize=(16, 12))

    # ---------- VISTA A: PLANTA ----------
    ax1 = fig.add_subplot(2, 1, 1)

    x_arr = np.linspace(-100, 500, 400)
    y_arr = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x_arr, y_arr)

    # Concentração central
    x_rel = X
    C_center = np.where(x_rel < 0, BG[PARAM_FOCO],
                        C_P0[PARAM_FOCO])
    C_center = np.broadcast_to(C_center, X.shape).copy()

    # Aplicar decaimento
    sig_Y = sigma_y(np.maximum(x_rel, 0), CEN_B["V"])
    lateral = np.exp(-Y**2 / (2 * sig_Y**2))
    Z = BG[PARAM_FOCO] + (C_center - BG[PARAM_FOCO]) * lateral

    # Heatmap
    levels = np.linspace(BG[PARAM_FOCO], PARAM_PADRAO * 1.05, 25)
    cf1 = ax1.contourf(X, Y, Z, levels=levels, cmap=CMAP_MISTURA,
                       alpha=0.95, extend="both")

    # Contorno do padrão
    ax1.contour(X, Y, Z, levels=[PARAM_PADRAO], colors=[COR_PADRAO],
                linewidths=3.5, linestyles="-")

    # Envelope
    ax1.plot(x_arr, +2*sig_Y[0, :], color=COR_P0, lw=2.5, ls="--",
             alpha=0.85, label="Envelope ±2σ_y")
    ax1.plot(x_arr, -2*sig_Y[0, :], color=COR_P0, lw=2.5, ls="--",
             alpha=0.85)
    ax1.fill_between(x_arr, -2*sig_Y[0, :], +2*sig_Y[0, :],
                     color=COR_P0, alpha=0.1)

    # Margens
    ax1.axhline(+4.09, color="#333", lw=2, ls="-", alpha=0.5)
    ax1.axhline(-4.09, color="#333", lw=2, ls="-", alpha=0.5)
    ax1.text(495, 4.2, "Margem esquerda", ha="right", fontsize=9,
             color="#333", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="#999", lw=0.8, alpha=0.9))
    ax1.text(495, -4.7, "Margem direita", ha="right", fontsize=9,
             color="#333", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="#999", lw=0.8, alpha=0.9))

    # P0
    ax1.scatter([0], [0], s=600, marker="*", color=COR_P0,
                edgecolor="black", linewidth=2.5, zorder=15)
    ax1.annotate("P0 (lançamento)", xy=(0, 0), xytext=(-60, 4.5),
                 fontsize=11, fontweight="bold", color=COR_P0,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec=COR_P0, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=COR_P0, lw=1.5))

    # Pontos
    for nome, x_pt in [("M2", -214.23), ("M1", -113.50),
                        ("J1", 97.00), ("J2", 296.28)]:
        ax1.scatter([x_pt], [0], s=150, marker="s", color="#2E8B57",
                    edgecolor="black", linewidth=1.5, zorder=12)
        ax1.text(x_pt, 0, f" {nome}", fontsize=10, fontweight="bold",
                 color="#2E8B57", va="center")

    ax1.set_xlim(-120, 520)
    ax1.set_ylim(-6, 6)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("Distância a partir de P0 (m)", fontsize=11)
    ax1.set_ylabel("Posição lateral (m)", fontsize=11)
    ax1.set_title("(a) VISTA EM PLANTA — Dispersão lateral da pluma",
                  fontsize=12, fontweight="bold", pad=10)

    cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.015, fraction=0.03, aspect=20)
    cbar1.set_label(PARAM_LABEL, fontsize=10)
    cbar1.ax.axhline(PARAM_PADRAO, color=COR_PADRAO, lw=3)

    # ---------- VISTA B: PERFIL ----------
    ax2 = fig.add_subplot(2, 1, 2)

    x_pr = np.linspace(0, 510.51, 400)
    terr = terreno(x_pr)
    leito = leito_do_rio(x_pr)

    # Fundo do terreno
    ax2.fill_between(x_pr, terr, 5.5, color="#8B7355", alpha=0.4,
                     label="Terreno")
    ax2.plot(x_pr, terr, color="#6D4C41", lw=2, alpha=0.8)

    # Leito
    ax2.plot(x_pr, leito, color=COR_LEITO, lw=4, zorder=5)

    # Pluma dentro da água
    nz = 50
    Z_linhas = np.linspace(0, 1, nz)
    prof_max = 2.5
    z_vals = Z_linhas * prof_max
    X_g, Z_g = np.meshgrid(x_pr, z_vals)
    x_relativo = x_pr - 214.23

    C_g = np.zeros_like(X_g)
    for j, xv in enumerate(x_pr):
        xr = x_relativo[j]
        if xr < 0:
            C_g[:, j] = BG[PARAM_FOCO]
        else:
            sig_z_local = sigma_z(np.array([xr]), CEN_B["V"])[0]
            dv = np.exp(-(z_vals**2) / (2 * sig_z_local**2))
            c_centro = C_P0[PARAM_FOCO]
            C_g[:, j] = BG[PARAM_FOCO] + (c_centro - BG[PARAM_FOCO]) * dv

    cota_sup = leito + prof_max * 0.95
    Y_g = cota_sup[np.newaxis, :] - Z_g

    cf2 = ax2.contourf(X_g, Y_g, C_g, levels=levels, cmap=CMAP_MISTURA,
                       alpha=0.95, extend="both", zorder=3)
    ax2.plot(x_pr, cota_sup, color=COR_RIO_AGUA, lw=2.5, zorder=4)

    # Contornos
    ax2.contour(X_g, Y_g, C_g, levels=[PARAM_PADRAO],
                colors=[COR_PADRAO], linewidths=3, linestyles="-", zorder=6)

    # P0
    x_p0 = 214.23
    cota_p0 = leito[np.argmin(np.abs(x_pr - x_p0))] + prof_max * 0.95
    ax2.scatter([x_p0], [cota_p0], s=450, marker="*", color=COR_P0,
                edgecolor="black", linewidth=2.5, zorder=20)

    # Marcadores
    for nome, dist_rel in [("M2", -214.23), ("M1", -113.50),
                            ("J1", 97.00), ("J2", 296.28)]:
        x_pt = dist_rel + 214.23
        ax2.scatter([x_pt], [leito[np.argmin(np.abs(x_pr - x_pt))] + 0.15],
                    s=120, marker="o", color="white",
                    edgecolor="#333", linewidth=2, zorder=15)
        ax2.text(x_pt, leito[np.argmin(np.abs(x_pr - x_pt))] + 0.15, nome,
                 fontsize=9, fontweight="bold", color="#333",
                 ha="center", va="center", zorder=16)

    ax2.set_xlim(-5, 520)
    ax2.set_ylim(1.1, 5.0)
    ax2.set_xlabel("Distância a partir de M2 (m)", fontsize=11)
    ax2.set_ylabel("Cota (m)", fontsize=11)
    ax2.set_title("(b) VISTA EM PERFIL — Dispersão vertical da pluma",
                  fontsize=12, fontweight="bold", pad=10)

    cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.015, fraction=0.03, aspect=20)
    cbar2.set_label(PARAM_LABEL, fontsize=10)
    cbar2.ax.axhline(PARAM_PADRAO, color=COR_PADRAO, lw=3)

    fig.suptitle(
        "Figura RP-2 — Vista Combinada (Planta + Perfil) da Zona de Mistura\n"
        f"Parâmetro: {PARAM_LABEL} — Cenário B ({CEN_B['label']})",
        fontsize=14, fontweight="bold", y=0.995)

    fig.tight_layout()
    salvar(fig, "fig_RP_2_planta_e_perfil.png")


# =============================================================================
# FIGURA RP-3 — DISPERSÃO 3D COM LEITO E PLUMA VOLUMÉTRICA
# =============================================================================
def fig_rp3():
    """
    Vista 3D completa: leito do rio + superfície + pluma volumétrica.
    A pluma aparece como nuvem semi-transparente dentro da água.
    """

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ---------- Leito do rio (superfície 3D) ----------
    x_leito = np.linspace(0, 510.51, 80)
    y_leito = np.linspace(-4.5, 4.5, 30)
    X_l, Y_l = np.meshgrid(x_leito, y_leito)

    # Cota do leito + variação lateral (calha central mais profunda)
    cota_base = leito_do_rio(x_leito)
    # Perfil transversal: calha central
    perfil_y = 0.5 * (Y_l**2) / 4.5  # mais fundo no centro
    Z_leito = cota_base[np.newaxis, :] + perfil_y * 0.3

    ax.plot_surface(X_l, Y_l, Z_leito,
                     cmap=plt.cm.copper, alpha=0.75,
                     edgecolor="none", rstride=2, cstride=2,
                     linewidth=0.2, antialiased=True)

    # ---------- Pluma volumétrica ----------
    nx, ny, nz = 60, 20, 15
    x_p = np.linspace(0, 510.51, nx)
    y_p = np.linspace(-3, 3, ny)
    z_p = np.linspace(0, 2.5, nz)  # altura da coluna d'água (0=fundo)
    X_p, Y_p, Z_p = np.meshgrid(x_p, y_p, z_p, indexing="ij")

    # Concentração em cada ponto
    x_rel = x_p - 214.23
    C_plume = np.zeros((nx, ny, nz))
    for i, xr in enumerate(x_rel):
        if xr < 0:
            C_plume[i, :, :] = BG[PARAM_FOCO]
        else:
            sig_y_local = sigma_y(np.array([xr]), CEN_B["V"])[0]
            sig_z_local = sigma_z(np.array([xr]), CEN_B["V"])[0]
            lateral = np.exp(-y_p**2 / (2 * sig_y_local**2))
            vertical = np.exp(-z_p**2 / (2 * sig_z_local**2))
            C_plume[i, :, :] = BG[PARAM_FOCO] + \
                (C_P0[PARAM_FOCO] - BG[PARAM_FOCO]) * \
                lateral[:, np.newaxis] * vertical[np.newaxis, :]

    # Cota 3D de cada ponto da pluma (fundo do rio + z)
    Z_cota = Z_leito[np.linspace(0, Z_leito.shape[0]-1, nx).astype(int),
                     np.linspace(0, Z_leito.shape[1]-1, ny).astype(int)][:, :, np.newaxis]
    Z_cota = Z_cota + Z_p
    Z_cota = np.broadcast_to(Z_cota, X_p.shape)

    # Visualizar a pluma como pontos (scatter volumétrico)
    # Filtrar apenas os pontos com concentração acima do fundo
    mask = C_plume > BG[PARAM_FOCO] * 1.02

    x_pts = X_p[mask]
    y_pts = Y_p[mask]
    z_pts = Z_cota[mask]
    c_pts = C_plume[mask]

    sc = ax.scatter(x_pts, y_pts, z_pts,
                     c=c_pts, cmap=CMAP_MISTURA,
                     s=8, alpha=0.6,
                     vmin=BG[PARAM_FOCO], vmax=0.12,
                     edgecolor="none",
                     depthshade=True)

    # ---------- Superfície da água (linha) ----------
    x_sup = np.linspace(0, 510.51, 80)
    y_sup = np.linspace(-4.5, 4.5, 30)
    X_s, Y_s = np.meshgrid(x_sup, y_sup)
    cota_sup = leito_do_rio(x_sup) + 2.5
    Z_s = np.tile(cota_sup, (len(y_sup), 1))

    ax.plot_wireframe(X_s, Y_s, Z_s,
                       color=COR_RIO_AGUA, linewidth=0.8, alpha=0.35,
                       rstride=2, cstride=2)

    # ---------- Ponto P0 ----------
    x_p0 = 214.23
    cota_p0 = np.interp(x_p0, PERFIL["dist"], PERFIL["cota_leito"]) + 2.4
    ax.scatter([x_p0], [0], [cota_p0], s=500, marker="*",
               color=COR_P0, edgecolor="black", linewidth=2.5, zorder=20)
    ax.text(x_p0, 0, cota_p0 + 0.4, "P0", fontsize=12,
            fontweight="bold", color=COR_P0, ha="center")

    # ---------- Eixos e título ----------
    ax.set_xlabel("Distância a partir de M2 (m)", fontsize=11, labelpad=12)
    ax.set_ylabel("Posição lateral (m)", fontsize=11, labelpad=12)
    ax.set_zlabel("Cota (m)", fontsize=11, labelpad=12)

    ax.set_xlim(0, 520)
    ax.set_ylim(-5, 5)
    ax.set_zlim(1.0, 5.0)
    ax.view_init(elev=25, azim=-65)

    ax.set_title(
        "Figura RP-3 — Dispersão 3D Completa — Rio Araçatuba\n"
        f"{PARAM_LABEL} — Cenário B ({CEN_B['label']}) — Nuvem volumétrica",
        fontsize=13.5, fontweight="bold", pad=22)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08, aspect=22)
    cbar.set_label(PARAM_LABEL, fontsize=10.5)
    cbar.ax.axhline(PARAM_PADRAO, color=COR_PADRAO, lw=3)

    fig.tight_layout()
    salvar(fig, "fig_RP_3_dispersao_3d.png")


# =============================================================================
# FIGURA RP-4 — PLUMA COM MÚLTIPLAS ISO-CONCENTRAÇÕES E BADGE
# =============================================================================
def fig_rp4():
    """
    Corte longitudinal com múltiplas iso-concentrações (efeito meteorológico),
    mostrando a "assinatura" da pluma em perfis sucessivos.
    """

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    parametros = [
        ("P_total", "Fósforo total (mg/L)", PADROES["P_total"],
         C_P0["P_total"], BG["P_total"], "#7B1FA2",
         CMAP_MISTURA, "Conforme desde P0"),
        ("N_NH3", "N-amoniacal (mg/L)", PADROES["N_NH3"],
         C_P0["N_NH3"], BG["N_NH3"], "#2E7D32",
         plt.cm.Greens, "Conforme desde P0"),
        ("N_NO3", "N-nitrato (mg/L)", PADROES["N_NO3"],
         C_P0["N_NO3"], BG["N_NO3"], "#1565C0",
         plt.cm.Blues, "Conforme desde P0"),
    ]

    x = np.linspace(0, 510.51, 400)
    terr = terreno(x)
    leito = leito_do_rio(x)

    nz = 50
    z_vals = np.linspace(0, 1, nz) * 2.5
    X_g, Z_g = np.meshgrid(x, z_vals)
    x_rel = x - 214.23

    for ax, (key, ylabel, padrao, c_p0, c_bg, cor_forte,
             cmap, status) in zip(axes, parametros):

        # Terreno
        ax.fill_between(x, terr, 5.5, color="#8B7355", alpha=0.35)
        ax.plot(x, terr, color="#6D4C41", lw=1.8, alpha=0.7)
        ax.plot(x, leito, color=COR_LEITO, lw=3.5, zorder=5)

        # Concentração
        C_g = np.zeros_like(X_g)
        for j, xv in enumerate(x):
            xr = x_rel[j]
            if xr < 0:
                C_g[:, j] = c_bg
            else:
                sig_z_local = sigma_z(np.array([xr]), CEN_B["V"])[0]
                dv = np.exp(-(z_vals**2) / (2 * sig_z_local**2))
                C_g[:, j] = c_bg + (c_p0 - c_bg) * dv

        cota_sup = leito + 2.5 * 0.95
        Y_g = cota_sup[np.newaxis, :] - Z_g

        # Heatmap
        cmin, cmax = c_bg * 0.98, max(padrao, c_p0) * 1.05
        levels = np.linspace(cmin, cmax, 25)

        cf = ax.contourf(X_g, Y_g, C_g, levels=levels, cmap=cmap,
                          alpha=0.92, extend="both", zorder=3)

        # Iso-concentrações múltiplas
        iso_levels = np.linspace(c_bg, max(c_p0, padrao), 6)
        cs = ax.contour(X_g, Y_g, C_g, levels=iso_levels,
                         colors=["white"], linewidths=1.2,
                         linestyles="--", alpha=0.75, zorder=6)

        # Contorno do padrão
        if c_bg < padrao < c_p0 * 1.2:
            ax.contour(X_g, Y_g, C_g, levels=[padrao],
                        colors=[COR_PADRAO], linewidths=3.5,
                        linestyles="-", zorder=7)

        # Superfície
        ax.plot(x, cota_sup, color=COR_RIO_AGUA, lw=2.2, zorder=4)

        # P0
        x_p0 = 214.23
        cota_p0 = leito[np.argmin(np.abs(x - x_p0))] + 2.4
        ax.scatter([x_p0], [cota_p0], s=400, marker="*",
                    color=COR_P0, edgecolor="black", linewidth=2, zorder=15)

        # Badge de status
        ax.text(0.985, 0.90, status, transform=ax.transAxes,
                ha="right", va="top", fontsize=11.5, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.45", fc="#2E7D32",
                          ec="black", lw=1.3))

        # Info da concentração
        ax.text(0.02, 0.90,
                f"Fundo = {c_bg} mg/L\nP0 = {c_p0:.3f} mg/L\nPadrão = {padrao} mg/L",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#999", lw=1))

        ax.set_xlim(-5, 520)
        ax.set_ylim(1.1, 5.0)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{key}", fontsize=12, pad=8)

        cbar = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.026)
        cbar.set_label(ylabel, fontsize=9.5)
        if c_bg < padrao:
            cbar.ax.axhline(padrao, color=COR_PADRAO, lw=2)

    axes[-1].set_xlabel("Distância a partir de M2 (m)", fontsize=11.5)

    fig.suptitle(
        "Figura RP-4 — Perfis de Pluma com Múltiplas Iso-concentrações\n"
        f"Rio Araçatuba — Cenário B ({CEN_B['label']})",
        fontsize=14, fontweight="bold", y=0.997)

    fig.tight_layout()
    salvar(fig, "fig_RP_4_isoconcentracoes.png")


# =============================================================================
# EXECUÇÃO
# =============================================================================
print("=" * 70)
print("  PERFIL DO RIO COM PLUMA — VISUALIZAÇÃO CINEMATOGRÁFICA")
print("=" * 70)

rodar(fig_rp1)
rodar(fig_rp2)
rodar(fig_rp3)
rodar(fig_rp4)

# Relatório
print("\n" + "=" * 70)
print("  RELATÓRIO FINAL")
print("=" * 70)
arqs = sorted(OUTPUT.glob("*"))
for a in arqs:
    kb = a.stat().st_size / 1024
    print(f"    ✓ {a.name}  ({kb:.0f} KB)")
print("=" * 70)
print(f"  Total: {len(arqs)} arquivos em {OUTPUT}")
print("=" * 70)