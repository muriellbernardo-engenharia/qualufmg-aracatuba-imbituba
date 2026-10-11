# -*- coding: utf-8 -*-
"""
=============================================================================
ZONA DE MISTURA — VERSÃO LIMPA (2 figuras apenas)
Rio Araçatuba — Modelo de Fischer (1979)
=============================================================================
"""

import warnings
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# =============================================================================
# ESTILO
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.titlepad": 10,
    "axes.edgecolor": "#333", "axes.linewidth": 0.9,
    "axes.grid": False,
    "legend.frameon": False,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "output_zm_limpa"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Paleta
AZUL = "#1E88E5"
VERDE = "#2E7D32"
VERMELHO = "#D32F2F"
LARANJA = "#F57C00"
VIOLETA = "#7B1FA2"
CINZA = "#616161"
MARROM = "#5D4037"

CORES_PONTOS = {"M1": VERDE, "P0": VERMELHO, "J1": VIOLETA, "J2": LARANJA}

# Dados
Q_RIO = 1.3558
Q_EF = 0.00234
FD = Q_RIO / Q_EF
B = 8.18
H = 1.05
U = 0.121
DY = 0.05
L_MIX = 0.4 * U * B**2 / DY
T_MIX_S = L_MIX / U
CHI_MIXED = Q_EF / (Q_RIO + Q_EF)

C_RIO_DBO = 20.40
C_EF_DBO = 22.40
LIM_C2_DBO = 5.0


def salvar(fig, nome):
    p = OUTPUT / nome
    fig.savefig(str(p.with_suffix(".png")), dpi=300,
                facecolor="white", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(str(p.with_suffix(".pdf")), facecolor="white",
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  [OK] {nome}.png + .pdf")


def rodar(func, numero, descricao):
    print(f"\n[{numero}] Gerando {descricao}...")
    try:
        func()
        print(f"  ✅ {descricao} gerada")
    except Exception as e:
        print(f"  ❌ ERRO: {type(e).__name__}: {e}")
        traceback.print_exc()


# =============================================================================
# MODELO
# =============================================================================
def chi_2d(x, y, U_v=U, H_v=H, Dy_v=DY, Q_ef_v=Q_EF, B_v=B, y_src=None):
    if y_src is None:
        y_src = B_v / 2
    x = np.maximum(x, 1e-6)
    fator = Q_ef_v / (U_v * H_v * np.sqrt(4 * np.pi * Dy_v * x / U_v))
    y_in = np.atleast_1d(y).astype(float)
    soma = np.zeros_like(y_in)
    for n in range(-3, 4):
        soma += np.exp(-U_v * (y_in - 2*n*B_v - y_src)**2 / (4*Dy_v*x))
        soma += np.exp(-U_v * (y_in - 2*n*B_v + y_src)**2 / (4*Dy_v*x))
    return fator * soma


# =============================================================================
# FIG 1 — VISTA EM PLANTA LIMPA (FIGURA PRINCIPAL)
# =============================================================================
def fig_planta():
    fig, ax = plt.subplots(figsize=(16, 7))

    # Grid
    nx, ny = 500, 250
    x_arr = np.linspace(0.3, 400, nx)
    y_arr = np.linspace(0, B, ny)
    X, Y = np.meshgrid(x_arr, y_arr)

    CHI = np.zeros_like(X)
    for i, xv in enumerate(x_arr):
        CHI[:, i] = chi_2d(xv, y_arr)

    # Pluma (log scale)
    levels = np.logspace(np.log10(1e-4), np.log10(0.05), 60)
    cf = ax.contourf(X, Y, CHI, levels=levels,
                      cmap="RdYlBu_r",
                      norm=LogNorm(vmin=1e-4, vmax=0.05),
                      extend="both", zorder=2)

    # Contornos brancos
    cs = ax.contour(X, Y, CHI, levels=[0.005, 0.01, 0.02],
                     colors="white", linewidths=1.0, alpha=0.7, zorder=3)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.3f", colors="white")

    # Margens (com hachuras)
    for y_borda, dy in [(0, -0.3), (B, +0.3)]:
        ax.plot([0, 400], [y_borda, y_borda],
                color=MARROM, lw=3, zorder=5, solid_capstyle="round")
        for xh in np.arange(0, 400, 12):
            ax.plot([xh, xh + 3], [y_borda, y_borda + dy],
                     color=MARROM, lw=0.8, alpha=0.6, zorder=5)

    # ZMI e ZMC (fundo sutil)
    ax.axvspan(0, L_MIX, color=LARANJA, alpha=0.07, zorder=1)
    ax.axvspan(L_MIX, 400, color=VERDE, alpha=0.05, zorder=1)

    # Linha divisória
    ax.axvline(L_MIX, color=VERDE, lw=1.8, ls="--", alpha=0.7, zorder=4)

    # Rótulos ZMI/ZMC
    ax.text(L_MIX/2, B + 0.35, "ZMI",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=LARANJA)
    ax.text((L_MIX + 400)/2, B + 0.35, "ZMC",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=VERDE)

    # Setas de fluxo
    for xf in np.arange(20, 400, 60):
        ax.annotate("", xy=(xf + 25, -0.9), xytext=(xf, -0.9),
                    arrowprops=dict(arrowstyle="->", color=AZUL, lw=1.5,
                                     alpha=0.7), zorder=3)
    ax.text(200, -1.4, "Fluxo →", ha="center",
            fontsize=10, color=AZUL, fontweight="bold", style="italic")

    # ETE
    ax.scatter(-25, B/2, s=180, marker="s", color="yellow",
                edgecolor=PRETO if False else "black", linewidth=1.5, zorder=10)
    ax.text(-25, B/2 - 1.2, "ETE", ha="center", va="top",
            fontsize=9, fontweight="bold")
    ax.annotate("", xy=(0, B/2), xytext=(-22, B/2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
                zorder=9)

    # Pontos de amostragem
    for nome, x_pos in [("P0", 0), ("M1", 100), ("J1", 109.17),
                         ("J2", 310.51)]:
        marker = "*" if nome == "P0" else "o"
        size = 500 if nome == "P0" else 150
        ax.scatter(x_pos, B/2, s=size, marker=marker,
                    color=CORES_PONTOS[nome],
                    edgecolor="white", linewidth=2.5,
                    zorder=15, alpha=0.95)
        # Label
        if nome == "P0":
            y_off = 20
        else:
            y_off = 16
        ax.annotate(nome, xy=(x_pos, B/2), xytext=(0, y_off),
                    textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold",
                    color=CORES_PONTOS[nome],
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=CORES_PONTOS[nome], lw=1.3, alpha=0.95),
                    zorder=16)

    # Régua
    for xr in np.arange(0, 401, 50):
        ax.plot([xr, xr], [-2.3, -1.9], color="black", lw=1.5, zorder=8)
        ax.text(xr, -2.7, f"{int(xr)}", ha="center",
                fontsize=9, fontweight="bold")
    ax.text(200, -3.2, "Distância a partir de P0 (m)",
            ha="center", fontsize=10, style="italic", color=CINZA)

    # L_mix annotation
    ax.annotate(f"L_mix = {L_MIX:.1f} m",
                xy=(L_MIX, B + 0.2), xytext=(L_MIX, B + 1.1),
                ha="center", fontsize=10, fontweight="bold", color=VERDE,
                arrowprops=dict(arrowstyle="-", color=VERDE, lw=1.2))

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.022, aspect=25)
    cbar.set_label("Fração mássica de efluente χ (%)", fontsize=10, labelpad=8)
    ticks = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t*100:.3f}%" if t < 0.01 else f"{t*100:.1f}%"
                          for t in ticks])
    cbar.ax.axhline(CHI_MIXED, color="white", lw=2.5, alpha=0.9)
    cbar.ax.text(1.8, CHI_MIXED, "  χ_mixed",
                 va="center", fontsize=8, fontweight="bold", color="white")

    # Info box compacto
    info = (f"Q_rio = {Q_RIO:.4f} m³/s\n"
            f"Q_ef   = {Q_EF:.5f} m³/s\n"
            f"FD      = {FD:.0f} : 1")
    ax.text(0.985, 0.97, info, transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#333", lw=1.2, alpha=0.95),
            zorder=20)

    ax.set_xlim(-40, 400)
    ax.set_ylim(-3.8, B + 1.5)
    ax.set_xlabel("")
    ax.set_ylabel("Posição transversal y (m)", fontsize=11, labelpad=8)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_title("Vista em Planta da Pluma de Mistura — Rio Araçatuba\n"
                 f"Modelo de Fischer (1979)  ·  L_mix = {L_MIX:.1f} m  ·  "
                 f"t_mix = {T_MIX_S/60:.1f} min",
                 fontsize=13, fontweight="bold", pad=16)

    fig.text(0.5, -0.02,
             "Residencial Imbituba — URB/69838  |  CREA-SC 222578-3  |  Out/2026  |  "
             "Zona esquemática conforme CONAMA 430/2011 (Art. 13)",
             ha="center", fontsize=8.5, color=CINZA, style="italic")

    salvar(fig, "fig_ZM_planta")


# =============================================================================
# FIG 2 — ANÁLISE EM 4 PAINÉIS
# =============================================================================
def fig_analise():
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28,
                  left=0.08, right=0.96, top=0.87, bottom=0.10)

    x_arr = np.linspace(0.3, 400, 400)
    y_arr = np.linspace(0, B, 150)

    # Pre-calcular
    chi_centro = chi_2d(x_arr, B/2)
    chi_borda = chi_2d(x_arr, 0.05*B)
    chi_media = np.array([np.trapz(chi_2d(xv, y_arr), y_arr) / B
                           for xv in x_arr])

    C_DBO_centro = C_RIO_DBO + chi_centro * (C_EF_DBO - C_RIO_DBO)
    C_DBO_borda = C_RIO_DBO + chi_borda * (C_EF_DBO - C_RIO_DBO)
    C_DBO_media = C_RIO_DBO + chi_media * (C_EF_DBO - C_RIO_DBO)

    # =========================================================================
    # PAINEL A — Perfis transversais
    # =========================================================================
    axA = fig.add_subplot(gs[0, 0])
    distancias = [5, 15, 30, 60, 120, 300]
    cores = plt.cm.plasma(np.linspace(0.1, 0.9, len(distancias)))
    for d, cor in zip(distancias, cores):
        chi_y = chi_2d(d, y_arr) * 100
        axA.semilogy(y_arr, chi_y, color=cor, lw=2.2,
                      label=f"x = {d} m",
                      solid_capstyle="round")
    axA.axhline(CHI_MIXED * 100, color="#333", lw=1.4, ls="--",
                 label=f"χ_mixed = {CHI_MIXED*100:.3f}%")
    axA.set_xlabel("Posição transversal y (m)", fontsize=10.5, labelpad=6)
    axA.set_ylabel("Fração mássica χ (%)", fontsize=10.5, labelpad=6)
    axA.set_title("A.  Perfis transversais", loc="left",
                   fontsize=11.5, fontweight="bold", pad=10)
    axA.legend(fontsize=8.5, loc="upper right", ncol=2,
               columnspacing=0.9, handlelength=1.6, labelspacing=0.35)
    axA.set_ylim(1e-3, 5)
    axA.set_xlim(0, B)
    axA.grid(True, which="both", linestyle=":", alpha=0.35, lw=0.5)
    axA.set_xticks([0, 2, 4, 6, 8])

    # =========================================================================
    # PAINEL B — Decaimento longitudinal de χ
    # =========================================================================
    axB = fig.add_subplot(gs[0, 1])
    axB.semilogy(x_arr, chi_centro * 100, color=VERMELHO, lw=2.2,
                  label="Centro (y = B/2)", solid_capstyle="round")
    axB.semilogy(x_arr, chi_media * 100, color=VERDE, lw=2.2, ls="--",
                  label="Média transversal", solid_capstyle="round")
    axB.semilogy(x_arr, chi_borda * 100, color=AZUL, lw=2.2,
                  label="Borda oposta (y ≈ 0)", solid_capstyle="round")
    axB.axhline(CHI_MIXED * 100, color="#333", lw=1.4, ls=":",
                 label=f"χ_mixed = {CHI_MIXED*100:.3f}%")
    axB.axvline(L_MIX, color=VERDE, lw=1.5, ls="--", alpha=0.7)
    axB.axvspan(L_MIX, 400, color=VERDE, alpha=0.06)

    axB.text(L_MIX + 5, 4, f"L_mix = {L_MIX:.1f} m",
             fontsize=9, color=VERDE, fontweight="bold")

    axB.set_xlabel("Distância a jusante (m)", fontsize=10.5, labelpad=6)
    axB.set_ylabel("Fração mássica χ (%)", fontsize=10.5, labelpad=6)
    axB.set_title("B.  Decaimento longitudinal de χ", loc="left",
                   fontsize=11.5, fontweight="bold", pad=10)
    axB.legend(fontsize=8.5, loc="upper right",
               columnspacing=0.9, handlelength=1.6, labelspacing=0.35)
    axB.set_ylim(5e-4, 5)
    axB.set_xlim(0, 400)
    axB.grid(True, which="both", linestyle=":", alpha=0.35, lw=0.5)

    # =========================================================================
    # PAINEL C — DBO₅ ao longo do trecho
    # =========================================================================
    axC = fig.add_subplot(gs[1, 0])
    axC.plot(x_arr, C_DBO_centro, color=VERMELHO, lw=2.2,
              label="Centro", solid_capstyle="round")
    axC.plot(x_arr, C_DBO_media, color=VERDE, lw=2.2, ls="--",
              label="Média transversal", solid_capstyle="round")
    axC.plot(x_arr, C_DBO_borda, color=AZUL, lw=2.2,
              label="Borda oposta", solid_capstyle="round")
    axC.axhline(LIM_C2_DBO, color="#8B2635", lw=2.0, ls="-.",
                 label=f"Padrão CONAMA C2 ≤ {LIM_C2_DBO:.0f} mg/L")
    axC.axvline(L_MIX, color=VERDE, lw=1.5, ls="--", alpha=0.7)
    axC.axvspan(L_MIX, 400, color=VERDE, alpha=0.06)

    axC.set_xlabel("Distância a jusante (m)", fontsize=10.5, labelpad=6)
    axC.set_ylabel("DBO₅ (mg/L)", fontsize=10.5, labelpad=6)
    axC.set_title("C.  DBO₅ ao longo do trecho", loc="left",
                   fontsize=11.5, fontweight="bold", pad=10)
    axC.legend(fontsize=8.5, loc="upper right",
               columnspacing=0.9, handlelength=1.6, labelspacing=0.35)
    axC.set_ylim(20.30, 20.55)
    axC.set_xlim(0, 400)
    axC.grid(True, linestyle=":", alpha=0.35, lw=0.5)

    # =========================================================================
    # PAINEL D — Comparação em P0, J1, J2
    # =========================================================================
    axD = fig.add_subplot(gs[1, 1])

    pontos_x = [0.5, 109.17, 310.51]
    pontos_nome = ["P0", "J1", "J2"]

    valores_centro = []
    valores_media = []
    for xp in pontos_x:
        c = chi_2d(xp, B/2)
        m = np.trapz(chi_2d(xp, y_arr), y_arr) / B
        valores_centro.append(C_RIO_DBO + c * (C_EF_DBO - C_RIO_DBO))
        valores_media.append(C_RIO_DBO + m * (C_EF_DBO - C_RIO_DBO))

    x_pos = np.arange(len(pontos_nome))
    w = 0.35
    bars1 = axD.bar(x_pos - w/2, valores_centro, width=w,
                     color=[CORES_PONTOS[p] for p in pontos_nome],
                     edgecolor="black", linewidth=1.0, alpha=0.95,
                     label="Concentração no centro")
    bars2 = axD.bar(x_pos + w/2, valores_media, width=w,
                     color=[CORES_PONTOS[p] for p in pontos_nome],
                     edgecolor="black", linewidth=1.0, alpha=0.45,
                     label="Média transversal")

    # Valores nas barras
    for b, v in zip(bars1, valores_centro):
        axD.text(b.get_x() + b.get_width()/2, v + 0.005,
                  f"{v:.2f}", ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold")
    for b, v in zip(bars2, valores_media):
        axD.text(b.get_x() + b.get_width()/2, v + 0.005,
                  f"{v:.2f}", ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold", color="#555")

    # Linha CONAMA
    axD.axhline(LIM_C2_DBO, color="#8B2635", lw=2.0, ls="-.",
                 label=f"Padrão CONAMA C2 ≤ {LIM_C2_DBO:.0f} mg/L")

    axD.set_xticks(x_pos)
    axD.set_xticklabels(pontos_nome, fontsize=11.5, fontweight="bold")
    axD.set_ylabel("DBO₅ (mg/L)", fontsize=10.5, labelpad=6)
    axD.set_title("D.  Concentração de DBO₅ nos pontos-chave", loc="left",
                   fontsize=11.5, fontweight="bold", pad=10)
    axD.legend(fontsize=8.5, loc="upper right", framealpha=0.95,
               edgecolor="#CCC")
    axD.set_ylim(19.5, 21.0)
    axD.grid(True, axis="y", linestyle=":", alpha=0.35, lw=0.5)

    # Nota crítica
    axD.text(0.5, -0.28,
             "Os valores permanecem acima do padrão CONAMA\n"
             "devido à condição pré-existente do corpo receptor.",
             transform=axD.transAxes, ha="center", va="top",
             fontsize=9, style="italic", color="#666",
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1",
                       ec=LARANJA, lw=1.0))

    # Títulos globais
    fig.text(0.5, 0.955,
             "Análise da Zona de Mistura — Rio Araçatuba",
             ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.915,
             f"Q_rio = {Q_RIO:.4f} m³/s  ·  Q_ef = {Q_EF:.5f} m³/s  ·  "
             f"FD = {FD:.0f}:1  ·  L_mix = {L_MIX:.1f} m",
             ha="center", fontsize=10, color="#444", style="italic")
    fig.text(0.5, 0.015,
             "Residencial Imbituba — URB/69838  |  CREA-SC 222578-3  |  Out/2026  "
             "|  Modelo de Fischer (1979) — métodos das imagens espelhadas",
             ha="center", fontsize=8.5, color=CINZA, style="italic")

    salvar(fig, "fig_ZM_analise")


# =============================================================================
# EXECUÇÃO
# =============================================================================
print("=" * 70)
print("  ZONA DE MISTURA — VERSÃO LIMPA")
print("=" * 70)

rodar(fig_planta, "1", "Vista em planta da pluma")
rodar(fig_analise, "2", "Análise em 4 painéis")

# Relatório
print("\n" + "=" * 70)
print("  RELATÓRIO FINAL")
print("=" * 70)
arqs = sorted(OUTPUT.glob("*.png"))
for a in arqs:
    kb = a.stat().st_size / 1024
    print(f"    ✓ {a.name}  ({kb:.0f} KB)")
print("=" * 70)
print(f"  Pasta: {OUTPUT}")
print("=" * 70)