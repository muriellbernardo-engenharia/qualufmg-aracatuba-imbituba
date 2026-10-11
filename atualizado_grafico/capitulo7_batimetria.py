# -*- coding: utf-8 -*-
"""
=============================================================================
CAPÍTULO 7 — GRÁFICOS DE BATIMETRIA, MORFOMETRIA E HIDRÁULICA
Rio Araçatuba — ETE Residencial Imbituba
=============================================================================
Gera 5 figuras:
  - Fig 7.1 — Perfil longitudinal do leito
  - Fig 7.2 — Seções transversais (5 cortes)
  - Fig 7.3 — Largura total × largura efetiva
  - Fig 7.4 — Velocidades superficial e média
  - Fig 7.5 — Vazões por seção

Autor: Eng.ª Muriell Bernardo Macedo (CREA-SC 222578-3)
Licença: MIT
=============================================================================
"""

import warnings
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# FIX DE PATH — garante que a pasta "dados" seja encontrada
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PASTA DE SAÍDA — sempre dentro de atualizado_grafico/
# -----------------------------------------------------------------------------
OUTPUT = SCRIPT_DIR / "output_cap7"
OUTPUT.mkdir(parents=True, exist_ok=True)

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

COR_LEITO   = "#4A3B2C"
COR_AGUA    = "#4A90E2"
COR_TERRENO = "#8B7355"
COR_P0      = "#E5A54B"
COR_EFETIVA = "#2E7D32"
COR_TOTAL   = "#B22222"
COR_VEL     = "#0D4A70"


# =============================================================================
# DADOS — Tabelas 7.2A, 7.2C, 7.2D, 7.3C, 7.3E
# =============================================================================
# Tabela 7.2A — Profundidades medidas (distâncias relativas a P0)
SECOES = {
    "M2": {"dist": -214.23, "L_ef": 9.60,  "L_total": 10.20,
           "h_esq": 1.80, "h_centro": 2.52, "h_dir": 2.23, "h_media": 2.18},
    "M1": {"dist": -113.50, "L_ef": 13.60, "L_total": 13.63,
           "h_esq": 1.10, "h_centro": 2.65, "h_dir": 0.80, "h_media": 1.52},
    "P0": {"dist":    0.00, "L_ef": 8.18,  "L_total": 8.64,
           "h_esq": 1.00, "h_centro": 2.15, "h_dir": 1.10, "h_media": 1.42},
    "J1": {"dist":  +97.00, "L_ef": 16.00, "L_total": 19.00,
           "h_esq": 0.50, "h_centro": 2.55, "h_dir": 0.65, "h_media": 1.23},
    "J2": {"dist": +296.28, "L_ef": 7.00,  "L_total": 12.37,
           "h_esq": 1.30, "h_centro": 2.22, "h_dir": 1.30, "h_media": 1.61},
}

# Tabela 7.2C — Perfil longitudinal (distância a partir de M2 = 0)
PERFIL = {
    "pontos": ["M2", "M1", "P0", "J1", "J2"],
    "dist":   [0.00, 100.73, 214.23, 311.23, 510.51],
    "cota_terreno": [4.22, 4.20, 4.18, 4.16, 4.11],
    "h_centro":     [2.52, 2.65, 2.15, 2.55, 2.22],
    "cota_leito":   [1.70, 1.55, 2.03, 1.61, 1.89],
}

# Tabela 7.3C — Velocidades
VEL = {
    "pontos": ["M2", "M1", "P0", "J1", "J2"],
    "V_sup":   [0.10417, 0.12346, 0.14233, 0.16077, 0.15060],
    "V_media": [0.08854, 0.10494, 0.12098, 0.13665, 0.12801],
}

# Tabela 7.3E — Vazões
VAZAO = {
    "pontos": ["M2", "M1", "P0", "J1", "J2"],
    "A":  [22.58, 28.33, 14.59, 30.27, 13.39],
    "Q":  [2.000, 2.973, 1.765, 4.136, 1.714],
}


def salvar(fig, nome):
    p = OUTPUT / nome
    fig.savefig(p, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[OK] {p}")
    plt.close(fig)


# =============================================================================
# FIGURA 7.1 — PERFIL LONGITUDINAL DO LEITO
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

x = PERFIL["dist"]
terr = PERFIL["cota_terreno"]
leito = PERFIL["cota_leito"]

# Área entre terreno e leito (água + margens)
ax.fill_between(x, leito, terr, color=COR_AGUA, alpha=0.28,
                label="Coluna d'água")

# Linhas
ax.plot(x, terr, color=COR_TERRENO, lw=2.5, marker="s",
        markersize=9, markeredgecolor="white", markeredgewidth=1.5,
        label="Cota do terreno (TOPODATA)")
ax.plot(x, leito, color=COR_LEITO, lw=2.8, marker="o",
        markersize=10, markeredgecolor="white", markeredgewidth=1.5,
        label="Cota do leito (medida em campo)")

# Rótulos dos pontos
for pt, xi, ti, li in zip(PERFIL["pontos"], x, terr, leito):
    ax.annotate(pt, xy=(xi, ti), xytext=(0, 12),
                textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold", color="#333",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#CCC", lw=0.8))
    ax.annotate(f"{li:.2f} m", xy=(xi, li), xytext=(0, -18),
                textcoords="offset points", ha="center",
                fontsize=9, color=COR_LEITO, fontweight="bold")

# Destaque P0
ax.axvline(214.23, color=COR_P0, lw=2, ls="--", alpha=0.85)
ax.text(214.23, 3.4, "P0 (lançamento)", ha="center",
        fontsize=10, fontweight="bold", color=COR_P0,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COR_P0, lw=1.2))

ax.set_xlim(-20, 530)
ax.set_ylim(1.2, 4.7)
ax.set_xlabel("Distância a partir de M2 (m)")
ax.set_ylabel("Cota (m)")
ax.set_title("Capítulo 7 — Figura 7.1\n"
             "Perfil Longitudinal do Leito do Rio Araçatuba",
             fontweight="bold")
ax.legend(loc="upper right", fontsize=9)

fig.tight_layout()
salvar(fig, "fig_7_1_perfil_longitudinal_leito.png")


# =============================================================================
# FIGURA 7.2 — SEÇÕES TRANSVERSAIS (5 cortes)
# =============================================================================
fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=False)

for ax, (nome, s) in zip(axes, SECOES.items()):
    L = s["L_ef"]
    h_e = s["h_esq"]
    h_c = s["h_centro"]
    h_d = s["h_dir"]

    # Coordenadas horizontais: 0, L/2, L
    x_sec = np.array([0, L / 2, L])
    # Profundidades (positivo para baixo → invertemos o eixo y)
    h_sec = np.array([h_e, h_c, h_d])

    # Interpolação parabólica suave
    from scipy.interpolate import interp1d
    f = interp1d(x_sec, h_sec, kind="quadratic")
    x_fino = np.linspace(0, L, 200)
    h_fino = f(x_fino)

    # Preenchimento da água (invertido: água abaixo da superfície)
    ax.fill_between(x_fino, 0, h_fino, color=COR_AGUA, alpha=0.4,
                    label="Seção molhada")
    ax.plot(x_fino, h_fino, color=COR_LEITO, lw=2.5,
            label="Leito do canal")

    # Linha da superfície
    ax.axhline(0, color=COR_AGUA, lw=1.8, ls="-")

    # Pontos medidos
    ax.scatter(x_sec, h_sec, s=90, marker="o",
               facecolor="white", edgecolor=COR_LEITO,
               linewidth=2, zorder=5)

    for xi, hi, lbl in zip(x_sec, h_sec,
                            [f"ME\n{hi:.2f} m",
                             f"Centro\n{hi:.2f} m",
                             f"MD\n{hi:.2f} m"]):
        ax.annotate(lbl, xy=(xi, hi), xytext=(0, -22),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, fontweight="bold", color="#333")

    # Área molhada pelo método de Simpson
    A = L * (h_e + 4 * h_c + h_d) / 6
    ax.text(0.98, 0.08,
            f"L_ef = {L:.2f} m\n"
            f"A (Simpson) = {A:.2f} m²",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5",
                      ec="#999", lw=1.2))

    ax.set_title(f"Seção {nome}  —  distância a P0: {s['dist']:+.2f} m",
                 fontsize=11, pad=6)
    ax.set_ylabel("Profundidade (m)", fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(max(h_sec) + 0.6, -0.4)

axes[-1].set_xlabel("Posição lateral na seção (m)", fontsize=10)

fig.suptitle("Capítulo 7 — Figura 7.2\n"
             "Seções Transversais Levantadas em Campo (Campanha C3)",
             fontsize=13, fontweight="bold", y=0.998)
fig.tight_layout()
salvar(fig, "fig_7_2_secoes_transversais.png")


# =============================================================================
# FIGURA 7.3 — LARGURA TOTAL × LARGURA EFETIVA
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

pontos = list(SECOES.keys())
x_pos = np.arange(len(pontos))
L_total = [SECOES[p]["L_total"] for p in pontos]
L_ef = [SECOES[p]["L_ef"] for p in pontos]
reducao = [(SECOES[p]["L_total"] - SECOES[p]["L_ef"]) / SECOES[p]["L_total"] * 100
           for p in pontos]

# Barras de largura total (fundo)
ax.bar(x_pos, L_total, width=0.55, color=COR_TOTAL, alpha=0.35,
       edgecolor=COR_TOTAL, linewidth=1.5, label="Largura total (com vegetação)")

# Barras de largura efetiva (frente)
ax.bar(x_pos, L_ef, width=0.55, color=COR_EFETIVA, alpha=0.85,
       edgecolor="black", linewidth=1.0, label="Largura efetiva (sem vegetação)")

# Anotações
for i, (lt, le, red) in enumerate(zip(L_total, L_ef, reducao)):
    ax.text(i, lt + 0.5, f"{lt:.2f} m", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=COR_TOTAL)
    ax.text(i, le - 1.1, f"{le:.2f} m", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="white")
    ax.text(i, -2.8, f"−{red:.1f}%", ha="center", va="top",
            fontsize=10, fontweight="bold", color=COR_TOTAL,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFEBEE",
                      ec=COR_TOTAL, lw=1.0))

ax.set_xticks(x_pos)
ax.set_xticklabels(pontos, fontsize=11, fontweight="bold")
ax.set_xlabel("Ponto de monitoramento")
ax.set_ylabel("Largura (m)")
ax.set_title("Capítulo 7 — Figura 7.3\n"
             "Largura Total × Largura Efetiva por Seção\n"
             "Redução causada por vegetação aquática emergente",
             fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.set_ylim(-5, max(L_total) * 1.15)

fig.tight_layout()
salvar(fig, "fig_7_3_largura_total_efetiva.png")


# =============================================================================
# FIGURA 7.4 — VELOCIDADES SUPERFICIAL E MÉDIA
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

pontos_v = VEL["pontos"]
x_v = np.arange(len(pontos_v))
Vsup = VEL["V_sup"]
Vmed = VEL["V_media"]

w = 0.35
b1 = ax.bar(x_v - w/2, Vsup, width=w, color=COR_AGUA, alpha=0.9,
            edgecolor="black", linewidth=1.0,
            label="V superficial (medida com flutuador)")
b2 = ax.bar(x_v + w/2, Vmed, width=w, color=COR_VEL, alpha=0.95,
            edgecolor="black", linewidth=1.0,
            label="V média (k = 0,85)")

for i, (vs, vm) in enumerate(zip(Vsup, Vmed)):
    ax.text(i - w/2, vs + 0.003, f"{vs:.4f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=COR_AGUA)
    ax.text(i + w/2, vm + 0.003, f"{vm:.4f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=COR_VEL)

ax.set_xticks(x_v)
ax.set_xticklabels(pontos_v, fontsize=11, fontweight="bold")
ax.set_xlabel("Ponto de monitoramento")
ax.set_ylabel("Velocidade (m/s)")
ax.set_title("Capítulo 7 — Figura 7.4\n"
             "Velocidades Superficial e Média por Seção\n"
             "Método do flutuador (ABNT NBR 13403)",
             fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.set_ylim(0, max(Vsup) * 1.25)

fig.tight_layout()
salvar(fig, "fig_7_4_velocidades.png")


# =============================================================================
# FIGURA 7.5 — VAZÕES POR SEÇÃO
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios": [1, 1]})

pontos_q = VAZAO["pontos"]
x_q = np.arange(len(pontos_q))
A_molhada = VAZAO["A"]
Q = VAZAO["Q"]

# --- Painel 1: Área molhada ---
bars1 = ax1.bar(x_q, A_molhada, width=0.6, color="#7B1FA2",
                alpha=0.85, edgecolor="black", linewidth=1.2)
for i, v in enumerate(A_molhada):
    ax1.text(i, v + 0.6, f"{v:.2f} m²", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
ax1.set_xticks(x_q)
ax1.set_xticklabels(pontos_q, fontsize=11, fontweight="bold")
ax1.set_xlabel("Ponto de monitoramento")
ax1.set_ylabel("Área molhada (m²)")
ax1.set_title("Área Molhada (método de Simpson)", fontsize=11, pad=10)
ax1.set_ylim(0, max(A_molhada) * 1.2)

# --- Painel 2: Vazão ---
cores_q = [COR_P0 if p == "P0" else COR_VEL for p in pontos_q]
bars2 = ax2.bar(x_q, Q, width=0.6, color=cores_q,
                alpha=0.9, edgecolor="black", linewidth=1.2)
for i, v in enumerate(Q):
    ax2.text(i, v + 0.1, f"{v:.3f} m³/s", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
ax2.set_xticks(x_q)
ax2.set_xticklabels(pontos_q, fontsize=11, fontweight="bold")
ax2.set_xlabel("Ponto de monitoramento")
ax2.set_ylabel("Vazão (m³/s)")
ax2.set_title("Vazão pelo Método da Continuidade", fontsize=11, pad=10)
ax2.set_ylim(0, max(Q) * 1.2)

# Linha de referência Q90
ax2.axhline(0.745, color=COR_TOTAL, lw=2, ls="--",
            label="Q90 = 0,745 m³/s (IMA 192/2025)")
ax2.legend(loc="upper right", fontsize=9)

fig.suptitle("Capítulo 7 — Figura 7.5\n"
             "Área Molhada e Vazão por Seção — Campanha C3",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
salvar(fig, "fig_7_5_areas_vazoes.png")


# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
print("\n" + "=" * 66)
print("  CAPÍTULO 7 — GRÁFICOS GERADOS COM SUCESSO")
print("=" * 66)
print(f"  Pasta de saída: {OUTPUT}")
print(f"  Total: 5 figuras PNG (400 DPI)")
print("=" * 66)
for f in sorted(OUTPUT.glob("*.png")):
    print(f"    - {f.name}")
print("=" * 66)