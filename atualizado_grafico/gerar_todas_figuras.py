# -*- coding: utf-8 -*-
"""
=============================================================================
GERADOR PROFISSIONAL DE FIGURAS — RELATÓRIO M91 RESIDENCIAL IMBITUBA
Rio Araçatuba — Estudo de Autodepuração e Concepção do Emissário
=============================================================================
Gera 28 figuras profissionais (2D, 3D, gradientes, plumas, perfis).
TODOS os dados estão embutidos — não depende de imports externos.
Cada figura é gerada em try/except isolado — uma falha não para as outras.

Autor: Eng.ª Muriell Bernardo Macedo (CREA-SC 222578-3)
Licença: MIT
=============================================================================
"""

import warnings
from pathlib import Path
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem display (mais rápido e sem erro)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

# =============================================================================
# PASTAS DE SAÍDA
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

DIR_CAP7 = SCRIPT_DIR / "output_capitulo7"
DIR_CAP8 = SCRIPT_DIR / "output_capitulo8"
DIR_CAP9 = SCRIPT_DIR / "output_capitulo9"
DIR_CAP10 = SCRIPT_DIR / "output_capitulo10"
DIR_ANEXO = SCRIPT_DIR / "output_anexo_A2"

for d in [DIR_CAP7, DIR_CAP8, DIR_CAP9, DIR_CAP10, DIR_ANEXO]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Pasta raiz: {SCRIPT_DIR}")
print(f"[INFO] Saída: {DIR_CAP7.name}, {DIR_CAP8.name}, {DIR_CAP9.name}, "
      f"{DIR_CAP10.name}, {DIR_ANEXO.name}\n")


# =============================================================================
# ESTILO GLOBAL
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.titlepad": 12,
    "axes.edgecolor": "#2B2B2B", "axes.linewidth": 1.0,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.7, "grid.linestyle": "-",
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "legend.fontsize": 9,
    "figure.dpi": 120, "savefig.dpi": 400,
    "savefig.facecolor": "white", "savefig.bbox": "tight",
})

# Paleta
COR_A       = "#0D4A70"   # Cenário A
COR_B       = "#C1440E"   # Cenário B
COR_PADRAO  = "#8B2635"   # Padrão CONAMA
COR_SAT     = "#4A7C59"   # Saturação OD
COR_P0      = "#E5A54B"   # Ponto P0
COR_AGUA    = "#4A90E2"
COR_LEITO   = "#4A3B2C"
COR_TERRENO = "#8B7355"
COR_EFETIVA = "#2E7D32"
COR_TOTAL   = "#B22222"

CMAP_RIO = LinearSegmentedColormap.from_list("rio",
    ["#1B4F72", "#2E86C1", "#5DADE2", "#85C1E9", "#AED6F1"])
CMAP_COTA = LinearSegmentedColormap.from_list("cota",
    ["#4A3B2C", "#8B7355", "#C4A57B", "#E5D3B3"])
CMAP_PLUME = LinearSegmentedColormap.from_list("plume",
    ["#F7FCF5", "#C7E9C0", "#74C476", "#31A354", "#FD8D3C", "#D94801", "#8B2635"])


# =============================================================================
# DADOS DO RELATÓRIO (todos inline)
# =============================================================================
# --- Cap 7 — Batimetria e Hidráulica (Tabelas 7.1 a 7.3F) ---
PONTOS_NOMES = ["M2", "M1", "P0", "J1", "J2"]
PONTOS_DIST = {"M2": -214.23, "M1": -113.50, "P0": 0.0,
               "J1": 97.00, "J2": 296.28}

SECOES = {
    "M2": dict(L_ef=9.60, L_tot=10.20, he=1.80, hc=2.52, hd=2.23, hm=2.18),
    "M1": dict(L_ef=13.60, L_tot=13.63, he=1.10, hc=2.65, hd=0.80, hm=1.52),
    "P0": dict(L_ef=8.18, L_tot=8.64, he=1.00, hc=2.15, hd=1.10, hm=1.42),
    "J1": dict(L_ef=16.00, L_tot=19.00, he=0.50, hc=2.55, hd=0.65, hm=1.23),
    "J2": dict(L_ef=7.00, L_tot=12.37, he=1.30, hc=2.22, hd=1.30, hm=1.61),
}

PERFIL = dict(
    dist=[0.00, 100.73, 214.23, 311.23, 510.51],
    cota_terreno=[4.22, 4.20, 4.18, 4.16, 4.11],
    cota_leito=[1.70, 1.55, 2.03, 1.61, 1.89],
)

INSITU = dict(
    OD=[6.2, 6.5, 6.3, 6.1, 6.1],
    pH=[7.12, 7.44, 7.69, 7.78, 7.43],
    T=[17.2, 17.6, 16.8, 16.9, 16.9],
    Cond=[65.9, 64.2, 62.8, 63.0, 64.5],
    TDS=[32.7, 32.0, 31.3, 31.4, 32.3],
)

VEL = dict(
    V_sup=[0.10417, 0.12346, 0.14233, 0.16077, 0.15060],
    V_med=[0.08854, 0.10494, 0.12098, 0.13665, 0.12801],
)

AREA_Q = dict(
    A=[22.58, 28.33, 14.59, 30.27, 13.39],
    Q=[2.000, 2.973, 1.765, 4.136, 1.714],
)

# --- Cap 9 — Emissário (Tabelas 9.5 a 9.20) ---
PLANOS = {
    "A": dict(V1=(0.00, 7.4388, 6.439),
              V2=(406.02, 2.8933, 4.542),
              V3=(588.02, 4.6918, 3.692), L=588.02),
    "B": dict(V1=(0.00, 7.4388, 6.439),
              V2=(418.73, 2.8933, 4.486),
              V3=(588.887, 4.6918, 3.692), L=588.887),
    "C": dict(V1=(0.00, 7.4388, 6.439),
              V2=(617.127, 3.9316, 2.932),
              L=617.127),
}

DIAMETROS = dict(
    DN=[75, 100, 150, 200],
    V=[1.024, 0.576, 0.256, 0.144],
    J=[0.01630, 0.00400, 0.00056, 0.00014],
)

SOLUCOES = {
    "A1 (Aterro)": dict(ini=340800, op=0, tot=340800),
    "B1 (Aterro)": dict(ini=338463, op=0, tot=338463),
    "C1 (Aterro)": dict(ini=337112, op=0, tot=337112),
    "A2 (Bomb.)":  dict(ini=182250, op=32376, tot=214626),
    "B2 (Bomb.)":  dict(ini=182375, op=32194, tot=214569),
    "C2 (Bomb.)":  dict(ini=185875, op=28249, tot=214124),
}

# --- Pluviometria (Tabela 2.4) ---
PLUV_MESES = ["Set/25","Out/25","Nov/25","Dez/25","Jan/26","Fev/26",
              "Mar/26","Abr/26","Mai/26","Jun/26","Jul/26","Ago/26","Set/26"]
PLUV_MED = [58.4, 22.0, 5.2, 45.4, 79.2, 94.8, 45.0,
            28.2, 31.6, 34.0, 27.4, 18.0, 12.0]
PLUV_HIST = [100, 105, 90, 100, 130, 135, 120, 110, 95, 85, 80, 90, 100]

# --- Modelo (Cap 8 + Anexo A2) ---
OD_SAT = 9.48
PROF = 1.78
F_DBO = 1.46
PADROES = dict(OD=5.0, DBO5=5.0, N_NH3=2.0, N_NO2=1.0, N_NO3=10.0,
               P_total=0.10, Coli=1000.0)

RIO_M1 = dict(OD=6.50, DBO5=20.40, N_org=0.16, N_NH3=0.05,
              N_NO2=0.03, N_NO3=0.40, P_org=0.04, P_inorg=0.04, Coli=78.0)
EFLUENTE = dict(Q=0.00234, OD=4.00, DBO5=22.40, N_org=1.93, N_NH3=0.05,
                N_NO2=0.0, N_NO3=24.47, P_org=1.51, P_inorg=1.51, Coli=1000.0)
COEF = dict(Kd=0.134, Ks=0.0, Kso=0.05, Koa=0.20, Kan=0.20, Knn=0.75,
            Kspo=0.02, Koi=0.27, Kspi=0.02, Kb=1.0,
            O2Namon=3.3, O2Nnitri=1.1, k_nitr=0.6, S_NH3=0.0)
CEN_A = dict(Q=2.973, V=0.10494, K2=0.800)
CEN_B = dict(Q=0.745, V=0.05253, K2=0.566)


# =============================================================================
# HELPERS
# =============================================================================
def salvar(fig, pasta, nome):
    """Salva figura e sempre fecha. Retorna True/False."""
    p = pasta / nome
    try:
        fig.savefig(str(p), dpi=400, bbox_inches="tight", facecolor="white")
        tamanho_kb = p.stat().st_size / 1024
        print(f"  [OK] {pasta.name}/{nome}  ({tamanho_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [ERRO ao salvar {nome}] {e}")
        return False
    finally:
        plt.close(fig)


def gradient_line(ax, x, y, cmap="viridis", lw=3.5, zorder=3, alpha=1.0):
    """Desenha linha com gradiente de cor ao longo do eixo x."""
    x = np.asarray(x); y = np.asarray(y)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw, zorder=zorder, alpha=alpha)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    return lc


def gradient_fill(ax, x, y_top, y_bottom, cmap="Blues",
                  alpha=0.6, n_strips=50, zorder=1):
    """Preenche área com gradiente vertical."""
    x = np.asarray(x)
    y_top = np.asarray(y_top)
    y_bottom = np.asarray(y_bottom)
    cmap_obj = plt.get_cmap(cmap)
    for i in range(n_strips):
        t0, t1 = i / n_strips, (i + 1) / n_strips
        yt0 = y_bottom + (y_top - y_bottom) * t0
        yt1 = y_bottom + (y_top - y_bottom) * t1
        cor = cmap_obj(0.3 + 0.6 * (1 - t0))
        ax.fill_between(x, yt0, yt1, color=cor, alpha=alpha,
                        linewidth=0, zorder=zorder)


def rodar(nome_func, *args, **kwargs):
    """Executa uma função de figura com try/except."""
    try:
        nome_func(*args, **kwargs)
    except Exception as e:
        print(f"  [ERRO em {nome_func.__name__}]: {e}")
        traceback.print_exc()


# =============================================================================
# MODELO QUAL-UFMG
# =============================================================================
def fn_(od, k):
    return od / (od + k) if (od + k) > 0 else 0.0


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


def simular(cen, xf):
    Qr, Qe = cen["Q"], EFLUENTE["Q"]
    Qt = Qr + Qe
    C0 = {k: (Qr * RIO_M1[k] + Qe * EFLUENTE[k]) / Qt for k in RIO_M1}
    U = cen["V"] * 86400.0
    y0 = [C0["DBO5"] * F_DBO, C0["OD"], C0["N_org"], C0["N_NH3"],
          C0["N_NO2"], C0["N_NO3"], C0["P_org"], C0["P_inorg"], C0["Coli"]]
    s = solve_ivp(lambda x, y: ode(x, y, U, cen["K2"], COEF, PROF),
                  (0, xf), y0, method="RK45",
                  rtol=1e-6, atol=1e-9, dense_output=True)
    xg = np.linspace(0, xf, int(xf) + 1)
    Y = s.sol(xg)
    L, OD, Norg, NNH3, NNO2, NNO3, Porg, Pinorg, Coli = Y
    return dict(x=xg, OD=OD, DBO5=L / F_DBO, N_NH3=NNH3, N_NO2=NNO2,
                N_NO3=NNO3, P_total=Porg + Pinorg, Coli=Coli,
                FD=Qr / Qe, Q_tot=Qt)


print("[INFO] Simulando QUAL-UFMG (Cap 8 e Anexo A2)...")
resA8 = simular(CEN_A, int(PONTOS_DIST["J2"]))
resB8 = simular(CEN_B, int(PONTOS_DIST["J2"]))
resA2 = simular(CEN_A, 1500)
resB2 = simular(CEN_B, 1500)
print("[INFO] Simulações concluídas.\n")


# =============================================================================
# =========================== CAPÍTULO 7 ======================================
# =============================================================================
print("=" * 70)
print("  CAPÍTULO 7 — BATIMETRIA, MORFOMETRIA E HIDRÁULICA")
print("=" * 70)


def fig_7_1():
    """Perfil longitudinal do leito com gradiente."""
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = PERFIL["dist"]
    terr = np.array(PERFIL["cota_terreno"])
    leito = np.array(PERFIL["cota_leito"])

    # Preenchimento com gradiente (coluna d'água)
    gradient_fill(ax, x, terr, leito, cmap="Blues", alpha=0.45, n_strips=60)

    # Linha do terreno com gradiente
    gradient_line(ax, x, terr, cmap="copper", lw=3.0, zorder=3)
    ax.plot(x, terr, color="none", marker="s", markersize=10,
            markeredgecolor=COR_TERRENO, markerfacecolor="white",
            markeredgewidth=2, zorder=5, label="Terreno (TOPODATA)")

    # Linha do leito com gradiente escuro
    gradient_line(ax, x, leito, cmap="copper_r", lw=3.5, zorder=4)
    ax.plot(x, leito, color="none", marker="o", markersize=11,
            markeredgecolor=COR_LEITO, markerfacecolor="white",
            markeredgewidth=2, zorder=6, label="Leito (medido em campo)")

    # Rótulos
    for pt, xi, ti, li in zip(PONTOS_NOMES, x, terr, leito):
        ax.annotate(pt, xy=(xi, ti), xytext=(0, 14),
                    textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold", color="#333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#BBB", lw=0.8))
        ax.annotate(f"{li:.2f} m", xy=(xi, li), xytext=(0, -20),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=COR_LEITO, fontweight="bold")

    # Destaque P0
    ax.axvline(214.23, color=COR_P0, lw=2.2, ls="--", alpha=0.85)
    ax.text(214.23, 3.6, "P0 (lançamento)", ha="center", fontsize=10,
            fontweight="bold", color=COR_P0,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=COR_P0, lw=1.2))

    ax.set_xlim(-20, 530); ax.set_ylim(1.2, 4.7)
    ax.set_xlabel("Distância a partir de M2 (m)")
    ax.set_ylabel("Cota (m)")
    ax.set_title("Figura 7.1 — Perfil Longitudinal do Leito do Rio Araçatuba",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5)
    salvar(fig, DIR_CAP7, "fig_7_1_perfil_longitudinal_leito.png")


def fig_7_2():
    """Seções transversais com preenchimento gradiente."""
    fig, axes = plt.subplots(5, 1, figsize=(11, 15))
    for ax, (nome, s) in zip(axes, SECOES.items()):
        L = s["L_ef"]
        h = np.array([s["he"], s["hc"], s["hd"]])
        xs = np.array([0, L/2, L])
        f = interp1d(xs, h, kind="quadratic")
        xf = np.linspace(0, L, 200)
        hf = f(xf)

        # Preenchimento com gradiente de profundidade
        gradient_fill(ax, xf, np.zeros_like(hf), hf,
                      cmap="Blues_r", alpha=0.7, n_strips=40)

        # Linha do leito
        gradient_line(ax, xf, hf, cmap="copper", lw=3.0, zorder=3)

        # Superfície da água
        ax.axhline(0, color=COR_AGUA, lw=2, zorder=2)

        # Pontos medidos
        ax.scatter(xs, h, s=140, marker="o", facecolor="white",
                   edgecolor=COR_LEITO, linewidth=2.5, zorder=5)

        for xi, hi, lbl in zip(xs, h,
                                [f"ME\n{hi:.2f} m", f"Centro\n{hi:.2f} m",
                                 f"MD\n{hi:.2f} m"]):
            ax.annotate(lbl, xy=(xi, hi), xytext=(0, -28),
                        textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold", color="#333")

        A = L * (s["he"] + 4 * s["hc"] + s["hd"]) / 6
        ax.text(0.98, 0.08,
                f"L_ef = {L:.2f} m  |  A = {A:.2f} m²",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5",
                          ec="#999", lw=1.2))

        ax.set_title(f"Seção {nome}  —  distância a P0: {s.get('dist', '')} "
                     f"{PONTOS_DIST[nome]:+.2f} m" if nome in PONTOS_DIST
                     else f"Seção {nome}",
                     fontsize=11, pad=6)
        ax.set_ylabel("Profundidade (m)", fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(max(h) + 0.7, -0.4)

    axes[-1].set_xlabel("Posição lateral na seção (m)", fontsize=10)
    fig.suptitle("Figura 7.2 — Seções Transversais Levantadas em Campo "
                 "(Campanha C3)", fontsize=13, fontweight="bold", y=0.998)
    fig.tight_layout()
    salvar(fig, DIR_CAP7, "fig_7_2_secoes_transversais.png")


def fig_7_3():
    """Largura total × efetiva com barras."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(PONTOS_NOMES))
    L_tot = [SECOES[p]["L_tot"] for p in PONTOS_NOMES]
    L_ef = [SECOES[p]["L_ef"] for p in PONTOS_NOMES]
    red = [(l1 - l2) / l1 * 100 for l1, l2 in zip(L_tot, L_ef)]

    ax.bar(x, L_tot, width=0.6, color=COR_TOTAL, alpha=0.30,
           edgecolor=COR_TOTAL, linewidth=2, label="Largura total")
    ax.bar(x, L_ef, width=0.6, color=COR_EFETIVA, alpha=0.85,
           edgecolor="black", linewidth=1.2, label="Largura efetiva")

    for i, (lt, le, r) in enumerate(zip(L_tot, L_ef, red)):
        ax.text(i, lt + 0.4, f"{lt:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=COR_TOTAL)
        ax.text(i, le - 1.0, f"{le:.2f}", ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="white")
        ax.text(i, -2.6, f"−{r:.1f}%", ha="center", va="top",
                fontsize=10, fontweight="bold", color=COR_TOTAL,
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFEBEE",
                          ec=COR_TOTAL, lw=1.0))

    ax.set_xticks(x); ax.set_xticklabels(PONTOS_NOMES, fontsize=12,
                                          fontweight="bold")
    ax.set_xlabel("Ponto de monitoramento")
    ax.set_ylabel("Largura (m)")
    ax.set_title("Figura 7.3 — Largura Total × Largura Efetiva por Seção\n"
                 "Redução por vegetação aquática emergente", fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(-5, max(L_tot) * 1.15)
    salvar(fig, DIR_CAP7, "fig_7_3_largura_total_efetiva.png")


def fig_7_4():
    """Velocidades com gradiente."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(PONTOS_NOMES))
    w = 0.36
    ax.bar(x - w/2, VEL["V_sup"], width=w, color=COR_AGUA, alpha=0.92,
           edgecolor="black", linewidth=1.0,
           label="V superficial (flutuador)")
    ax.bar(x + w/2, VEL["V_med"], width=w, color=COR_A, alpha=0.95,
           edgecolor="black", linewidth=1.0, label="V média (k = 0,85)")

    for i, (vs, vm) in enumerate(zip(VEL["V_sup"], VEL["V_med"])):
        ax.text(i - w/2, vs + 0.003, f"{vs:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=COR_AGUA)
        ax.text(i + w/2, vm + 0.003, f"{vm:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=COR_A)

    ax.set_xticks(x); ax.set_xticklabels(PONTOS_NOMES, fontsize=12,
                                          fontweight="bold")
    ax.set_xlabel("Ponto de monitoramento")
    ax.set_ylabel("Velocidade (m/s)")
    ax.set_title("Figura 7.4 — Velocidades Superficial e Média por Seção\n"
                 "Método do flutuador (ABNT NBR 13403)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, max(VEL["V_sup"]) * 1.25)
    salvar(fig, DIR_CAP7, "fig_7_4_velocidades.png")


def fig_7_5():
    """Áreas e vazões."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(PONTOS_NOMES))

    cores_A = ["#7B1FA2", "#8E24AA", COR_P0, "#6A1B9A", "#4A148C"]
    ax1.bar(x, AREA_Q["A"], width=0.65, color=cores_A, alpha=0.9,
            edgecolor="black", linewidth=1.2)
    for i, v in enumerate(AREA_Q["A"]):
        ax1.text(i, v + 0.6, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(PONTOS_NOMES, fontsize=11,
                                            fontweight="bold")
    ax1.set_xlabel("Ponto"); ax1.set_ylabel("Área molhada (m²)")
    ax1.set_title("Área Molhada (método de Simpson)", fontsize=11.5, pad=10)
    ax1.set_ylim(0, max(AREA_Q["A"]) * 1.2)

    cores_Q = [COR_A, COR_A, COR_P0, COR_A, COR_A]
    ax2.bar(x, AREA_Q["Q"], width=0.65, color=cores_Q, alpha=0.9,
            edgecolor="black", linewidth=1.2)
    for i, v in enumerate(AREA_Q["Q"]):
        ax2.text(i, v + 0.1, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(PONTOS_NOMES, fontsize=11,
                                            fontweight="bold")
    ax2.set_xlabel("Ponto"); ax2.set_ylabel("Vazão (m³/s)")
    ax2.set_title("Vazão (equação da continuidade)", fontsize=11.5, pad=10)
    ax2.set_ylim(0, max(AREA_Q["Q"]) * 1.2)
    ax2.axhline(0.745, color=COR_TOTAL, lw=2.2, ls="--",
                label="Q90 = 0,745 m³/s")
    ax2.legend(loc="upper right", fontsize=9.5)

    fig.suptitle("Figura 7.5 — Área Molhada e Vazão por Seção (Campanha C3)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    salvar(fig, DIR_CAP7, "fig_7_5_areas_vazoes.png")


rodar(fig_7_1)
rodar(fig_7_2)
rodar(fig_7_3)
rodar(fig_7_4)
rodar(fig_7_5)
print()


# =============================================================================
# =========================== CAPÍTULO 8 ======================================
# =============================================================================
print("=" * 70)
print("  CAPÍTULO 8 — MODELAGEM QUAL-UFMG")
print("=" * 70)


def perfil_base(key, ylabel, ylim, titulo, padrao_val=None,
                padrao_lbl=None, sat=None, log=False, xmax=None):
    """Função genérica de perfil com gradiente."""
    if xmax is None:
        xmax = PONTOS_DIST["J2"]
    fig, ax = plt.subplots(figsize=(12, 5.5))

    xA, yA = resA8["x"], resA8[key]
    xB, yB = resB8["x"], resB8[key]

    if log:
        ax.semilogy(xA, yA, color=COR_A, lw=3.0, alpha=0.35, zorder=1)
        gradient_line(ax, xA, yA, cmap="Blues", lw=3.0, zorder=2)
        ax.semilogy(xB, yB, color=COR_B, lw=3.0, alpha=0.35, zorder=1)
        gradient_line(ax, xB, yB, cmap="Oranges", lw=3.0, zorder=2)
    else:
        ax.plot(xA, yA, color=COR_A, lw=3.0, alpha=0.35, zorder=1)
        gradient_line(ax, xA, yA, cmap="Blues", lw=3.0, zorder=2)
        ax.plot(xB, yB, color=COR_B, lw=3.0, alpha=0.35, zorder=1)
        gradient_line(ax, xB, yB, cmap="Oranges", lw=3.0, zorder=2)

    # Legendas "fake" (Line2D)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=COR_A, lw=3, label="Cenário A — Q = 2,973 m³/s"),
           Line2D([0], [0], color=COR_B, lw=3, ls="--", label="Cenário B — Q90 = 0,745 m³/s")]

    if sat:
        ax.axhline(sat[0], color=COR_SAT, lw=1.8, ls=":", label=sat[1])
    if padrao_val is not None:
        ax.axhline(padrao_val, color=COR_PADRAO, lw=2.2, ls="-.",
                   label=padrao_lbl, zorder=1)

    # Faixa de não conformidade
    if padrao_val is not None and ylim:
        if key == "OD":
            ax.axhspan(ylim[0], padrao_val, color="#FFEBEE", alpha=0.3, zorder=0)
        else:
            ax.axhspan(padrao_val, ylim[1], color="#FFEBEE", alpha=0.3, zorder=0)

    ax.axvline(0, color=COR_P0, lw=2.2, ls="--", alpha=0.9, label="P0 (lançamento)")

    for pt, xp in PONTOS_DIST.items():
        if pt != "P0" and xp > 0 and xp <= xmax:
            ax.axvline(xp, color="#B0B0B0", lw=0.9, ls=":", alpha=0.6)
            ax.text(xp, ylim[1] * 0.98, pt, ha="center",
                    fontsize=9, color="#666", fontweight="bold")

    ax.set_xlim(-20, xmax + 15)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Distância a partir de P0 (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def fig_8_1():
    fig = perfil_base("OD", "Oxigênio dissolvido (mg/L)", (5.5, 10.0),
        "Figura 8.1 — Perfil de Oxigênio Dissolvido\n"
        "Rio Araçatuba — Trecho de monitoramento (M2 → J2)",
        PADROES["OD"], f"Padrão C2 (≥ {PADROES['OD']} mg/L)",
        sat=(OD_SAT, f"OD saturação ({OD_SAT} mg/L)"))
    salvar(fig, DIR_CAP8, "fig_8_1_OD.png")


def fig_8_2():
    fig = perfil_base("DBO5", "DBO₅ (mg/L)", (18, 22),
        "Figura 8.2 — Perfil de DBO₅\n"
        "Condição pré-existente acima do padrão (M2 = 18,36 mg/L)",
        PADROES["DBO5"], f"Padrão C2 (≤ {PADROES['DBO5']} mg/L)")
    salvar(fig, DIR_CAP8, "fig_8_2_DBO5.png")


def fig_8_3():
    fig = perfil_base("N_NH3", "N-amoniacal (mg/L)", (0, 2.2),
        "Figura 8.3 — Perfil de N-amoniacal",
        PADROES["N_NH3"], f"Padrão C2 (≤ {PADROES['N_NH3']} mg/L)")
    salvar(fig, DIR_CAP8, "fig_8_3_NH3.png")


def fig_8_4():
    fig = perfil_base("N_NO3", "N-nitrato (mg/L)", (0, 11),
        "Figura 8.4 — Perfil de N-nitrato",
        PADROES["N_NO3"], f"Padrão C2 (≤ {PADROES['N_NO3']} mg/L)")
    salvar(fig, DIR_CAP8, "fig_8_4_NO3.png")


def fig_8_5():
    fig = perfil_base("P_total", "Fósforo total (mg/L)", (0, 0.15),
        "Figura 8.5 — Perfil de Fósforo Total",
        PADROES["P_total"], f"Padrão C2 (≤ {PADROES['P_total']} mg/L)")
    salvar(fig, DIR_CAP8, "fig_8_5_Ptotal.png")


def fig_8_6():
    fig = perfil_base("Coli", "Coliformes (NMP/100 mL)", (50, 200),
        "Figura 8.6 — Perfil de Coliformes Termotolerantes (escala log)",
        PADROES["Coli"], f"Padrão C2 (≤ {int(PADROES['Coli'])} NMP/100 mL)",
        log=True)
    salvar(fig, DIR_CAP8, "fig_8_6_coliformes.png")


def fig_8_7():
    """Painel consolidado 3x2."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    params = [
        ("OD", "OD (mg/L)", (5.5, 10.0), 5.0, "≥ 5,0", False),
        ("DBO5", "DBO₅ (mg/L)", (18, 22), 5.0, "≤ 5,0", False),
        ("N_NH3", "N-NH₃ (mg/L)", (0, 2.2), 2.0, "≤ 2,0", False),
        ("N_NO3", "N-NO₃ (mg/L)", (0, 11), 10.0, "≤ 10,0", False),
        ("P_total", "P-total (mg/L)", (0, 0.15), 0.10, "≤ 0,10", False),
        ("Coli", "Coliformes (NMP/100mL)", (50, 200), 1000.0, "≤ 1000", True),
    ]
    for ax, (key, ylbl, ylim, pval, plbl, use_log) in zip(axes.flat, params):
        xA, yA = resA8["x"], resA8[key]
        xB, yB = resB8["x"], resB8[key]
        if use_log:
            ax.semilogy(xA, yA, color=COR_A, lw=2.2, label="Cenário A")
            ax.semilogy(xB, yB, color=COR_B, lw=2.2, ls="--",
                        label="Cenário B")
        else:
            gradient_line(ax, xA, yA, cmap="Blues", lw=2.5, zorder=2)
            gradient_line(ax, xB, yB, cmap="Oranges", lw=2.5, zorder=2)
            ax.plot([], [], color=COR_A, lw=2.5, label="Cenário A")
            ax.plot([], [], color=COR_B, lw=2.5, ls="--", label="Cenário B")
        ax.axhline(pval, color=COR_PADRAO, lw=1.6, ls="-.")
        ax.axvline(0, color=COR_P0, lw=1.6, ls="--", alpha=0.7)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylbl, fontsize=9.5)
        ax.text(0.98, 0.95, plbl, transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=COR_PADRAO, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Distância de P0 (m)")
    fig.suptitle("Figura 8.7 — Painel Consolidado dos Parâmetros Modelados",
                 fontsize=13.5, fontweight="bold", y=1.00)
    fig.tight_layout()
    salvar(fig, DIR_CAP8, "fig_8_7_painel_consolidado.png")


def fig_8_8():
    """PLUMA 3D de Fósforo Total — CENÁRIO B."""
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection="3d")

    x_lim = 310
    nx, ny = 120, 60
    x3 = np.linspace(0, x_lim, nx)
    y3 = np.linspace(-5, 5, ny)
    X3, Y3 = np.meshgrid(x3, y3)

    # Concentração central (do modelo)
    P_center = np.interp(x3, resB8["x"], resB8["P_total"])
    P_bg = 0.082

    # Coeficientes de dispersão (rio pequeno)
    U_B = CEN_B["V"]
    h = 1.78; S = 0.000215
    u_star = np.sqrt(9.81 * h * S)
    eps_y = 0.15 * h * u_star
    sigma_y0 = 0.5
    sig = np.sqrt(sigma_y0**2 + 2 * eps_y * np.maximum(x3, 1e-6) / U_B)

    # Concentração 2D
    Z3 = np.zeros_like(X3)
    for j, (s_, cv) in enumerate(zip(sig, P_center)):
        lat = np.exp(-Y3[:, j]**2 / (2 * s_**2))
        lat_n = lat / max(lat.max(), 1e-12)
        Z3[:, j] = P_bg + (cv - P_bg) * lat_n

    # Superfície
    surf = ax.plot_surface(X3, Y3, Z3, cmap=CMAP_PLUME,
                            edgecolor="none", alpha=0.95,
                            rstride=2, cstride=2, linewidth=0,
                            antialiased=True, vmin=P_bg, vmax=0.12)

    # Plano do padrão CONAMA
    Z_pad = np.full_like(X3, PADROES["P_total"])
    ax.plot_wireframe(X3, Y3, Z_pad, color=COR_PADRAO, linewidth=1.0,
                       alpha=0.55, rstride=8, cstride=8)

    # Ponto P0
    ax.scatter([0], [0], [P_bg + 0.02], s=400, marker="*",
               color=COR_P0, edgecolor="black", linewidth=2, zorder=10,
               label="P0 (lançamento)")

    # Contorno no fundo (sombra)
    ax.contour(X3, Y3, Z3, zdir="z", offset=0.05, levels=12,
               cmap=CMAP_PLUME, alpha=0.6, linewidths=1.2)

    ax.set_xlabel("Distância a partir de P0 (m)", fontsize=10.5, labelpad=12)
    ax.set_ylabel("Posição lateral no canal (m)", fontsize=10.5, labelpad=12)
    ax.set_zlabel("Fósforo total (mg/L)", fontsize=10.5, labelpad=12)
    ax.set_title("Figura 8.8 — Pluma de Dispersão 3D do Fósforo Total\n"
                 "Cenário B (Q90 = 0,745 m³/s) — Padrão CONAMA = 0,10 mg/L",
                 fontsize=12.5, fontweight="bold", pad=20)
    ax.view_init(elev=22, azim=-58)
    ax.set_zlim(0.05, 0.13)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, aspect=22)
    cbar.set_label("Fósforo total (mg/L)", fontsize=10.5)

    fig.tight_layout()
    salvar(fig, DIR_CAP8, "fig_8_8_pluma_3d_Ptotal.png")


rodar(fig_8_1)
rodar(fig_8_2)
rodar(fig_8_3)
rodar(fig_8_4)
rodar(fig_8_5)
rodar(fig_8_6)
rodar(fig_8_7)
rodar(fig_8_8)
print()


# =============================================================================
# =========================== CAPÍTULO 9 ======================================
# =============================================================================
print("=" * 70)
print("  CAPÍTULO 9 — CONCEPÇÃO DO EMISSÁRIO")
print("=" * 70)


def fig_9_1():
    """Perfil 3 planos com gradiente."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for ax, (nome, d) in zip(axes, PLANOS.items()):
        L = d["L"]
        V1 = d["V1"]; V3 = d.get("V3", d["V2"]); V2 = d.get("V2")

        if nome == "C":
            # 2 vértices
            dd = [V1[0], V3[0]]
            cc = [V1[1], V3[1]]
            cr = [V1[2], V3[2]]
        else:
            dd = [V1[0], V2[0], V3[0]]
            cc = [V1[1], V2[1], V3[1]]
            cr = [V1[2], V2[2], V3[2]]

        f_t = interp1d(dd, cc, kind="linear")
        f_c = interp1d(dd, cr, kind="linear")
        df = np.linspace(0, L, 300)
        ct = f_t(df); ccrown = f_c(df)

        # Aterro
        recob = ccrown + 1.0
        mask = recob > ct
        ax.fill_between(df, ct, recob, where=mask, color="#A0826D",
                        alpha=0.5, label="Aterro necessário")

        # Terreno com gradiente
        gradient_line(ax, df, ct, cmap="copper", lw=2.8, zorder=3)
        ax.plot([], [], color=COR_TERRENO, lw=2.8, label="Terreno natural")

        # Crown
        gradient_line(ax, df, ccrown, cmap="Blues", lw=3.0, zorder=4)
        ax.plot([], [], color=COR_A, lw=3.0, label="Crown (topo do tubo)")

        # Vértices
        for vn, v in [("V1", V1), ("V2", V2), ("V3", d.get("V3"))]:
            if v is None:
                continue
            ax.scatter(v[0], v[1], s=140, marker="o", facecolor="white",
                       edgecolor=COR_TERRENO, linewidth=2.5, zorder=6)
            ax.annotate(f"{vn}\n{v[1]:.2f} m", xy=(v[0], v[1]),
                        xytext=(0, 16), textcoords="offset points",
                        ha="center", fontsize=9.5, fontweight="bold",
                        color="#333",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="#CCC", lw=0.8))

        # Poço (soluções com bombeamento)
        if V2:
            ax.scatter(V2[0], V2[1], s=400, marker="s",
                       facecolor=COR_B, edgecolor="black", linewidth=1.5,
                       zorder=7, label="Poço de sucção")

        ax.set_xlim(-10, L + 10)
        ax.set_ylim(min(ct) - 0.5, max(ct) + 2.5)
        ax.set_xlabel("Distância a partir da ETE (m)", fontsize=10)
        ax.set_ylabel("Cota (m)", fontsize=10)
        ax.set_title(f"Plano {nome}  —  Comprimento total: {L:.1f} m",
                     fontsize=11.5, pad=8)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("Figura 9.1 — Perfil Longitudinal do Emissário — 3 Traçados",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    salvar(fig, DIR_CAP9, "fig_9_1_perfil_emissario_3_planos.png")


def fig_9_2():
    """Comparação diâmetros comerciais."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    labels = [f"DN {d}" for d in DIAMETROS["DN"]]
    cores = [COR_TOTAL if d == 100 else COR_A for d in DIAMETROS["DN"]]

    bars = ax1.bar(labels, DIAMETROS["V"], color=cores, alpha=0.88,
                    edgecolor="black", linewidth=1.3, width=0.6)
    for i, v in enumerate(DIAMETROS["V"]):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax1.axhline(0.50, color=COR_EFETIVA, lw=2, ls="--",
                label="V mínima = 0,50 m/s")
    ax1.axhline(3.00, color=COR_TOTAL, lw=2, ls="--",
                label="V máxima = 3,00 m/s")
    ax1.set_ylabel("Velocidade (m/s)", fontsize=11)
    ax1.set_title("Velocidade por diâmetro", fontsize=12, pad=10)
    ax1.legend(fontsize=9.5, loc="upper right")
    ax1.set_ylim(0, max(DIAMETROS["V"]) * 1.35)

    ax2.bar(labels, DIAMETROS["J"], color=cores, alpha=0.88,
            edgecolor="black", linewidth=1.3, width=0.6)
    for i, v in enumerate(DIAMETROS["J"]):
        ax2.text(i, v + 0.0004, f"{v:.5f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_ylabel("J (m/m)", fontsize=11)
    ax2.set_title("Perda de carga unitária", fontsize=12, pad=10)
    ax2.set_ylim(0, max(DIAMETROS["J"]) * 1.35)

    fig.suptitle("Figura 9.2 — Comparação de Diâmetros Comerciais "
                 "(Qmáx = 4,17 L/s)\nAdotado: DN 100 (PEAD PE 100)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    salvar(fig, DIR_CAP9, "fig_9_2_comparacao_diametros.png")


def fig_9_3():
    """Comparação custos das soluções."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [1.2, 1]})
    nomes = list(SOLUCOES.keys())
    y = np.arange(len(nomes))
    ini = [SOLUCOES[n]["ini"] for n in nomes]
    op = [SOLUCOES[n]["op"] for n in nomes]
    tot = [SOLUCOES[n]["tot"] for n in nomes]

    cores = [COR_B if "Bomb" in n else "#8B7355" for n in nomes]
    ax1.barh(y, ini, color=cores, alpha=0.88, height=0.6,
             edgecolor="black", linewidth=0.8, label="Custo inicial")
    ax1.barh(y, op, left=ini, color="#F4DCCF", alpha=0.95, height=0.6,
             edgecolor="black", linewidth=0.8, label="Custo operacional (20 anos)")

    for i, t in enumerate(tot):
        ax1.text(t + 5000, i, f"R$ {t:,.0f}", va="center",
                 fontsize=10, fontweight="bold")
    ax1.set_yticks(y); ax1.set_yticklabels(nomes, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Custo total em 20 anos (R$)", fontsize=10.5)
    ax1.set_title("Comparação de Custos", fontsize=12, pad=10)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlim(0, max(tot) * 1.28)

    rank = sorted(SOLUCOES.items(), key=lambda x: x[1]["tot"])
    nr = [r[0] for r in rank]
    vr = [r[1]["tot"] for r in rank]
    cr = [COR_A if "Bomb" in n else "#8B7355" for n in nr]
    ax2.barh(range(len(nr)), vr, color=cr, alpha=0.92, height=0.6,
             edgecolor="black", linewidth=0.8)
    for i, v in enumerate(vr):
        ax2.text(v + 3000, i, f"{i+1}º  R$ {v:,.0f}", va="center",
                 fontsize=10.5, fontweight="bold")
    ax2.set_yticks(range(len(nr))); ax2.set_yticklabels(nr, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Custo total (R$)", fontsize=10.5)
    ax2.set_title("Ranking (menor custo primeiro)", fontsize=12, pad=10)
    ax2.set_xlim(0, max(vr) * 1.38)
    ax2.annotate("Solução recomendada",
                 xy=(vr[0], 0), xytext=(vr[0] * 0.35, 1.7),
                 fontsize=10.5, color=COR_A, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec=COR_A, lw=1.3),
                 arrowprops=dict(arrowstyle="->", color=COR_A, lw=1.6))

    fig.suptitle("Figura 9.3 — Análise Comparativa Técnico-Econômica\n"
                 "Aterro topográfico × Estação elevatória",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    salvar(fig, DIR_CAP9, "fig_9_3_comparacao_solucoes.png")


def fig_9_4():
    """Perdas de carga por solução."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sols = ["A1", "B1", "C1", "A2", "B2", "C2"]
    dH = [2.587, 2.592, 2.716, 6.136, 6.082, 5.509]
    cores = [COR_PADRAO if "1" in s else COR_A for s in sols]

    bars = ax.bar(sols, dH, color=cores, alpha=0.88,
                  edgecolor="black", linewidth=1.3, width=0.6)
    for b, v in zip(bars, dH):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.3f} m",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Perda de carga total (m)", fontsize=11)
    ax.set_xlabel("Solução técnica", fontsize=11)
    ax.set_title("Figura 9.4 — Perda de Carga Total por Solução\n"
                 "Aterro (gravidade) × Bombeamento (recalque)",
                 fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COR_PADRAO, alpha=0.88, label="Aterro"),
        Patch(facecolor=COR_A, alpha=0.88, label="Bombeamento"),
    ], loc="upper left", fontsize=10)
    ax.set_ylim(0, max(dH) * 1.22)
    salvar(fig, DIR_CAP9, "fig_9_4_perdas_carga_solucoes.png")


def fig_9_5():
    """Isométrico 3D dos 3 traçados."""
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")

    cores_planos = {"A": "#E74C3C", "B": COR_A, "C": "#27AE60"}

    for nome, d in PLANOS.items():
        L = d["L"]
        V1 = d["V1"]
        V3 = d.get("V3", d["V2"])
        V2 = d.get("V2")

        if nome == "C":
            pts_d = [V1[0], V3[0]]
            pts_c = [V1[1], V3[1]]
        else:
            pts_d = [V1[0], V2[0], V3[0]]
            pts_c = [V1[1], V2[1], V3[1]]

        f = interp1d(pts_d, pts_c, kind="linear")
        df = np.linspace(0, L, 200)
        cf = f(df)

        # Perfil do terreno
        ax.plot(df, np.zeros_like(df), cf,
                color=cores_planos[nome], lw=3, label=f"Plano {nome}")

        # Marca V1
        ax.scatter([df[0]], [0], [cf[0]], s=200, marker="o",
                   color=cores_planos[nome], edgecolor="black",
                   linewidth=1.5, zorder=10)
        # Marca V3
        ax.scatter([df[-1]], [0], [cf[-1]], s=300, marker="*",
                   color=COR_P0, edgecolor="black", linewidth=1.5, zorder=10)

    ax.set_xlabel("Distância da ETE (m)", fontsize=10.5, labelpad=12)
    ax.set_ylabel("Lateral (m)", fontsize=10.5, labelpad=12)
    ax.set_zlabel("Cota (m)", fontsize=10.5, labelpad=12)
    ax.set_title("Figura 9.5 — Vista Isométrica 3D dos 3 Traçados do Emissário\n"
                 "★ = ponto de lançamento",
                 fontsize=12.5, fontweight="bold", pad=18)
    ax.view_init(elev=25, azim=-70)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    salvar(fig, DIR_CAP9, "fig_9_5_isometrico_3d.png")


rodar(fig_9_1)
rodar(fig_9_2)
rodar(fig_9_3)
rodar(fig_9_4)
rodar(fig_9_5)
print()


# =============================================================================
# =========================== CAPÍTULO 10 =====================================
# =============================================================================
print("=" * 70)
print("  CAPÍTULO 10 — ANÁLISE INTEGRADA")
print("=" * 70)


def fig_10_1():
    """Semáforo de conformidade."""
    def razao(res):
        params = ["OD", "DBO5", "N_NH3", "N_NO2", "N_NO3", "P_total", "Coli"]
        padr = [PADROES["OD"], PADROES["DBO5"], PADROES["N_NH3"],
                PADROES["N_NO2"], PADROES["N_NO3"], PADROES["P_total"],
                PADROES["Coli"]]
        pts = ["P0", "J1", "J2"]
        M = np.zeros((len(params), len(pts)))
        for i, p in enumerate(params):
            for j, pt in enumerate(pts):
                idx = int(np.argmin(np.abs(res["x"] - PONTOS_DIST[pt])))
                v = res[p][idx]
                M[i, j] = padr[i] / v if p == "OD" else v / padr[i]
        return M

    MA = razao(resA8)
    MB = razao(resB8)
    pts = ["P0", "J1", "J2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, M, tit in [(axes[0], MA, "Cenário A"),
                       (axes[1], MB, "Cenário B (Q90)")]:
        im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=2.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(pts, fontsize=12,
                                                     fontweight="bold")
        ax.set_yticks(range(7))
        ax.set_yticklabels(["OD", "DBO₅", "N-NH₃", "N-NO₂", "N-NO₃",
                            "P-total", "Coliformes"], fontsize=10.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                r = M[i, j]
                if r <= 0.5:
                    txt, c = "OK", "white"
                elif r <= 1.0:
                    txt, c = f"{r:.2f}\nOK", "black"
                elif r <= 2.0:
                    txt, c = f"{r:.2f}\n⚠", "white"
                else:
                    txt, c = f"{r:.1f}×\n✗", "white"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9.5, color=c, fontweight="bold")
        ax.set_title(tit, fontsize=12.5, pad=10)
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.025)
    cbar.set_label("Razão (concentração / padrão CONAMA 357/2005)\n"
                   "Verde = conforme  |  Vermelho = não conforme",
                   fontsize=9.5)
    cbar.ax.axhline(1.0, color="black", lw=2)

    fig.suptitle("Figura 10.1 — Semáforo de Conformidade Ambiental",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    salvar(fig, DIR_CAP10, "fig_10_1_semaforo_conformidade.png")


def fig_10_2():
    """Balanço de cargas."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cargas = [
        ("DBO₅", RIO_M1["DBO5"], EFLUENTE["DBO5"]),
        ("NTK", RIO_M1["N_NH3"] + RIO_M1["N_org"],
         EFLUENTE["N_NH3"] + EFLUENTE["N_org"]),
        ("P-total", RIO_M1["P_org"] + RIO_M1["P_inorg"],
         EFLUENTE["P_org"] + EFLUENTE["P_inorg"]),
    ]
    Qr, Qe = CEN_B["Q"], EFLUENTE["Q"]
    for ax, (nome, cr, ce) in zip(axes, cargas):
        c_rio = Qr * cr * 86.4
        c_eff = Qe * ce * 86.4
        tot = c_rio + c_eff
        ax.barh([1], [c_rio], color=COR_A, alpha=0.92, height=0.5,
                label="Rio a montante")
        ax.barh([1], [c_eff], left=[c_rio], color=COR_P0, alpha=0.95,
                height=0.5, label="Efluente ETE")
        ax.text(c_rio / 2, 1,
                f"{c_rio:.1f} kg/d\n({100*c_rio/tot:.2f}%)",
                ha="center", va="center", color="white",
                fontsize=11.5, fontweight="bold")
        ax.annotate(f"Efluente:\n{c_eff:.2f} kg/d ({100*c_eff/tot:.3f}%)",
                    xy=(c_rio + c_eff/2, 1.3),
                    xytext=(c_rio * 0.7, 1.75),
                    fontsize=10, color=COR_B, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec=COR_B, lw=1.3),
                    arrowprops=dict(arrowstyle="->", color=COR_B, lw=1.4))
        ax.set_yticks([]); ax.set_xlim(0, tot * 1.15)
        ax.set_xlabel("Carga (kg/dia)", fontsize=10.5)
        ax.set_title(nome, fontsize=12.5, pad=10)
        ax.set_ylim(0.5, 2.0)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Figura 10.2 — Balanço de Cargas — Rio × Efluente (Q90)\n"
                 "Contribuição do empreendimento < 0,4% da carga total",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    salvar(fig, DIR_CAP10, "fig_10_2_balanco_cargas.png")


def fig_10_3():
    """Conformidade ao longo do rio."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    v_m2 = {"DBO5": 18.36, "N_NH3": 0.05, "N_NO3": 0.40, "P_total": 0.11}
    v_m1 = {"DBO5": 20.40, "N_NH3": 0.05, "N_NO3": 0.40, "P_total": 0.08}
    series = [
        ("DBO5", "DBO₅", v_m2["DBO5"], v_m1["DBO5"], "#B22222"),
        ("N_NH3", "N-NH₃", v_m2["N_NH3"], v_m1["N_NH3"], "#2E7D32"),
        ("N_NO3", "N-NO₃", v_m2["N_NO3"], v_m1["N_NO3"], "#1565C0"),
        ("P_total", "P-total", v_m2["P_total"], v_m1["P_total"], "#7B1FA2"),
    ]
    for key, lbl, vm2, vm1, cor in series:
        p = PADROES[key]
        ax.plot([-200, -100], [100*vm2/p, 100*vm1/p], color=cor, lw=2.8,
                ls="--", alpha=0.7, marker="o", markersize=9,
                markeredgecolor="white", markeredgewidth=2)
        ax.plot([-100, 0], [100*vm1/p, 100*resB8[key][0]/p],
                color=cor, lw=2.8, ls="--", alpha=0.7)
        gradient_line(ax, resB8["x"], 100*resB8[key]/p, cmap="Reds" if cor == "#B22222" else "Greens",
                      lw=3.0, zorder=3)
        ax.plot([], [], color=cor, lw=3, label=lbl)

    ax.axhline(100, color=COR_PADRAO, lw=2.5, ls="-.",
               label="Padrão CONAMA (100%)", zorder=1)
    ax.axhspan(0, 100, color="#E8F5E9", alpha=0.35, zorder=0)
    ax.axhspan(100, 500, color="#FFEBEE", alpha=0.35, zorder=0)
    for pt, xp in PONTOS_DIST.items():
        ax.axvline(xp, color="#B0B0B0", lw=0.9, ls=":", zorder=0)
        ax.text(xp, 470, pt, ha="center", fontsize=10,
                fontweight="bold", color="#444",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#CCC", lw=0.7))
    ax.axvline(0, color=COR_P0, lw=2.2, ls="--", alpha=0.9, zorder=2)

    ax.set_xlim(-220, 340); ax.set_ylim(0, 500)
    ax.set_xlabel("Distância a partir de P0 (m)", fontsize=11)
    ax.set_ylabel("Concentração em relação ao padrão (%)", fontsize=11)
    ax.set_title("Figura 10.3 — Conformidade Ambiental ao Longo do Rio\n"
                 "Linhas acima de 100% indicam não conformidade",
                 fontsize=12.5, pad=12)
    ax.legend(loc="upper right", fontsize=9.5, ncol=2)
    fig.tight_layout()
    salvar(fig, DIR_CAP10, "fig_10_3_conformidade_rio.png")


def fig_10_4():
    """Pluviometria comparativa."""
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(PLUV_MESES))
    ax.bar(x - 0.21, PLUV_MED, width=0.42, color="#4A90E2",
           edgecolor="black", linewidth=0.8,
           label="Medido (CEMADEN 2025-2026)")
    ax.bar(x + 0.21, PLUV_HIST, width=0.42, color="#E5A54B",
           edgecolor="black", linewidth=0.8, alpha=0.8,
           label="Média histórica (30 anos)")

    for i, v in enumerate(PLUV_MED):
        ax.text(i - 0.21, v + 2, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8.5, color="#333", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(PLUV_MESES, rotation=45,
                                          ha="right", fontsize=10)
    ax.set_ylabel("Precipitação mensal (mm)", fontsize=11)
    ax.set_xlabel("Mês / Ano", fontsize=11)
    ax.set_title("Figura 10.4 — Regime Pluviométrico Comparativo\n"
                 "Estação Campo D'Una (CEMADEN) — Set/2025 a Set/2026",
                 fontsize=12.5, pad=12)

    total = sum(PLUV_MED)
    ax.text(0.98, 0.95,
            f"Total medido: {total:.1f} mm\nMédia histórica: 1450 mm/ano",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5",
                      ec="#999", lw=1.2))
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    salvar(fig, DIR_CAP10, "fig_10_4_pluviometria.png")


rodar(fig_10_1)
rodar(fig_10_2)
rodar(fig_10_3)
rodar(fig_10_4)
print()


# =============================================================================
# =========================== ANEXO A2 ========================================
# =============================================================================
print("=" * 70)
print("  ANEXO A2 — SIMULAÇÃO ESTENDIDA ATÉ 1.500 m")
print("=" * 70)


def perfil_a2(key, ylbl, ylim, tit, pval, plbl, sat=None, log=False):
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    xA, yA = resA2["x"], resA2[key]
    xB, yB = resB2["x"], resB2[key]

    if log:
        ax.semilogy(xA, yA, color=COR_A, lw=3, alpha=0.35, zorder=1)
        gradient_line(ax, xA, yA, cmap="Blues", lw=3, zorder=2)
        ax.semilogy(xB, yB, color=COR_B, lw=3, alpha=0.35, zorder=1)
        gradient_line(ax, xB, yB, cmap="Oranges", lw=3, zorder=2)
    else:
        gradient_line(ax, xA, yA, cmap="Blues", lw=3, zorder=2)
        gradient_line(ax, xB, yB, cmap="Oranges", lw=3, zorder=2)

    from matplotlib.lines import Line2D
    if sat:
        ax.axhline(sat[0], color=COR_SAT, lw=1.8, ls=":", label=sat[1])
    ax.axhline(pval, color=COR_PADRAO, lw=2.2, ls="-.",
               label=plbl, zorder=1)
    ax.axvline(0, color=COR_P0, lw=2.2, ls="--", alpha=0.9, label="P0")

    for xp in [310, 500, 1000]:
        ax.axvline(xp, color="#CCCCCC", lw=0.9, ls=":", alpha=0.7)
        ax.text(xp, ylim[1]*0.98, f"{xp} m", ha="center",
                fontsize=8.5, color="#666")

    ax.set_xlim(-20, 1520); ax.set_ylim(*ylim)
    ax.set_xlabel("Distância a partir de P0 (m)")
    ax.set_ylabel(ylbl)
    ax.set_title(tit, fontweight="bold")
    leg = [Line2D([0], [0], color=COR_A, lw=3, label="Cenário A — Q = 2,973 m³/s"),
           Line2D([0], [0], color=COR_B, lw=3, ls="--", label="Cenário B — Q90 = 0,745 m³/s")]
    ax.legend(handles=leg + ax.get_legend_handles_labels()[0],
              loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def fig_i1():
    fig = perfil_a2("OD", "OD (mg/L)", (5.5, 10.0),
        "Anexo A2 — Figura I.1\nPerfil de OD — Simulação Estendida até 1.500 m",
        PADROES["OD"], f"Padrão C2 (≥ {PADROES['OD']} mg/L)",
        sat=(OD_SAT, f"OD saturação ({OD_SAT} mg/L)"))
    salvar(fig, DIR_ANEXO, "fig_anexo_I_OD_1500m.png")


def fig_i2():
    fig = perfil_a2("DBO5", "DBO₅ (mg/L)", (20, 21),
        "Anexo A2 — Figura I.2\nPerfil de DBO₅ — Simulação Estendida até 1.500 m",
        PADROES["DBO5"], f"Padrão C2 (≤ {PADROES['DBO5']} mg/L)")
    salvar(fig, DIR_ANEXO, "fig_anexo_I_DBO5_1500m.png")


def fig_i3():
    fig = perfil_a2("N_NH3", "N-amoniacal (mg/L)", (0, 2.2),
        "Anexo A2 — Figura I.3\nPerfil de N-amoniacal — Até 1.500 m",
        PADROES["N_NH3"], f"Padrão C2 (≤ {PADROES['N_NH3']} mg/L)")
    salvar(fig, DIR_ANEXO, "fig_anexo_I_NH3_1500m.png")


def fig_i4():
    fig = perfil_a2("N_NO3", "N-nitrato (mg/L)", (0, 11),
        "Anexo A2 — Figura I.4\nPerfil de N-nitrato — Até 1.500 m",
        PADROES["N_NO3"], f"Padrão C2 (≤ {PADROES['N_NO3']} mg/L)")
    salvar(fig, DIR_ANEXO, "fig_anexo_I_NO3_1500m.png")


def fig_i5():
    fig = perfil_a2("P_total", "P-total (mg/L)", (0, 0.15),
        "Anexo A2 — Figura I.5\nPerfil de Fósforo Total — Até 1.500 m",
        PADROES["P_total"], f"Padrão C2 (≤ {PADROES['P_total']} mg/L)")
    salvar(fig, DIR_ANEXO, "fig_anexo_I_Ptotal_1500m.png")


def fig_i6():
    fig = perfil_a2("Coli", "Coliformes (NMP/100 mL)", (50, 200),
        "Anexo A2 — Figura I.6\nPerfil de Coliformes (escala log) — Até 1.500 m",
        PADROES["Coli"], f"Padrão C2 (≤ {int(PADROES['Coli'])} NMP/100 mL)",
        log=True)
    salvar(fig, DIR_ANEXO, "fig_anexo_I_coliformes_1500m.png")


rodar(fig_i1)
rodar(fig_i2)
rodar(fig_i3)
rodar(fig_i4)
rodar(fig_i5)
rodar(fig_i6)
print()


# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
print("=" * 70)
print("  RELATÓRIO FINAL")
print("=" * 70)

total = 0
for nome, d in [("Cap 7", DIR_CAP7), ("Cap 8", DIR_CAP8),
                ("Cap 9", DIR_CAP9), ("Cap 10", DIR_CAP10),
                ("Anexo A2", DIR_ANEXO)]:
    arqs = sorted(d.glob("*.png"))
    total += len(arqs)
    print(f"\n  {nome} — {d.name}  ({len(arqs)} figuras):")
    for a in arqs:
        tamanho = a.stat().st_size / 1024
        print(f"    ✓ {a.name}  ({tamanho:.0f} KB)")

print("\n" + "=" * 70)
print(f"  TOTAL: {total} figuras PNG geradas em 400 DPI")
print(f"  Localização: {SCRIPT_DIR}")
print("=" * 70)
print("\n  Estatísticas (Cenário B — Anexo A2):")
print(f"    OD mínimo:              {resB2['OD'].min():.2f} mg/L")
print(f"    DBO₅ em J2 (296 m):     {resB2['DBO5'][296]:.2f} mg/L")
print(f"    P-total em J2 (296 m):  {resB2['P_total'][296]:.4f} mg/L")
print(f"    N-NH₃ em J2 (296 m):    {resB2['N_NH3'][296]:.4f} mg/L")
print(f"    Coliformes em J2:       {resB2['Coli'][296]:.1f} NMP/100mL")
print("=" * 70)
print("\n  ✓ Script concluído com sucesso!")