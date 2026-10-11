# =====================================================================
# PAINEL RESUMO - ANEXO A2 (2 KM)
# Figura unica com 6 subplots - todos os parametros
# Autor: Eng. Muriell Bernardo Macedo (CREA-SC 222578-3)
# =====================================================================

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "axes.titlepad": 8,
    "axes.edgecolor": "#2B2B2B", "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "output_final_2km"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# PARAMETROS (referencia M1 - igual ao Capitulo 8)
# ---------------------------------------------------------------------
Q_RIO = 0.745
Q_EF = 0.00234
S_DECL = 0.000215
V_M1 = 0.10494
Q_M1 = 2.973

V_Q90 = V_M1 * np.sqrt(Q_RIO / Q_M1)
K2_A = 0.800
K2_Q90 = K2_A * np.sqrt(V_Q90 / V_M1)

# ---------------------------------------------------------------------
# DADOS DE ENTRADA
# ---------------------------------------------------------------------
RIO_M1 = dict(DBO5=20.40, OD=6.50, N_org=0.16, N_NH3=0.05,
              N_NO2=0.03, N_NO3=0.40, P_org=0.04, P_inorg=0.04, Coli=78.0)
EFLUENTE = dict(DBO5=22.40, OD=4.00, N_org=1.93, N_NH3=0.05,
                N_NO2=0.0, N_NO3=24.47, P_org=1.51, P_inorg=1.51, Coli=1000.0)
COEF = dict(Kd=0.134, Ks=0.0, Kso=0.05, Koa=0.20, Kan=0.20, Knn=0.75,
            Kspo=0.02, Koi=0.27, Kspi=0.02, Kb=1.0,
            O2Namon=3.3, O2Nnitri=1.1, k_nitr=0.6, S_NH3=0.0)
OD_SAT = 9.48
PROF = 1.78
F_DBO = 1.46
PADROES = dict(OD=5.0, DBO5=5.0, N_NH3=2.0, N_NO3=10.0,
               P_total=0.10, Coli=1000.0)
X_MAX = 2000.0
N_PONTOS = 2000

# ---------------------------------------------------------------------
# CONDICAO INICIAL EM P0
# ---------------------------------------------------------------------
Q_total = Q_RIO + Q_EF
C0 = {k: (Q_RIO * RIO_M1[k] + Q_EF * EFLUENTE[k]) / Q_total for k in RIO_M1}

# ---------------------------------------------------------------------
# MODELO QUAL-UFMG
# ---------------------------------------------------------------------
def ode(x, y):
    U = V_Q90 * 86400.0
    K2 = K2_Q90
    L, OD, Norg, NNH3, NNO2, NNO3, Porg, Pinorg, Coli = y
    f = OD / (OD + COEF["k_nitr"]) if (OD + COEF["k_nitr"]) > 0 else 0.0
    return [
        -(COEF["Kd"] + COEF["Ks"]) * L / U,
        (K2 * (OD_SAT - OD) - COEF["Kd"] * L
         - COEF["O2Namon"] * NNH3 * COEF["Kan"] * f
         - COEF["O2Nnitri"] * NNO2 * COEF["Knn"] * f) / U,
        -(COEF["Kso"] + COEF["Koa"]) * Norg / U,
        (COEF["Koa"] * Norg - COEF["Kan"] * NNH3 * f
         + COEF["S_NH3"] / PROF) / U,
        (COEF["Kan"] * NNH3 * f - COEF["Knn"] * NNO2 * f) / U,
        COEF["Knn"] * NNO2 * f / U,
        -(COEF["Kspo"] + COEF["Koi"]) * Porg / U,
        (COEF["Koi"] * Porg - COEF["Kspi"] * Pinorg) / U,
        -COEF["Kb"] * Coli / U,
    ]

y0 = [C0["DBO5"] * F_DBO, C0["OD"], C0["N_org"], C0["N_NH3"],
      C0["N_NO2"], C0["N_NO3"], C0["P_org"], C0["P_inorg"], C0["Coli"]]

print("[INFO] Simulando ate " + str(int(X_MAX)) + " m...")
sol = solve_ivp(ode, (0, X_MAX), y0, method="RK45",
                rtol=1e-6, atol=1e-9, dense_output=True, max_step=10)
xg = np.linspace(0, X_MAX, N_PONTOS)
Y = sol.sol(xg)
L, OD, Norg, NNH3, NNO2, NNO3, Porg, Pinorg, Coli = Y
res = dict(x=xg, OD=OD, DBO5=L/F_DBO, N_NH3=NNH3, N_NO3=NNO3,
           P_total=Porg + Pinorg, Coli=Coli)

# ---------------------------------------------------------------------
# DADOS DE CAMPO (para sobrepor)
# ---------------------------------------------------------------------
PONTOS_CAMPO = {
    "J1": dict(x=109.17, OD=6.10, DBO5=21.42, N_NH3=0.05,
               N_NO3=0.40, P_total=0.07, Coli=44.0),
    "J2": dict(x=310.51, OD=6.10, DBO5=22.44, N_NH3=0.05,
               N_NO3=0.40, P_total=0.15, Coli=71.0),
}

# =====================================================================
# PAINEL 3x2 - RESULTADO FINAL
# =====================================================================
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.07,
                    hspace=0.42, wspace=0.22)

# Configuracao dos 6 paineis
paineis = [
    (axes[0, 0], "OD", "Oxigenio Dissolvido (mg/L)",
     "A - OD", PADROES["OD"], ">=", False),
    (axes[0, 1], "DBO5", "DBO5 (mg/L)",
     "B - DBO5", PADROES["DBO5"], "<=", False),
    (axes[1, 0], "N_NH3", "N-amoniacal (mg/L)",
     "C - N-amoniacal", PADROES["N_NH3"], "<=", False),
    (axes[1, 1], "N_NO3", "N-nitrato (mg/L)",
     "D - N-nitrato", PADROES["N_NO3"], "<=", False),
    (axes[2, 0], "P_total", "Fosforo Total (mg/L)",
     "E - Fosforo Total", PADROES["P_total"], "<=", False),
    (axes[2, 1], "Coli", "Coliformes (NMP/100 mL)",
     "F - Coliformes", PADROES["Coli"], "<=", True),
]

for ax, key, ylabel, titulo, padrao, op, log in paineis:
    y_model = res[key]
    y_min = float(np.min(y_model))
    y_max = float(np.max(y_model))
    y_range = y_max - y_min

    # Limites Y (zoom com folga)
    y_lo = y_min - 0.25 * y_range
    y_hi = y_max + 0.25 * y_range
    if y_range < 1e-6:
        y_lo = y_min * 0.9
        y_hi = y_max * 1.1

    # Curva do modelo
    if log:
        ax.semilogy(res["x"], y_model, color="#0D4A70", lw=2.5,
                     label="Modelo QUAL-UFMG", zorder=4)
    else:
        ax.plot(res["x"], y_model, color="#0D4A70", lw=2.5,
                 label="Modelo QUAL-UFMG", zorder=4)

    # Dados de campo
    xs_c, ys_c, ns_c = [], [], []
    for nome, d in PONTOS_CAMPO.items():
        if key in d:
            xs_c.append(d["x"])
            ys_c.append(d[key])
            ns_c.append(nome)
    if xs_c:
        ax.scatter(xs_c, ys_c, s=80, marker="D", color="#D35400",
                    edgecolor="black", linewidth=1.4, zorder=10,
                    label="Medicao de campo")

    # Padrao CONAMA
    if padrao >= y_lo and padrao <= y_hi:
        ax.axhline(padrao, color="#8B2635", lw=2, ls="--",
                   label="Padrao CONAMA C2", zorder=3)
    else:
        # Padrao fora do campo visual - anotacao no canto
        if padrao > y_hi:
            direcao = "acima"
            y_pos = 0.97
            va = "top"
        else:
            direcao = "abaixo"
            y_pos = 0.03
            va = "bottom"
        ax.text(0.98, y_pos,
                "Padrao C2 = " + str(padrao) + " (" + direcao + ")",
                transform=ax.transAxes, ha="right", va=va,
                fontsize=8.5, color="#8B2635", fontweight="bold",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFEBEE",
                          ec="#8B2635", lw=1))

    # Valores nos extremos
    ax.text(0.02, 0.97,
            "x=0: " + str(round(y_model[0], 3)),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, family="monospace", color="#0D4A70",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#0D4A70", lw=0.6, alpha=0.9))
    ax.text(0.98, 0.03,
            "x=2km: " + str(round(y_model[-1], 3)),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, family="monospace", color="#0D4A70",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#0D4A70", lw=0.6, alpha=0.9))

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Distancia a partir de P0 (m)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(loc="best", fontsize=7.5)

    # Titulo do painel (letra + nome curto)
    ax.set_title(titulo, loc="left", fontsize=10.5,
                 fontweight="bold", color="#222", pad=6)

# ---------------------------------------------------------------------
# TITULO GERAL
# ---------------------------------------------------------------------
fig.text(0.5, 0.96,
         "Anexo A2 - Simulacao Estendida do QUAL-UFMG ate 2.000 m a Jusante",
         ha="center", fontsize=13.5, fontweight="bold")
fig.text(0.5, 0.925,
         "Q_rio = 0,745 m3/s (Q90)  |  V = " + str(round(V_Q90, 5)) +
         " m/s  |  K2 = " + str(round(K2_Q90, 4)) + " d-1  |  S = " +
         str(S_DECL) + " m/m",
         ha="center", fontsize=9.5, color="#444", style="italic")

# Rodape
fig.text(0.5, 0.02,
         "Residencial Imbituba - URB/69838  |  CREA-SC 222578-3  |  Out/2026",
         ha="center", fontsize=8, color="#666", style="italic")

# Salvar
saida_png = OUT / "fig_A2_painel_resumo.png"
saida_pdf = OUT / "fig_A2_painel_resumo.pdf"
fig.savefig(str(saida_png), dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.3)
fig.savefig(str(saida_pdf), facecolor="white",
            bbox_inches="tight", pad_inches=0.3)
plt.close(fig)

print("")
print("[OK] Painel salvo: " + str(saida_png))
print("[OK] Painel salvo: " + str(saida_pdf))
print("[SUCESSO] Painel resumo do Anexo A2 gerado!")