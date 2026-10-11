# =====================================================================
# MODELO QUAL-UFMG - TRECHO ESTENDIDO DE 2 KM
# Rio Aracatuba - ETE Residencial Imbituba
# REFERENCIA: M1 (consistente com Capitulo 8)
# Graficos com zoom nos eixos Y
# =====================================================================

import warnings
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.titlepad": 12,
    "axes.edgecolor": "#2B2B2B", "axes.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC", "legend.fontsize": 9,
    "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "output_final_2km"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  MODELO QUAL-UFMG - TRECHO ESTENDIDO 2 KM")
print("  Referencia hidraulica: M1 (igual ao Capitulo 8)")
print("=" * 70)

# ---------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------
Q_RIO = 0.745
Q_EF = 0.00234
S_DECL = 0.000215

V_M1 = 0.10494
Q_M1 = 2.973

V_Q90 = V_M1 * np.sqrt(Q_RIO / Q_M1)
K2_A = 0.800
K2_Q90 = K2_A * np.sqrt(V_Q90 / V_M1)
t_percurso = 2000.0 / V_Q90 / 86400.0

print("[PARAMETROS]")
print("  V(Q90) = " + str(round(V_Q90, 5)) + " m/s")
print("  K2(Q90) = " + str(round(K2_Q90, 4)) + " d-1")
print("")

# ---------------------------------------------------------------
# DADOS
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# CONDICAO INICIAL
# ---------------------------------------------------------------
Q_total = Q_RIO + Q_EF
C0 = {k: (Q_RIO * RIO_M1[k] + Q_EF * EFLUENTE[k]) / Q_total for k in RIO_M1}

# ---------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------
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

print("[SIMULANDO ate 2000 m...]")
sol = solve_ivp(ode, (0, X_MAX), y0, method="RK45",
                rtol=1e-6, atol=1e-9, dense_output=True, max_step=10)
xg = np.linspace(0, X_MAX, N_PONTOS)
Y = sol.sol(xg)
L, OD, Norg, NNH3, NNO2, NNO3, Porg, Pinorg, Coli = Y
res = dict(x=xg, OD=OD, DBO5=L/F_DBO, N_NH3=NNH3, N_NO2=NNO2,
           N_NO3=NNO3, P_total=Porg + Pinorg, Coli=Coli)
print("[OK] Simulacao concluida.")
print("")

# ---------------------------------------------------------------
# FIGURAS COM ZOOM
# ---------------------------------------------------------------
# (nome, ylabel, padrao, cor_modelo)
# ylim sera calculado DINAMICAMENTE para dar zoom
configs = [
    ("OD", "Oxigenio Dissolvido (mg/L)", PADROES["OD"],
     "Anexo A2 - Perfil de OD (2 km)", False),
    ("DBO5", "DBO5 (mg/L)", PADROES["DBO5"],
     "Anexo A2 - Perfil de DBO5 (2 km)", False),
    ("N_NH3", "N-amoniacal (mg/L)", PADROES["N_NH3"],
     "Anexo A2 - Perfil de N-amoniacal (2 km)", False),
    ("N_NO3", "N-nitrato (mg/L)", PADROES["N_NO3"],
     "Anexo A2 - Perfil de N-nitrato (2 km)", False),
    ("P_total", "Fosforo Total (mg/L)", PADROES["P_total"],
     "Anexo A2 - Perfil de Fosforo Total (2 km)", False),
    ("Coli", "Coliformes (NMP/100 mL)", PADROES["Coli"],
     "Anexo A2 - Perfil de Coliformes (2 km)", True),
]

print("[GERANDO FIGURAS - com zoom nos eixos Y]")

for key, ylabel, padrao_val, titulo, log in configs:
    try:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)

        # Dados do modelo
        y_model = res[key]
        y_min_model = float(np.min(y_model))
        y_max_model = float(np.max(y_model))
        y_range_model = y_max_model - y_min_model

        # Limites ZOOM (com folga de 20% em cima e embaixo)
        y_lo = y_min_model - 0.25 * y_range_model
        y_hi = y_max_model + 0.25 * y_range_model
        if y_range_model < 1e-6:
            y_lo = y_min_model * 0.9
            y_hi = y_max_model * 1.1

        # Plot do modelo
        if log:
            ax.semilogy(res["x"], y_model, color="#0D4A70", lw=3,
                         label="Modelo QUAL-UFMG", zorder=3)
        else:
            ax.plot(res["x"], y_model, color="#0D4A70", lw=3,
                     label="Modelo QUAL-UFMG", zorder=3)

        # Dados de campo (se tiver)
        xs_c, ys_c, ns_c = [], [], []
        pontos_campo = {
            "J1": dict(x=109.17, OD=6.10, DBO5=21.42, N_NH3=0.05,
                       N_NO3=0.40, P_total=0.07, Coli=44.0),
            "J2": dict(x=310.51, OD=6.10, DBO5=22.44, N_NH3=0.05,
                       N_NO3=0.40, P_total=0.15, Coli=71.0),
        }
        for nome, d in pontos_campo.items():
            if key in d:
                xs_c.append(d["x"])
                ys_c.append(d[key])
                ns_c.append(nome)

        if xs_c:
            ax.scatter(xs_c, ys_c, s=140, marker="D", color="#D35400",
                        edgecolor="black", linewidth=1.8, zorder=10,
                        label="Medicoes de campo")
            for xi, yi, ni in zip(xs_c, ys_c, ns_c):
                ax.annotate(ni, xy=(xi, yi), xytext=(0, 14),
                            textcoords="offset points", ha="center",
                            fontsize=10, fontweight="bold",
                            color="#D35400")

        # Linha do padrao CONAMA (com anotacao no canto)
        if padrao_val >= y_lo and padrao_val <= y_hi:
            ax.axhline(padrao_val, color="#8B2635", lw=2.5, ls="--",
                       label="Padrao CONAMA C2", zorder=2)
        else:
            # Padrao fora do campo visual — anotar no canto
            if padrao_val > y_hi:
                direcao = "acima"
                y_pos = 0.98
            else:
                direcao = "abaixo"
                y_pos = 0.02
            ax.text(0.98, y_pos,
                    "Padrao C2 = " + str(padrao_val) + " mg/L (" + direcao + " do campo)",
                    transform=ax.transAxes, ha="right", va="top" if y_pos > 0.5 else "bottom",
                    fontsize=9.5, color="#8B2635", fontweight="bold",
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE",
                              ec="#8B2635", lw=1.2))

        ax.set_xlim(0, X_MAX)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Distancia a partir de P0 (m)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
        ax.legend(loc="best", fontsize=9.5)

        # Anotacoes: valor inicial e final
        ax.text(0.02, 0.98,
                "x=0: " + str(round(y_model[0], 4)),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, family="monospace", color="#0D4A70",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD",
                          ec="#0D4A70", lw=0.8))
        ax.text(0.98, 0.02,
                "x=2000: " + str(round(y_model[-1], 4)),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, family="monospace", color="#0D4A70",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD",
                          ec="#0D4A70", lw=0.8))

        # Titulo
        fig.text(0.5, 0.94, titulo, ha="center",
                 fontsize=13, fontweight="bold")
        fig.text(0.5, 0.88,
                 "Q_rio = " + str(Q_RIO) + " m3/s  |  V = " +
                 str(round(V_Q90, 5)) + " m/s  |  K2 = " +
                 str(round(K2_Q90, 4)) + " d-1",
                 ha="center", fontsize=9, color="#444", style="italic")
        fig.text(0.5, 0.02,
                 "Residencial Imbituba - URB/69838  |  CREA-SC 222578-3  |  Out/2026",
                 ha="center", fontsize=8, color="#666", style="italic")

        nome = "fig_A2_" + key
        fig.savefig(str(OUT / (nome + ".png")), dpi=300,
                    facecolor="white", bbox_inches="tight", pad_inches=0.25)
        fig.savefig(str(OUT / (nome + ".pdf")),
                    facecolor="white", bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print("  [OK] " + nome + ".png (Y: " +
              str(round(y_lo, 3)) + " a " + str(round(y_hi, 3)) + ")")
    except Exception as e:
        print("  [ERRO] " + key + ": " + str(e))
        traceback.print_exc()

# ---------------------------------------------------------------
# FIGURA HIDRAULICA
# ---------------------------------------------------------------
print("")
print("[GERANDO FIGURA HIDRAULICA]")

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.subplots_adjust(hspace=0.20, top=0.90, bottom=0.09,
                    left=0.09, right=0.97)

# Painel 1: perfil
z_perfil = 4.18 - S_DECL * xg
axes[0].plot(xg, z_perfil, color="#4A3B2C", lw=2.5)
axes[0].fill_between(xg, z_perfil.min() - 0.3, z_perfil,
                      color="#8B7355", alpha=0.2)
axes[0].set_ylabel("Cota (m)", fontsize=10.5)
axes[0].set_title("Perfil Longitudinal - S = 0,000215 m/m",
                   fontsize=11, fontweight="bold", pad=8)
axes[0].grid(True, linestyle=":", alpha=0.35)
axes[0].text(0.98, 0.10,
             "Desnivel total: " + str(round(S_DECL * 2000, 3)) + " m",
             transform=axes[0].transAxes, ha="right", va="bottom",
             fontsize=9.5, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec="#999", lw=1))

# Painel 2: declividade
axes[1].plot(xg, np.full_like(xg, S_DECL * 1000),
              color="#1E88E5", lw=2.5, label="S(x) = 0,215 mm/m")
axes[1].set_ylabel("Declividade (mm/m)", fontsize=10.5)
axes[1].set_ylim(0, 0.5)
axes[1].legend(fontsize=9, loc="upper right")
axes[1].grid(True, linestyle=":", alpha=0.35)

# Painel 3: velocidade
axes[2].plot(xg, np.full_like(xg, V_Q90),
              color="#2E7D32", lw=2.5,
              label="V(Q90) = " + str(round(V_Q90, 5)) + " m/s")
axes[2].scatter([0], [V_M1], s=120, marker="D", color="#D35400",
                 edgecolor="black", linewidth=1.5, zorder=10,
                 label="Medicao em M1 = " + str(V_M1) + " m/s")
axes[2].set_ylabel("Velocidade (m/s)", fontsize=10.5)
axes[2].set_xlabel("Distancia a partir de P0 (m)", fontsize=11)
axes[2].set_ylim(0, 0.13)
axes[2].legend(fontsize=9, loc="upper right")
axes[2].grid(True, linestyle=":", alpha=0.35)

fig.text(0.5, 0.96,
         "Caracterizacao Hidraulica do Trecho de 2 km",
         ha="center", fontsize=13, fontweight="bold")
fig.text(0.5, 0.02,
         "Residencial Imbituba - URB/69838  |  CREA-SC 222578-3  |  Out/2026",
         ha="center", fontsize=8, color="#666", style="italic")

fig.savefig(str(OUT / "fig_A2_hidraulica.png"), dpi=300,
            facecolor="white", bbox_inches="tight", pad_inches=0.25)
fig.savefig(str(OUT / "fig_A2_hidraulica.pdf"),
            facecolor="white", bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("  [OK] fig_A2_hidraulica.png + .pdf")

# ---------------------------------------------------------------
# RESUMO
# ---------------------------------------------------------------
print("")
print("=" * 70)
print("  RESULTADOS EM PONTOS-CHAVE")
print("=" * 70)
print("  x (m)   OD      DBO5    N-NH3   N-NO3   P-tot   Coli")
print("-" * 70)
for xv in [0, 100, 300, 500, 1000, 1500, 2000]:
    idx = int(np.argmin(np.abs(res["x"] - xv)))
    print("  " + str(xv).ljust(7)
          + str(round(res["OD"][idx], 3)).ljust(8)
          + str(round(res["DBO5"][idx], 3)).ljust(8)
          + str(round(res["N_NH3"][idx], 4)).ljust(8)
          + str(round(res["N_NO3"][idx], 4)).ljust(8)
          + str(round(res["P_total"][idx], 4)).ljust(8)
          + str(round(res["Coli"][idx], 1)))

print("=" * 70)
print("[SUCESSO] Figuras com zoom geradas!")