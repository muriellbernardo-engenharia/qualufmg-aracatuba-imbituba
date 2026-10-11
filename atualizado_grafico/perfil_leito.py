# -*- coding: utf-8 -*-
"""Perfil longitudinal do leito do Rio Araçatuba"""

import matplotlib.pyplot as plt
import numpy as np

# Dados
pontos = ["M2", "M1", "P0", "J1", "J2"]
distancia = [0.00, 100.73, 214.23, 311.23, 510.51]
cota_terreno = [4.22, 4.20, 4.18, 4.16, 4.11]
h_centro = [2.52, 2.65, 2.15, 2.55, 2.22]
cota_leito = [t - h for t, h in zip(cota_terreno, h_centro)]

# Interpolação fina
x_fino = np.linspace(0, 510.51, 300)
y_terreno = np.interp(x_fino, distancia, cota_terreno)
y_leito = np.interp(x_fino, distancia, cota_leito)

# Gráfico
fig, ax = plt.subplots(figsize=(12, 5))

# Área do terreno
ax.fill_between(x_fino, y_terreno, y_leito, color="#CFE4EE", alpha=0.8,
                label="Coluna d'água")
# Linha do terreno
ax.plot(x_fino, y_terreno, color="#4A3B2C", lw=2, label="Superfície da água")
# Linha do leito
ax.plot(x_fino, y_leito, color="#1F5C7A", lw=2.5, label="Leito do rio")

# Pontos
for nome, d, t, l in zip(pontos, distancia, cota_terreno, cota_leito):
    ax.scatter(d, t, s=100, marker="o", facecolor="white",
               edgecolor="#4A3B2C", linewidth=2, zorder=5)
    ax.scatter(d, l, s=100, marker="v", facecolor="#1F5C7A",
               edgecolor="black", linewidth=1, zorder=5)
    ax.annotate(nome, xy=(d, t), xytext=(0, 12),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold")

# Anotações
ax.set_xlabel("Distância a partir de M2 (m)", fontsize=11)
ax.set_ylabel("Cota (m)", fontsize=11)
ax.set_title("Perfil Longitudinal do Leito — Rio Araçatuba",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xlim(-20, 530)
ax.set_ylim(1.2, 4.6)

plt.tight_layout()
plt.savefig("fig_perfil_leito.png", dpi=300)
plt.show()
print("Gráfico salvo: fig_perfil_leito.png")