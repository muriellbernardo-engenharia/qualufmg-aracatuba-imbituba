# -*- coding: utf-8 -*-
import csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt

SCRIPT_DIR = Path(__file__).resolve().parent
ARQ = SCRIPT_DIR / "perfil_rio_2km_v3.csv"

with open(ARQ, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    dados = [(float(r[0]), float(r[1])) for r in reader]

d = np.array([x[0] for x in dados])
z = np.array([x[1] for x in dados])

# Filtro mediana 5 (125 m)
z_smooth = medfilt(z, kernel_size=5)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Painel 1 - bruto
ax1.plot(d, z, 'o-', color='#1E88E5', lw=1.5, markersize=4,
         label="Perfil bruto (v3)")
ax1.plot(d, z_smooth, '-', color='#E53935', lw=2.5,
         label="Filtro mediana 5")
ax1.set_xlabel("Distancia a partir de P01 (m)")
ax1.set_ylabel("Cota (m)")
ax1.set_title("Perfil Extraido - v3")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Painel 2 - declividade local
dz = -np.gradient(z_smooth, d) * 1000  # mm/m
ax2.plot(d, dz, '-', color='#2E7D32', lw=2)
ax2.axhline(np.median(dz), color='#8B2635', lw=1.5, ls='--',
            label="S media = " + str(round(np.median(dz), 3)) + " mm/m")
ax2.set_xlabel("Distancia a partir de P01 (m)")
ax2.set_ylabel("Declividade S (mm/m)")
ax2.set_title("Declividade Local")
ax2.legend()
ax2.grid(True, alpha=0.3)

saida = SCRIPT_DIR / "diagnostico_v3.png"
fig.savefig(str(saida), dpi=150, bbox_inches='tight')
print("[OK] Salvo: " + str(saida))
print("[INFO] Cota: min=" + str(round(z.min(), 2)) +
      " max=" + str(round(z.max(), 2)))
print("[INFO] Desnivel: " + str(round(z[0] - z[-1], 2)) + " m")
print("[INFO] Declividade media: " + str(round(np.median(dz), 3)) + " mm/m")