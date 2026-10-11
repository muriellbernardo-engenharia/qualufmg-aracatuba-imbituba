# -*- coding: utf-8 -*-
# Limpeza final do perfil v3
import csv
import numpy as np
from pathlib import Path
from scipy.signal import medfilt
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
ARQ_IN = SCRIPT_DIR / "perfil_rio_2km_v3.csv"
ARQ_OUT = SCRIPT_DIR / "perfil_final.csv"

with open(ARQ_IN, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    dados = [(float(r[0]), float(r[1])) for r in reader]

d = np.array([x[0] for x in dados])
z = np.array([x[1] for x in dados])
print("[INFO] " + str(len(d)) + " pontos lidos")
print("[INFO] Faixa bruta: " + str(round(z.min(), 2)) + " a " + str(round(z.max(), 2)))

# 1) Mediana 7 (175 m) - remove picos isolados
z_med = medfilt(z, kernel_size=7)
print("[INFO] Apos mediana: " + str(round(z_med.min(), 2)) + " a " + str(round(z_med.max(), 2)))

# 2) Rejeitar pontos abaixo de 1.5 m (anomalia do MDE)
mask = z_med > 1.5
print("[INFO] Pontos rejeitados (< 1.5 m): " + str(int((~mask).sum())))

# 3) Interpolar suave
f = interp1d(d[mask], z_med[mask], kind='linear', fill_value='extrapolate')
z_final = f(d)

# 4) Suavizar mais (mediana 3)
z_final = medfilt(z_final, kernel_size=3)

print("[INFO] Faixa final: " + str(round(z_final.min(), 2)) + " a " + str(round(z_final.max(), 2)))
print("[INFO] Desnivel final: " + str(round(z_final[0] - z_final[-1], 2)) + " m")

# Salvar
with open(ARQ_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['distance_m', 'elevation_m'])
    for di, zi in zip(d, z_final):
        w.writerow([str(round(di, 2)), str(round(zi, 3))])

print("")
print("[OK] Salvo: " + str(ARQ_OUT))