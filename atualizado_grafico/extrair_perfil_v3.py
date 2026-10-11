# -*- coding: utf-8 -*-
# Extrai perfil do rio - 2 km DIRECAO CORRETA (para jusante)
# Versao 3 - corrige direcao + usa minimo local (leito)

from qgis.core import QgsProject, QgsPointXY
import numpy as np
from pathlib import Path
import csv

NOME_RIO = "rio aracatuba"
NOME_MDE = "mde_reprojetado"
PASTA = Path(r"C:\Users\murie\Downloads\orcamento cooltive garopaba\qualufmg-aracatuba-imbituba\atualizado_grafico")

P01_E = 726290.4763318696059287
P01_N = 6888447.66134350001811981

DISTANCIA = 2000.0
PASSO = 25.0
JANELA = 30.0   # metros para buscar cota minima (3x3)

print("=" * 60)
print("EXTRACAO DE PERFIL - v3 (direcao corrigida)")
print("=" * 60)

# Rio
rio = QgsProject.instance().mapLayersByName(NOME_RIO)[0]
geom = list(rio.getFeatures())[0].geometry()
if geom.isMultipart():
    linhas = geom.asMultiPolyline()
    linha = max(linhas, key=lambda l: sum(
        np.hypot(l[i+1].x()-l[i].x(), l[i+1].y()-l[i].y())
        for i in range(len(l)-1)))
else:
    linha = geom.asPolyline()
pts = np.array([(p.x(), p.y()) for p in linha])
print("[OK] " + str(len(pts)) + " vertices")

# Localizar P01
d = np.sqrt((pts[:, 0] - P01_E)**2 + (pts[:, 1] - P01_N)**2)
idx_p01 = int(np.argmin(d))
print("[OK] P01 no vertice " + str(idx_p01))

# ===================================================================
# DETERMINAR DIRECAO PELA COTA (rio flui para BAIXO)
# ===================================================================
mde = QgsProject.instance().mapLayersByName(NOME_MDE)[0]
prov = mde.dataProvider()

# Olhar 100 m a frente e 100 m atras para ver qual lado desce
def cota_no_ponto(x, y):
    val, ok = prov.sample(QgsPointXY(x, y), 1)
    return float(val) if ok else 0.0

# Buscar vizinhos
i_a = max(0, idx_p01 - 10)
i_b = min(len(pts)-1, idx_p01 + 10)

cota_atras = cota_no_ponto(pts[i_a, 0], pts[i_a, 1])
cota_frente = cota_no_ponto(pts[i_b, 0], pts[i_b, 1])
cota_p01 = cota_no_ponto(P01_E, P01_N)

print("[INFO] Cota P01: " + str(round(cota_p01, 2)) + " m")
print("[INFO] Cota 'atras' (idx-" + str(idx_p01 - i_a) + "): " + str(round(cota_atras, 2)) + " m")
print("[INFO] Cota 'frente' (idx+" + str(i_b - idx_p01) + "): " + str(round(cota_frente, 2)) + " m")

# Jusante = lado com cota MENOR
if cota_frente < cota_atras:
    direction = +1
    print("[OK] Direcao: +1 (frente desce = jusante)")
else:
    direction = -1
    print("[OK] Direcao: -1 (atras desce = jusante)")

# ===================================================================
# Extrair trecho
# ===================================================================
idx = idx_p01
total = 0.0
trecho = [pts[idx].copy()]
while total < DISTANCIA:
    i_next = idx + direction
    if i_next < 0 or i_next >= len(pts):
        print("[AVISO] Fim do rio")
        break
    seg = np.linalg.norm(pts[i_next] - pts[idx])
    total += seg
    trecho.append(pts[i_next].copy())
    idx = i_next
trecho = np.array(trecho)
print("[OK] Trecho: " + str(len(trecho)) + " vertices, " + str(round(total, 1)) + " m")

# Densificar
seg_len = np.linalg.norm(np.diff(trecho, axis=0), axis=1)
dist_acum = np.concatenate([[0], np.cumsum(seg_len)])
d_novo = np.arange(0, dist_acum[-1] + 0.1, PASSO)
x_novo = np.interp(d_novo, dist_acum, trecho[:, 0])
y_novo = np.interp(d_novo, dist_acum, trecho[:, 1])
print("[OK] " + str(len(d_novo)) + " pontos densificados")

# ===================================================================
# Amostrar MDE - buscar MINIMO em 3x3 (leito do rio)
# ===================================================================
print("[INFO] Amostrando MDE com janela 3x3...")
elevs = []
for xn, yn in zip(x_novo, y_novo):
    vals = []
    for dx in [-JANELA, 0, JANELA]:
        for dy in [-JANELA, 0, JANELA]:
            v, ok = prov.sample(QgsPointXY(xn+dx, yn+dy), 1)
            if ok:
                vals.append(float(v))
    elevs.append(min(vals) if vals else 0.0)
elevs = np.array(elevs)

print("[OK] Cota: min=" + str(round(elevs.min(), 2)) +
      " max=" + str(round(elevs.max(), 2)) +
      " desnivel=" + str(round(elevs[0] - elevs[-1], 2)) + " m")

# Salvar CSV
csv_path = PASTA / "perfil_rio_2km_v3.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['distance_m', 'elevation_m'])
    for dn, el in zip(d_novo, elevs):
        w.writerow([str(round(dn, 2)), str(round(el, 3))])

print("")
print("[OK] CSV: " + str(csv_path))
print("")
print("Primeiras 5 linhas:")
with open(csv_path, 'r') as f:
    for l in f.readlines()[:6]:
        print("   " + l.strip())
print("Ultimas 3 linhas:")
with open(csv_path, 'r') as f:
    for l in f.readlines()[-3:]:
        print("   " + l.strip())
print("")
print("[SUCESSO]")