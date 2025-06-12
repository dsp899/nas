import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import os

# --- Configuración ---
metodo_agregacion = 'mean'  # Opciones: 'mean', 'max', 'min', 'median'
metrica = 'accuracy'        # Opciones: 'accuracy', 'reward', 'normalized_reward'
moving_mean = True
modo = 'creciente'          # Opciones: 'creciente', 'fija'
archivo_json = "../nas_log_20250525_153457.json"
directorio_salida = "figuras"
tamano_bloque = 4

# Estilo
matplotlib.rcParams['font.style'] = 'italic'
sns.set_palette("gray")
os.makedirs(directorio_salida, exist_ok=True)

# Salidas
def crear_nombre_salida(suffix):
    base = f"{directorio_salida}/loss_vs_{metrica}"
    if moving_mean:
        return f"{base}_{suffix}_moving_mean_{modo}.png"
    return f"{base}_{suffix}.png"

salida_imagen = crear_nombre_salida(metodo_agregacion)
salida_imagen_dispersion = crear_nombre_salida(f"{metodo_agregacion}_dispersion")
salida_imagen_comparativa = crear_nombre_salida("comparativa_agregaciones")

# --- Funciones ---
def calcular_bloques_estadisticos(valores, tamano_bloque, metodo='mean'):
    bloques = [valores[i:i + tamano_bloque] for i in range(0, len(valores), tamano_bloque)]
    agregados, stds, vars_ = [], [], []
    for bloque in bloques:
        if not bloque:
            continue
        if metodo == 'mean':
            agregados.append(np.mean(bloque))
        elif metodo == 'max':
            agregados.append(np.max(bloque))
        elif metodo == 'min':
            agregados.append(np.min(bloque))
        elif metodo == 'median':
            agregados.append(np.median(bloque))
        else:
            raise ValueError(f"Método de agregación no válido: {metodo}")
        stds.append(np.std(bloque))
        vars_.append(np.var(bloque))
    return agregados, stds, vars_

def media_movil(valores, modo='creciente', ventana=20, descartar_iniciales=4):
    resultado = []
    for i in range(len(valores)):
        if i < descartar_iniciales:
            continue
        if modo == 'fija':
            inicio = max(0, i - ventana + 1)
            ventana_valores = valores[inicio:i + 1]
        elif modo == 'creciente':
            ventana_valores = valores[:i + 1]
        else:
            raise ValueError("El parámetro 'modo' debe ser 'fija' o 'creciente'")
        resultado.append(np.mean(ventana_valores))
    return resultado

# --- Cargar datos JSON ---
with open(archivo_json, "r") as f:
    data = json.load(f)

# --- Extraer métricas ---
per_epoch = data["statistics"]["per_epoch"]
losses, accuracies, rewards, normalized_rewards = [], [], [], []
for epoch in per_epoch:
    current = epoch["stats"]["current_epoch"]
    controller = current["controller"]
    losses.append(controller["final_loss"])
    accuracies.extend(current["accuracies"]["values"])
    rewards.extend(current["raw_rewards"]["values"])
    normalized_rewards.extend(current["normalized_rewards"]["values"])

# Selección de métrica
if metrica == 'accuracy':
    metricas = accuracies
elif metrica == 'reward':
    metricas = rewards
elif metrica == 'normalized_reward':
    metricas = normalized_rewards

# --- Agregación principal seleccionada ---
metricas_bloques, std_bloques, var_bloques = calcular_bloques_estadisticos(metricas, tamano_bloque, metodo_agregacion)
if moving_mean:
    metrica_final = media_movil(metricas_bloques, modo, ventana=10, descartar_iniciales=20)
    std_final = media_movil(std_bloques, modo, ventana=10, descartar_iniciales=20)
    var_final = media_movil(var_bloques, modo, ventana=10, descartar_iniciales=20)
else:
    metrica_final = metricas_bloques
    std_final = std_bloques
    var_final = var_bloques

# --- Recorte de longitud ---
longitud_minima = min(len(losses), len(metrica_final), len(std_final), len(var_final))
losses = losses[:longitud_minima]
metrica_final = metrica_final[:longitud_minima]
std_final = std_final[:longitud_minima]
var_final = var_final[:longitud_minima]
x = list(range(longitud_minima))

# --- Figura principal ---
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xlabel("Iterations", fontsize=12)
ax1.set_ylabel("Loss", color='black', fontsize=12)
line1, = ax1.plot(x, losses, label="Loss", color='gray', linewidth=2)
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.set_ylabel(f"{metrica.capitalize()}", color='black', fontsize=12)
line2, = ax2.plot(x, metrica_final, label=metodo_agregacion.title(), color='black', linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor='black')

lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)

fig.tight_layout()
plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig(salida_imagen, bbox_inches='tight')
plt.savefig(f"{salida_imagen.split('.')[0]}.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
plt.savefig(f"{salida_imagen.split('.')[0]}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f"[✔] Gráfica guardada en: {salida_imagen}")

# --- Figura de dispersión (idéntico estilo a figura principal) ---
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xlabel("Iterations", fontsize=12)
ax1.set_ylabel("Standard Deviation", color='black', fontsize=12)
line1, = ax1.plot(x, std_final, label="Standard Deviation", color='gray', linewidth=2)
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.set_ylabel("Variance", color='black', fontsize=12)
line2, = ax2.plot(x, var_final, label="Variance", color='black', linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor='black')

lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=10)

fig.tight_layout()
plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig(salida_imagen_dispersion, dpi=300, bbox_inches='tight')
plt.savefig(f"{salida_imagen_dispersion.split('.')[0]}.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
plt.savefig(f"{salida_imagen_dispersion.split('.')[0]}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f"[✓] Gráfica guardada en: {salida_imagen_dispersion}")


# --- Figura comparativa de métodos ---
metodos = ['mean', 'max', 'min', 'median']
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xlabel("Iterations", fontsize=12)
ax1.set_ylabel("Loss", color='black')
line_loss, = ax1.plot(losses[:len(x)], color='gray', linewidth=2, label="Loss")
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.set_ylabel(metrica.capitalize(), color='black')
lineas = [line_loss]
etiquetas = ["Loss"]
colores = ['black', 'dimgray', 'darkgray', 'silver']

for metodo, color in zip(metodos, colores):
    metrica_m, _, _ = calcular_bloques_estadisticos(metricas, tamano_bloque, metodo)
    if moving_mean:
        metrica_m = media_movil(metrica_m, modo, ventana=3, descartar_iniciales=20)
    metrica_m = metrica_m[:len(x)]
    line = ax2.plot(metrica_m, label=metodo.title(), linestyle='--', linewidth=2, color=color)
    lineas.extend(line)
    etiquetas.append(metodo.title())

fig.legend(lineas, etiquetas, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
plt.tight_layout()
plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig(salida_imagen_comparativa, dpi=300, bbox_inches='tight')
plt.savefig(f"{salida_imagen_comparativa.split('.')[0]}.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
plt.savefig(f"{salida_imagen_comparativa.split('.')[0]}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f"[★] Gráfica comparativa guardada en: {salida_imagen_comparativa}")
