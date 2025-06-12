import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import os
# --- Configuración ---
metodo_agregacion = 'median'  # Opciones: 'mean', 'max', 'min', 'median'


# archivo_json = "../nas_log_20250605_010533.json" #myLSTMNAS antiguo con discounted_reward y Ev 1
archivo_json = "../nas_log_20250605_093327.json" #myLSTMNAS antiguo sin discounted_reward y Ev 1
directorio_salida = f"figuras"
salida_imagen = f"{directorio_salida}/loss_vs_accuracy_{metodo_agregacion}.png"
tamano_bloque = 4

matplotlib.rcParams['font.style'] = 'italic'
sns.set_palette("gray")  # Escala de grises

# --- Crear carpeta de salida ---
os.makedirs(directorio_salida, exist_ok=True)




def agrupar_por_bloques(valores, tamano_bloque, metodo='mean'):
    bloques = [valores[i:i + tamano_bloque] for i in range(0, len(valores), tamano_bloque)]
    resultado = []

    for bloque in bloques:
        if not bloque:
            continue
        if metodo == 'mean':
            resultado.append(np.mean(bloque))
        elif metodo == 'max':
            resultado.append(np.max(bloque))
        elif metodo == 'min':
            resultado.append(np.min(bloque))
        elif metodo == 'median':
            resultado.append(np.median(bloque))
        else:
            raise ValueError(f"Método de agregación no válido: {metodo}")
    
    return resultado



# --- Función para estadísticas móviles crecientes ---
def estadisticas_moviles_crecientes(valores):
    medias = []
    for i in range(1, len(valores) + 1):
        ventana = valores[:i]
        medias.append(np.mean(ventana))
    return medias

# --- Cargar datos JSON ---
with open(archivo_json, "r") as f:
    data = json.load(f)

# --- Extraer final_loss y accuracies ---
per_epoch = data["statistics"]["per_epoch"]
losses = []
accuracies = []

for epoch in per_epoch:
    current = epoch["stats"]["current_epoch"]
    controller = current["controller"]
    losses.append(controller["final_loss"])
    accuracies.extend(current["accuracies"]["values"])  # Lista de 4 valores por epoch

# --- Procesar accuracy ---
accuracy_bloques = agrupar_por_bloques(accuracies, tamano_bloque, metodo_agregacion)
accuracy_moving_mean = estadisticas_moviles_crecientes(accuracy_bloques)

# --- Ajustar longitud para que coincidan los ejes ---
longitud_minima = min(len(losses), len(accuracy_bloques))
losses = losses[:longitud_minima]
accuracy_bloques = accuracy_bloques[:longitud_minima]
x = list(range(longitud_minima))

# --- Graficar ---
fig, ax1 = plt.subplots(figsize=(10, 5))

# Eje izquierdo - Final Loss
ax1.set_xlabel("Iterations", fontsize=12)
ax1.set_ylabel("Loss", color='black', fontsize=12)
line1, = ax1.plot(x, losses, label="Loss", color='gray', linewidth=2)
ax1.tick_params(axis='y', labelcolor='black')

# Eje derecho - Accuracy (moving average)
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy (Moving Average)", color='black', fontsize=12)
line2, = ax2.plot(x, accuracy_bloques, label="Accuracy", color='black', linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor='black')

# Leyenda centrada debajo del eje x
lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)




# Estilo
fig.tight_layout()
plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig(salida_imagen, bbox_inches='tight')
plt.close()

print(f"[✔] Gráfica guardada en: {salida_imagen}")
