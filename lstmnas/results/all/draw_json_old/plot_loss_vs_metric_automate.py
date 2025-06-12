import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import os

# --- Configuración general ---
archivo_json = "../nas_log_20250525_153457.json"
directorio_salida = "figuras"
tamano_bloque = 4
ventana_movil = 3
descartar_iniciales_creciente = 20
descartar_iniciales_fija = 4

matplotlib.rcParams['font.style'] = 'italic'
sns.set_palette("gray")
os.makedirs(directorio_salida, exist_ok=True)

# --- Funciones auxiliares ---

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

def media_movil(valores, modo='fija', ventana=20, descartar_iniciales=0):
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

per_epoch = data["statistics"]["per_epoch"]
losses = []
accuracies = []
rewards = []
normalized_rewards = []

for epoch in per_epoch:
    current = epoch["stats"]["current_epoch"]
    controller = current["controller"]
    losses.append(controller["final_loss"])
    accuracies.extend(current["accuracies"]["values"])
    rewards.extend(current["raw_rewards"]["values"])
    normalized_rewards.extend(current["normalized_rewards"]["values"])

# --- Configuraciones a iterar ---
metricas_dict = {
    'accuracy': accuracies,
    'reward': rewards,
    'normalized_rewards': normalized_rewards
}

metodos_agregacion = ['mean', 'max', 'min', 'median']
modos_moving = ['creciente', 'fija']

# --- Iterar combinaciones ---
for nombre_metrica, valores in metricas_dict.items():
    for metodo in metodos_agregacion:

        # Sin media móvil
        metricas_bloques, std_bloques, var_bloques = calcular_bloques_estadisticos(valores, tamano_bloque, metodo)
        metrica_final = metricas_bloques
        std_final = std_bloques
        var_final = var_bloques

        longitud_minima = min(len(losses), len(metrica_final), len(std_final), len(var_final))
        x = list(range(longitud_minima))

        # Guardar sin moving mean
        nombre_archivo = f"{directorio_salida}/loss_vs_{nombre_metrica}_{metodo}.png"
        nombre_disp = f"{directorio_salida}/loss_vs_{nombre_metrica}_{metodo}_dispersion.png"

        # --- Gráfica principal ---
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Iterations", fontsize=12)
        ax1.set_ylabel("Loss", color='black', fontsize=12)
        l1, = ax1.plot(x, losses[:longitud_minima], label="Loss", color='gray', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='black')
        ax2 = ax1.twinx()
        ax2.set_ylabel(f"{nombre_metrica}", color='black', fontsize=12)
        l2, = ax2.plot(x, metrica_final[:longitud_minima], label=nombre_metrica, color='black', linestyle='--', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='black')
        fig.legend([l1, l2], ["Loss", nombre_metrica], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)
        fig.tight_layout()
        plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        plt.savefig(nombre_archivo, bbox_inches='tight')
        plt.close()

        # --- Gráfica de dispersión ---
        fig, ax1 = plt.subplots(figsize=(10, 5))
        l1, = ax1.plot(std_final[:longitud_minima], color="dimgray", linewidth=2, label="Standard Deviation")
        ax1.set_ylabel("Standard Deviation", fontsize=12, color="black")
        ax1.tick_params(axis='y', labelcolor="black")
        ax2 = ax1.twinx()
        l2, = ax2.plot(var_final[:longitud_minima], color="lightgray", linewidth=2, label="Variance")
        ax2.set_ylabel("Variance", fontsize=12, color="black")
        ax2.tick_params(axis='y', labelcolor="black")
        ax1.set_xlabel("Architecture Index", fontsize=12)
        fig.legend([l1, l2], ["Standard Deviation", "Variance"], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
        plt.tight_layout()
        plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        plt.savefig(nombre_disp, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[✓] Guardadas: {nombre_archivo} / {nombre_disp}")

        # Con media móvil
        for modo in modos_moving:
            metricas_bloques, std_bloques, var_bloques = calcular_bloques_estadisticos(valores, tamano_bloque, metodo)

            if modo == 'creciente':
                metrica_final = media_movil(metricas_bloques, modo=modo, ventana=ventana_movil, descartar_iniciales=descartar_iniciales_creciente)
                std_final = media_movil(std_bloques, modo=modo, ventana=ventana_movil, descartar_iniciales=descartar_iniciales_fija)
                var_final = media_movil(var_bloques, modo=modo, ventana=ventana_movil, descartar_iniciales=descartar_iniciales_fija)
            else:
                metrica_final = media_movil(metricas_bloques, modo=modo, ventana=ventana_movil, descartar_iniciales=descartar_iniciales_fija)
                std_final = media_movil(std_bloques, modo=modo, ventana=ventana_movil, descartar_iniciales=descartar_iniciales_fija)
                var_final = media_movil(var_bloques, modo=modo, ventana=ventana_movil, descartar_iniciales=descartar_iniciales_fija)

            longitud_minima = min(len(losses), len(metrica_final), len(std_final), len(var_final))
            x = list(range(longitud_minima))

            nombre_archivo = f"{directorio_salida}/loss_vs_{nombre_metrica}_{metodo}_moving_mean_{modo}.png"
            nombre_disp = f"{directorio_salida}/loss_vs_{nombre_metrica}_{metodo}_dispersion_moving_mean_{modo}.png"

            # --- Gráfica principal ---
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel("Iterations", fontsize=12)
            ax1.set_ylabel("Loss", color='black', fontsize=12)
            l1, = ax1.plot(x, losses[:longitud_minima], label="Loss", color='gray', linewidth=2)
            ax1.tick_params(axis='y', labelcolor='black')
            ax2 = ax1.twinx()
            ax2.set_ylabel(f"{nombre_metrica}", color='black', fontsize=12)
            l2, = ax2.plot(x, metrica_final[:longitud_minima], label=nombre_metrica, color='black', linestyle='--', linewidth=2)
            ax2.tick_params(axis='y', labelcolor='black')
            fig.legend([l1, l2], ["Loss", nombre_metrica], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)
            fig.tight_layout()
            plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
            plt.savefig(nombre_archivo, bbox_inches='tight')
            plt.close()

            # --- Gráfica de dispersión ---
            fig, ax1 = plt.subplots(figsize=(10, 5))
            l1, = ax1.plot(std_final[:longitud_minima], color="dimgray", linewidth=2, label="Standard Deviation")
            ax1.set_ylabel("Standard Deviation", fontsize=12, color="black")
            ax1.tick_params(axis='y', labelcolor="black")
            ax2 = ax1.twinx()
            l2, = ax2.plot(var_final[:longitud_minima], color="lightgray", linewidth=2, label="Variance")
            ax2.set_ylabel("Variance", fontsize=12, color="black")
            ax2.tick_params(axis='y', labelcolor="black")
            ax1.set_xlabel("Architecture Index", fontsize=12)
            fig.legend([l1, l2], ["Standard Deviation", "Variance"], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
            plt.tight_layout()
            plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
            plt.savefig(nombre_disp, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[✓] Guardadas: {nombre_archivo} / {nombre_disp}")
