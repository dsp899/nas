#!/bin/bash
#"pmi50" "bm50" "hoi50" "hhi50" "sports50" "all50"
#"pmi" "bm" "hoi" "hhi" "sports" "all"

gpu="1"

# Definir el array con los nombres de los datasets
datasets=("pmi" "bm" "hoi" "hhi" "sports")

# Definir nombre del fichero de log
logfile="predict_log_$(date +'%Y%m%d_%H%M%S').log"

# Guardar tiempo de inicio total
start_total=$(date +%s)

# Crear o limpiar el fichero de log
echo "Inicio de ejecución: $(date)" > "$logfile"
echo "-------------------------------------" >> "$logfile"

# Iterar sobre cada elemento del array
for dataset in "${datasets[@]}"
do
    echo "Procesando dataset: $dataset" | tee -a "$logfile"
    
    # Ejecutar y medir el primer comando
    echo "Ejecutando VGG16..." | tee -a "$logfile"
    start=$(date +%s)
    python3 run_cnn.py --operation predict --cnn vgg16 --data "$dataset" --frames 15 --size 299 --gpu "$gpu" | tee -a "$logfile"
    end=$(date +%s)
    runtime=$((end-start))
    echo "VGG16 terminado en $runtime segundos" | tee -a "$logfile"
    
    # Ejecutar y medir el segundo comando
    echo "Ejecutando InceptionV3..." | tee -a "$logfile"
    start=$(date +%s)
    python3 run_cnn.py --operation predict --cnn inceptionV3 --data "$dataset" --frames 15 --size 299 --gpu "$gpu" | tee -a "$logfile"
    end=$(date +%s)
    runtime=$((end-start))
    echo "InceptionV3 terminado en $runtime segundos" | tee -a "$logfile"

    # Ejecutar y medir el tercer comando
    echo "Ejecutando ResNet50..." | tee -a "$logfile"
    start=$(date +%s)
    python3 run_cnn.py --operation predict --cnn resnet50 --data "$dataset" --frames 15 --size 299 --gpu "$gpu" | tee -a "$logfile"
    end=$(date +%s)
    runtime=$((end-start))
    echo "ResNet50 terminado en $runtime segundos" | tee -a "$logfile"
    
    echo "Finalizado procesamiento de: $dataset" | tee -a "$logfile"
    echo "-------------------------------------" | tee -a "$logfile"
done

# Guardar tiempo final total
end_total=$(date +%s)
runtime_total=$((end_total-start_total))

echo "Todos los datasets han sido procesados" | tee -a "$logfile"
echo "Tiempo total de ejecución: $runtime_total segundos" | tee -a "$logfile"
echo "Fin de ejecución: $(date)" | tee -a "$logfile"
