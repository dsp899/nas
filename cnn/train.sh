# #!/bin/bash
# #"pmi50" "bm50" "hoi50" "hhi50" "sports50" "all50"
# #"pmi" "bm" "hoi" "hhi" "sports" "all"
# # Definir el array con los nombres de los datasets
# datasets=("pmi" "bm" "hoi" "hhi" "sports")

# # Iterar sobre cada elemento del array
# for dataset in "${datasets[@]}"
# do
#     echo "Procesando dataset: $dataset"
    
#     # Ejecutar los comandos de Python para cada dataset
#     python3 run_cnn.py --operation train --cnn vgg16 --data "$dataset" --frames 15 --size 299
#     python3 run_cnn.py --operation train --cnn inceptionV3 --data "$dataset" --frames 15 --size 299
#     python3 run_cnn.py --operation train --cnn resnet50 --data "$dataset" --frames 15 --size 299
    
#     echo "Finalizado procesamiento de: $dataset"
#     echo "-------------------------------------"
# done

# echo "Todos los datasets han sido procesados"

#!/bin/bash

# Crear carpeta para logs si no existe
mkdir -p logs

# Array de datasets
datasets=("pmi" "bm" "hoi" "hhi" "sports")
# Array de modelos
modelos=("vgg16" "inceptionV3" "resnet50")

# Log global
global_log="training_log.txt"
echo "Inicio del entrenamiento: $(date)" > "$global_log"

# Iterar sobre datasets y modelos
for dataset in "${datasets[@]}"; do
    echo "Procesando dataset: $dataset" | tee -a "$global_log"
    
    for modelo in "${modelos[@]}"; do
        log_file="logs/${dataset}_${modelo}_log.txt"
        echo "Entrenando $modelo con dataset $dataset" | tee -a "$global_log"
        
        # Ejecutar el script y guardar tanto stdout como stderr
        python3 run_cnn.py --operation train --cnn "$modelo" --data "$dataset" --frames 15 --size 299 \
            2>&1 | tee "$log_file"
        
        echo "Finalizado $modelo con $dataset" | tee -a "$global_log"
        echo "-------------------------------------" | tee -a "$global_log"
    done
done

echo "Todos los datasets han sido procesados: $(date)" | tee -a "$global_log"
