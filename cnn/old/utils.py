import shutil
import os

def create_or_clear_directory(directory_path):
    # Comprobar si el directorio existe
    if os.path.exists(directory_path):
        # Borrar el contenido del directorio
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error al borrar {file_path}: {e}")
    else:
        # Crear el directorio
        try:
            os.makedirs(directory_path)
            print(f"Directorio {directory_path} creado.")
        except Exception as e:
            print(f"No se pudo crear el directorio {directory_path}: {e}")

def create_directory(directory_path):
    if os.path.exists(directory_path):
        pass
    else:
        # Crear el directorio
        try:
            os.makedirs(directory_path)
            print(f"Directorio {directory_path} creado.")
        except Exception as e:
            print(f"No se pudo crear el directorio {directory_path}: {e}")


def remove_directory_with_contents(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory {directory_path} and its contents removed successfully.")
    except Exception as e:
        print(f"Failed to remove directory {directory_path}: {e}")