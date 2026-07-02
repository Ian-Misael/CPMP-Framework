import os
import argparse
from dotenv import load_dotenv
from fabric import Connection
from pathlib import Path
import tarfile

# Cargar variables de entorno
load_dotenv()

USER = os.getenv("user")
HOST = os.getenv("host")

# --- CONFIGURACIÓN LOCAL ---

DESTINO_LOCAL = Path(__file__).resolve().parent
DIRECTORIO_REMOTO = f"/work/{USER}/CPMP"

# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Recupera archivos o carpetas del servidor remoto.")
    parser.add_argument("objetivos_remotos", nargs="+", help="Rutas completas de archivos o carpetas en el servidor")
    args = parser.parse_args()

    print(f"Conectando a {USER}@{HOST}...")

    try:
        with Connection(host=HOST, user=USER) as c:
            for objetivo in args.objetivos_remotos:
                ruta_remota = f"{DIRECTORIO_REMOTO}/{objetivo}".replace("//", "/")
                nombre_base = os.path.basename(objetivo.rstrip('/'))
                
                # Nombre del comprimido y ruta local final
                archivo_comprimido = f"{nombre_base}.tar.gz"
                ruta_remota_comprimida = f"/tmp/{archivo_comprimido}" 
                ruta_local_comprimida = os.path.join(DESTINO_LOCAL, archivo_comprimido)
                ruta_local_final = os.path.join(DESTINO_LOCAL, nombre_base)

                print(f"Preparando {ruta_remota}...")
                
                # 2. Comprimir en servidor
                c.run(f"tar -czf {ruta_remota_comprimida} -C {os.path.dirname(ruta_remota)} {nombre_base}")

                try:
                    print(f"Descargando: {archivo_comprimido}...")
                    c.get(remote=ruta_remota_comprimida, local=ruta_local_comprimida)
                    
                    # 3. Descomprimir localmente
                    print(f"Descomprimiendo en: {ruta_local_final}")
                    with tarfile.open(ruta_local_comprimida, "r:gz") as tar:
                        tar.extractall(path=DESTINO_LOCAL)
                    
                    # 4. Limpieza (borramos el .tar.gz local y el del servidor)
                    os.remove(ruta_local_comprimida)
                    c.run(f"rm {ruta_remota_comprimida}")
                    
                    print(f"✅ Éxito: {nombre_base} lista para usar.")
                    
                except Exception as e:
                    print(f"❌ Error en el proceso: {e}")

    except Exception as e:
        print(f"\n[ERROR DE CONEXIÓN]: {e}")

if __name__ == "__main__":
    main()