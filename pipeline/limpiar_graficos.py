"""Remove every file in graficos/ that graficar_fronteras_red.py does not produce.

Dry run by default: it only lists what would be deleted. Pass --aplicar to delete.
Scope: only files inside programa_ann/graficos. Empty subfolders left behind are removed too.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graficar_fronteras_red import DATASETS, SALIDA, nombre_salida  # noqa: E402

CONSERVAR = {nombre_salida(ds) for ds in DATASETS}


def sobrantes():
    if not SALIDA.exists():
        return []
    return sorted(p for p in SALIDA.rglob("*") if p.is_file() and not (p.parent == SALIDA and p.name in CONSERVAR))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aplicar", action="store_true", help="delete the files instead of listing them")
    args = parser.parse_args()

    lista = sobrantes()
    for p in lista:
        print(("borrar  " if args.aplicar else "se borraría  ") + str(p.relative_to(SALIDA)))
    print(f"{len(lista)} archivo(s) sobrante(s); se conservan: {sorted(CONSERVAR)}")

    if not args.aplicar:
        print("Modo prueba. Ejecuta con --aplicar para borrar.")
        return 0

    for p in lista:
        p.unlink()
    if SALIDA.exists():
        for carpeta in sorted((p for p in SALIDA.rglob("*") if p.is_dir()), key=lambda d: len(d.parts), reverse=True):
            if not any(carpeta.iterdir()):
                carpeta.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
