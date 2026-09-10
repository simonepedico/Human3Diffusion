import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run joint 2D&3D diffusion inference (Mesh Generation).')
    parser.add_argument('--output', type=str, default='output', help='Directory containing inference output files.')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--mesh_quality', type=str, default='high', choices=['high', 'low', 'None'], help='Quality of the generated mesh.')
    args = parser.parse_args()

    if args.mesh_quality == 'None':
        print("[INFO] mesh_quality è impostato su 'None'. Nessuna mesh verrà generata.")
        return

    from core.tsdf_mesh import generate_tsdf_mesh

    pifuhd_ckpt = os.path.join(args.checkpoints, 'pifuhd.pt')
    if not os.path.exists(pifuhd_ckpt):
        raise FileNotFoundError(f"Impossibile trovare il checkpoint per la mesh in: {pifuhd_ckpt}")

    print(f"[INFO] Scansione della cartella '{args.output}' per la ricerca di modelli 'gs.ply'...\n")

    processed_count = 0

    # Scansiona tutte le sottocartelle dentro la cartella output
    for root, dirs, files in os.walk(args.output):
        if 'gs.ply' in files:
            # Calcola il nome/percorso relativo per la stampa dei log
            rel_folder = os.path.relpath(root, args.output)
            mesh_path = os.path.join(root, 'tsdf-rgbd.ply')

            # Se la mesh esiste già, saltiamo
            if os.path.exists(mesh_path):
                print(f"[INFO] Mesh già presente per '{rel_folder}', salto.")
                continue

            print(f"--> Generazione mesh ({args.mesh_quality} quality) per: '{rel_folder}'")
            
            # Genera la mesh usando la cartella di output corrente che contiene gs.ply
            generate_tsdf_mesh(root, pifuhd_ckpt, quality=args.mesh_quality)
            processed_count += 1

    print(f"\n[OK] Generazione mesh completata! Elaborati {processed_count} soggetti in '{args.output}'.")

if __name__ == '__main__':
    main()
