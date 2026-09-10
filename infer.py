import os
import argparse
import torch

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run joint 2D&3D diffusion inference.')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output images.')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Directory containing test images.')
    # Percorso facoltativo al VAE a 7 canali (file .safetensors o cartella vae/)
    parser.add_argument('--vae_ckpt', type=str, default=None, help='Path to trained 7-channel VAE checkpoint or directory.')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import necessary modules
    from core.dataloader_inference import joint_diffusion_inference_dataset
    from torch.utils.data.dataloader import DataLoader
    from core.diffusion3d_pipeline import get_2ddiffusion_model, get_3ddiffusion_model, joint_2d_3d_diffusion, save_generation_results
    from core.options import Options
    
    # Path del modello 2D (UNet)
    dict2ddiffusion_path = os.path.join(args.checkpoints, 'model.safetensors')
    if not os.path.exists(dict2ddiffusion_path):
        if os.path.exists(os.path.join(args.checkpoints, 'unet')):
            dict2ddiffusion_path = args.checkpoints

    # Rilevamento automatico del VAE a 7 canali se non specificato
    vae_path = args.vae_ckpt
    if vae_path is None:
        potential_vae_dir = os.path.join(args.checkpoints, 'vae')
        potential_vae_file = os.path.join(args.checkpoints, 'vae_model.safetensors')
        if os.path.exists(potential_vae_dir):
            vae_path = potential_vae_dir
            print(f"[INFO] Trovata cartella VAE automatica in: {vae_path}")
        elif os.path.exists(potential_vae_file):
            vae_path = potential_vae_file
            print(f"[INFO] Trovato file VAE automatico in: {vae_path}")

    # Caricamento pipeline 2D con VAE a 7 canali
    pipe = get_2ddiffusion_model(dict2ddiffusion_path, device, vae_ckpt_path=vae_path)

    pipe = pipe.to(torch.float32)
    pipe.enable_vae_slicing()

    # Load 3D diffusion model
    opt = Options()
    dict3ddiffusion_path = os.path.join(args.checkpoints, 'model_1.safetensors')
    diffusion3dgs_model = get_3ddiffusion_model(dict3ddiffusion_path, device, opt)

    # Scansione ricorsiva di tutte le sottocartelle in test_imgs
    rgb_files = []
    for root, dirs, files in os.walk(args.test_imgs):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                f_lower = file.lower()
                # Esclude mappe depth e normal per isolare il render RGB principale
                if not ("_depth" in f_lower or "_normal" in f_lower or "depth_processed" in f_lower or "normal_processed" in f_lower):
                    rgb_files.append(os.path.join(root, file))
    
    rgb_files = sorted(rgb_files)

    if not rgb_files:
        raise FileNotFoundError(f"Non ho trovato nessuna immagine RGB principale in nessuna sottocartella di: {args.test_imgs}")
        
    print(f"[INFO] Trovati {len(rgb_files)} file RGB per l'inferenza distribuiti nelle sottocartelle.")

    # Inizializzazione dataset
    dataset = joint_diffusion_inference_dataset(opt, rgb_path_list=rgb_files, white_bg=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    os.makedirs(args.output, exist_ok=True)

    # Inference loop
    for i, batch in enumerate(dataloader):
        current_rgb_path = rgb_files[i]
        
        # Mantiene la struttura di sottocartelle anche nella destinazione di output
        rel_path = os.path.relpath(current_rgb_path, args.test_imgs)
        rel_dir = os.path.dirname(rel_path)
        base_filename = os.path.splitext(os.path.basename(current_rgb_path))[0]

        if rel_dir and rel_dir != ".":
            subject_id = os.path.join(rel_dir, base_filename)
        else:
            subject_id = base_filename

        subject_save_folder = os.path.join(args.output, subject_id)
        
        if os.path.exists(os.path.join(subject_save_folder, 'gs.ply')):
            print(f"[INFO] {subject_id} già elaborato, salto.")
            continue

        WEIGHT_DTYPE = torch.float32
        os.makedirs(subject_save_folder, exist_ok=True)

        rgb_path = current_rgb_path
        base, ext = os.path.splitext(rgb_path)
        folder = os.path.dirname(rgb_path)

        # Ricerca flessibile delle mappe depth e normal (supporta sia nome_depth che depth_processed)
        cand_depths = [
            f"{base}_depth{ext}",
            os.path.join(folder, f"depth_processed{ext}"),
            os.path.join(folder, "depth_processed.png")
        ]
        cand_normals = [
            f"{base}_normal{ext}",
            os.path.join(folder, f"normal_processed{ext}"),
            os.path.join(folder, "normal_processed.png")
        ]

        depth_path = next((p for p in cand_depths if os.path.exists(p)), None)
        normal_path = next((p for p in cand_normals if os.path.exists(p)), None)

        if depth_path and normal_path:
            from PIL import Image
            import torchvision.transforms.functional as TF
            
            depth_img = Image.open(depth_path).convert("L")
            normal_img = Image.open(normal_path).convert("RGB")
            
            depth_img = depth_img.resize((256, 256), Image.BILINEAR)
            normal_img = normal_img.resize((256, 256), Image.BILINEAR)
            
            depth_tensor = TF.to_tensor(depth_img) * 2.0 - 1.0
            normal_tensor = TF.to_tensor(normal_img) * 2.0 - 1.0
            
            batch['context_depth'] = depth_tensor.unsqueeze(0).unsqueeze(1)
            batch['context_normal'] = normal_tensor.unsqueeze(0).unsqueeze(1)
            
            print(f"[INFO] Mappe Depth e Normal caricate per: {subject_id}")
        else:
            raise FileNotFoundError(f"Impossibile trovare le mappe richieste per {rgb_path} dentro '{folder}'")

        gaussians = joint_2d_3d_diffusion(batch, device, pipe, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)

        # Salvataggio risultati
        save_generation_results(subject_save_folder, batch, device, gaussians, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)

if __name__ == "__main__":
    main()
