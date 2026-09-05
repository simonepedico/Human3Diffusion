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
        # Se la cartella checkpoints contiene la pipeline esportata per intero
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

    # Prepare dataset and dataloader
    all_files = [os.path.join(args.test_imgs, f) for f in os.listdir(args.test_imgs) if f.endswith('.png')]
    
    rgb_files = [f for f in all_files if not ("_depth" in f or "_normal" in f)]
    rgb_files = sorted(rgb_files)
    
    if not rgb_files:
        raise FileNotFoundError(f"Non ho trovato nessuna immagine RGB principale in: {args.test_imgs}")
        
    print(f"[INFO] File RGB rilevati per l'inferenza: {rgb_files}")

    # Inizializzazione dataset
    dataset = joint_diffusion_inference_dataset(opt, rgb_path_list=rgb_files, white_bg=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    os.makedirs(args.output, exist_ok=True)

    # Inference loop
    for i, batch in enumerate(dataloader):
        current_rgb_path = rgb_files[i]
        
        subject_id = os.path.splitext(os.path.basename(current_rgb_path))[0]
        subject_save_folder = os.path.join(args.output, subject_id)
        
        if os.path.exists(os.path.join(subject_save_folder, 'gs.ply')):
            print(f"[INFO] {subject_id} già elaborato, salto.")
            continue

        WEIGHT_DTYPE = torch.float32
        os.makedirs(subject_save_folder, exist_ok=True)

        rgb_path = current_rgb_path
        base, ext = os.path.splitext(rgb_path)
        depth_path = f"{base}_depth{ext}"
        normal_path = f"{base}_normal{ext}"

        if os.path.exists(depth_path) and os.path.exists(normal_path):
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
            raise FileNotFoundError(f"Impossibile trovare le mappe richieste per {rgb_path}: {depth_path} o {normal_path}")

        gaussians = joint_2d_3d_diffusion(batch, device, pipe, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)

        # Salvataggio risultati
        save_generation_results(subject_save_folder, batch, device, gaussians, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)

if __name__ == "__main__":
    main()
