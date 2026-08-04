import os
import argparse
import torch

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run joint 2D&3D diffusion inference.')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output images.')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Directory containing test images.')
    # PEDICO: percorso adapter layer 
    parser.add_argument('--adapter_ckpt', type=str, default=None, help='Path to the trained adapter layer checkpoint.')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import necessary modules
    from core.dataloader_inference import joint_diffusion_inference_dataset
    from torch.utils.data.dataloader import DataLoader
    from core.diffusion3d_pipeline import get_2ddiffusion_model, get_3ddiffusion_model, joint_2d_3d_diffusion, save_generation_results
    from core.options import Options
    
    # Load 2D diffusion model
    dict2ddiffusion_path = os.path.join(args.checkpoints, 'model.safetensors')
    
    # PEDICO: inizializzo a pesi random se l'adapter non viene caricato
    adapter_path = args.adapter_ckpt
    if adapter_path is None:
        potential_adapter = os.path.join(args.checkpoints, 'adapter_layer.pt')
        if os.path.exists(potential_adapter):
            adapter_path = potential_adapter
            print(f"[INFO] Trovato adapter layer automatico in: {adapter_path}")
        else:
            print("[ATTENZIONE] Nessun adapter_layer.pt trovato nella cartella dei checkpoint. Verrà usato l'adapter di default non addestrato!")


    pipe = get_2ddiffusion_model(dict2ddiffusion_path, device, adapter_ckpt_path=adapter_path)

    # PEDICO
    pipe = pipe.to(torch.float32)
    pipe.enable_vae_slicing() # provvisorio o sforo la ram di colab

    # Load 3D diffusion model
    opt = Options()
    dict3ddiffusion_path = os.path.join(args.checkpoints, 'model_1.safetensors')
    diffusion3dgs_model = get_3ddiffusion_model(dict3ddiffusion_path, device, opt)

    # Prepare dataset and dataloader
    # cerco files nella cartella
    all_files = [os.path.join(args.test_imgs, f) for f in os.listdir(args.test_imgs) if f.endswith('.png')]
    
    # tengo oslo il render (come file principale)
    rgb_files = [f for f in all_files if not ("_depth" in f or "_normal" in f)]
    rgb_files = sorted(rgb_files) # Ordine alfabetico per sicurezza
    
    if not rgb_files:
        raise FileNotFoundError(f"Non ho trovato nessuna immagine RGB principale in: {args.test_imgs}")
        
    print(f"[INFO] File RGB rilevato per l'inferenza: {rgb_files}")

    # inizializzazione dataset
    dataset = joint_diffusion_inference_dataset(opt, rgb_path_list=rgb_files, white_bg=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run inference loop
    for i, batch in enumerate(dataloader):
        #prendo percorso dalla lista files
        current_rgb_path = rgb_files[i]
        
        # genero nuova cartellla in cartella output
        subject_id = os.path.splitext(os.path.basename(current_rgb_path))[0]
        subject_save_folder = os.path.join(args.output, subject_id)
        
        if os.path.exists(subject_save_folder + '/gs.ply'):
            continue

        WEIGHT_DTYPE = torch.float32
        os.makedirs(subject_save_folder, exist_ok=True)

        # cerco normal e depth
        rgb_path = current_rgb_path
        base, ext = os.path.splitext(rgb_path)
        depth_path = f"{base}_depth{ext}"
        normal_path = f"{base}_normal{ext}"

        if os.path.exists(depth_path) and os.path.exists(normal_path):
            from PIL import Image
            import torchvision.transforms.functional as TF
            
            # carico immagini
            # se avessi la depth in piu cnaali, la converto in monocanale
            depth_img = Image.open(depth_path).convert("L")
            normal_img = Image.open(normal_path).convert("RGB")
            
            # ridimiensioni immagini 256x256
            depth_img = depth_img.resize((256, 256), Image.BILINEAR)
            normal_img = normal_img.resize((256, 256), Image.BILINEAR)
            
            # trasformo in tensor e  normalizzo nel range [-1.0, 1.0]
            depth_tensor = TF.to_tensor(depth_img) * 2.0 - 1.0
            normal_tensor = TF.to_tensor(normal_img) * 2.0 - 1.0
            
            # aggiungo dimensioni per simulare output dataloader
            # Risultato finale desiderato dopo lo squeeze: (B, C, H, W) -> es. (1, 3, 256, 256)
            batch['context_depth'] = depth_tensor.unsqueeze(0).unsqueeze(1)   # Diventa (1, 1, 1, H, W)
            batch['context_normal'] = normal_tensor.unsqueeze(0).unsqueeze(1) # Diventa (1, 1, 3, H, W)
            
            print("[INFO] Mappe Depth e Normal caricate e formattate correttamente.")
        else:
            raise FileNotFoundError(f"Impossibile trovare le mappe richieste: {depth_path} o {normal_path}")
       

        gaussians = joint_2d_3d_diffusion(batch, device, pipe, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)

        # Save results
        save_generation_results(subject_save_folder, batch, device, gaussians, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)

if __name__ == "__main__":
    main()
