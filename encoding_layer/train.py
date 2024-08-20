import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from model import VAE
from dataloader import MapDataset
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from tqdm import tqdm
from collections import defaultdict

def regularize_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--norm_type", "-nt",
                    help="Type of normalisation layer used, BatchNorm (bn) or GroupNorm (gn)", type=str, default="bn")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=16)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=256)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)

parser.add_argument("--num_res_blocks", '-nrb',
                    help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=16)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))
# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--deep_model", '-dm', action='store_true',
                    help="Deep Model adds an additional res-identity block to each down/up sampling stage")

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    writer = SummaryWriter()

    dataset = MapDataset(data_dir=args.dataset_root, device=device)
    test_split = 0.9
    n_train_examples = int(len(dataset) * test_split)
    n_test_examples = len(dataset) - n_train_examples
    train_set, test_set = random_split(dataset, [n_train_examples, n_test_examples], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    dataiter = iter(test_loader)
    test_pose, test_egos, test_allos = next(dataiter)
    vae_net = VAE(
        channel_in=test_egos.shape[1],
        ch=args.ch_multi,
        blocks=args.block_widths,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        norm_type=args.norm_type,
        deep_model=args.deep_model
    ).to(device)

    # Setup optimizer
    optimizer = optim.Adam(vae_net.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    num_model_params = 0
    for param in vae_net.parameters():
        num_model_params += param.flatten().shape[0]

    print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
    fm_size = args.image_size//(2 ** len(args.block_widths))
    print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))

    # Create the save directory if it does not exist
    if not os.path.isdir(args.save_dir + "/Models"):
        os.makedirs(args.save_dir + "/Models")
    if not os.path.isdir(args.save_dir + "/Results"):
        os.makedirs(args.save_dir + "/Results")

    # Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
    # If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
    # it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
    save_file_name = args.model_name + "_" + str(args.image_size)
    if args.load_checkpoint:
        if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
            checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                                    map_location="cpu")
            print("-Checkpoint loaded!")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            vae_net.load_state_dict(checkpoint['model_state_dict'])

            if not optimizer.param_groups[0]["lr"] == args.lr:
                print("Updating lr!")
                optimizer.param_groups[0]["lr"] = args.lr

            start_epoch = checkpoint["epoch"]
            data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
        else:
            raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")
    else:
        # If checkpoint does exist raise an error to prevent accidental overwriting
        if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
            raise ValueError("Warning Checkpoint exists -> add -cp flag to use this checkpoint")
        else:
            print("Starting from scratch")
            start_epoch = 0
            # Loss and metrics logger
            data_logger = defaultdict(lambda: [])
    print("")
    
    for epoch in range(start_epoch, args.nepoch):
        vae_net.train()
        print(f"Epoch {epoch} start!")
        with tqdm(train_loader, leave=True) as t_step:
            for i, (pose, ego_map, allo_map) in enumerate(t_step):
                t_step.set_description(f"Epoch {epoch+1}")
                current_iter = i + epoch * len(train_loader)
                ego_map = ego_map.to(device)
                bs, c, h, w = ego_map.shape

                # We will train with mixed precision!
                with torch.cuda.amp.autocast():
                    recon_img, mu, log_var = vae_net(ego_map, pose)

                    kl_loss = regularize_loss(mu, log_var)
                    mse_loss = F.mse_loss(recon_img, allo_map)
                    loss = args.kl_scale * kl_loss + mse_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
                scaler.step(optimizer)
                scaler.update()

                # Log losses and other metrics for evaluation!
                data_logger["mu"].append(mu.mean().item())
                data_logger["mu_var"].append(mu.var().item())
                data_logger["log_var"].append(log_var.mean().item())
                data_logger["log_var_var"].append(log_var.var().item())

                data_logger["kl_loss"].append(kl_loss.item())
                data_logger["img_mse"].append(mse_loss.item())
                t_step.set_postfix(reg_loss=kl_loss.item(), recon_mse=mse_loss.item(), total_loss=loss.item())
                writer.add_scalar("Loss/reconstruction_loss", mse_loss, current_iter)
                writer.add_scalar("Loss/regularization_loss", kl_loss, current_iter)
                writer.add_scalar("Loss/total_loss", loss, current_iter)
                
                # Save results and a checkpoint at regular intervals
                if (current_iter + 1) % args.save_interval == 0:
                    # In eval mode the model will use mu as the encoding instead of sampling from the distribution
                    vae_net.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            # Save an example from testing and log a test loss
                            recon_img, mu, log_var = vae_net(test_egos.to(device), test_pose.to(device))
                            data_logger['test_mse_loss'].append(F.mse_loss(recon_img, test_allos.to(device)).item())

                        # Keep a copy of the previous save in case we accidentally save a model that has exploded...
                        # if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                        #     shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt", dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

                        # Save a checkpoint
                        torch.save({
                            'epoch': epoch + 1,
                            'data_logger': dict(data_logger),
                            'model_state_dict': vae_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, args.save_dir + "/Models/" + save_file_name + ".pt")

                        torch.save({
                            'epoch': epoch + 1,
                            'data_logger': dict(data_logger),
                            'model_state_dict': vae_net.encoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, args.save_dir + "/Models/" + save_file_name + "_encoder.pt")
                        
                        torch.save({
                            'epoch': epoch + 1,
                            'data_logger': dict(data_logger),
                            'model_state_dict': vae_net.decoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, args.save_dir + "/Models/" + save_file_name + "_decoder.pt")

                        # Set the model back into training mode!!
                        vae_net.train()
    writer.flush()