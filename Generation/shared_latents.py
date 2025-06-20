import argparse, os, itertools
from models import ATMS
from torch.optim import AdamW
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    ReversePermutation,
    MaskedAffineAutoregressiveTransform,
    RandomPermutation
)
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import wandb

from custom_pipeline_phil import Generator4Embeds
from eegdatasets_leaveone import EEGDataset
from utils_phil import extract_id_from_string, set_seed
from util import wandb_logger

import gc
import torch
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def load_ATMS_loaders(args, sub, device):
    print(f"Loading EEG dataset for subject {sub} on device {device}")

    if args.insubject: # per subject
        clip_dataset = {'train': EEGDataset(args.data_path, subjects=[sub], train=True, device=device),
                        'test':  EEGDataset(args.data_path, subjects=[sub], train=False, device=device)}
    else:
        # Leave one subject out
        clip_dataset = {'train': EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=True),
                        'test':  EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=False)}

    # Loaders
    clip_loaders = {'train': DataLoader(clip_dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True),
                        'test': DataLoader(clip_dataset['test'], batch_size=1, shuffle=False, num_workers=0, drop_last=True)}    
    
    return clip_dataset, clip_loaders



class shared_latents():
    def __init__(self, args, subject):
        self.args = args
        set_seed(self.args.seed)
        self.subject = subject
        self.set_device()  # Set the device based on the provided argument
        self.dataset, self.dataloader = load_ATMS_loaders(self.args, subject, self.device) # Load EEG dataset

        self.logger = wandb_logger(args, self.subject)

        # Instantiate Flow
        self.flow = self.create_flow_model(feature_dim=1024, num_layers=5, hidden_features=2048)  # Create flow model
        # instantiate ATMS model
        self.eeg_model = ATMS(args = args)
        self.eeg_model.to(self.device)
        self.optimizer = AdamW(itertools.chain(self.eeg_model.parameters()), lr=args.lr)

        self.best_test_acc = 0.0
        self.epoch = 0 

        self.generator = None # Only load when needed


    def set_device(self):

        if self.args.device == 'cuda' and torch.cuda.is_available():
            device = 'cuda'
        elif self.args.device == 'mps' and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        self.device = torch.device(device)

        print(f"Using device: {self.device}")

    def create_flow_model(self, feature_dim=1024, num_layers=5, hidden_features=2048):
        transforms = []

        for _ in range(num_layers):
            transforms.append(RandomPermutation(features=feature_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=feature_dim,
                    hidden_features=hidden_features,
                    num_blocks=2,
                    use_residual_blocks=True,
                    random_mask=False
                )
            )

        transform = CompositeTransform(transforms)
        base_dist = StandardNormal(shape=[feature_dim])
        flow = Flow(transform, base_dist).to(self.device)

        return flow

    def contrastive_clip_loss(self, z1, z2, temperature=0.07):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.T / temperature
        labels = torch.arange(len(z1)).to(z1.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def compute_retrieval_accuracy(self, stage):
        self.eeg_model.eval()
        self.flow.eval()
        
        # Precompute image embeddings in latent space
        z_img_all = self.flow._transform(self.dataset[stage].img_features.to(self.device))[0]
        z_img_all = F.normalize(z_img_all, dim=-1)

        img_labels = self.dataset[stage].labels.to(self.device)

        total_correct = 0
        total_samples = 0

        for eeg_data, labels, *_ in self.dataloader[stage]:
            eeg_data = eeg_data.to(self.device)
            labels = labels.to(self.device)
            subject_ids = torch.full((eeg_data.size(0),), 0, dtype=torch.long).to(self.device)  # use correct subject ID logic

            with torch.no_grad():
                z_eeg = self.eeg_model(eeg_data, subject_ids)
                z_eeg = F.normalize(z_eeg, dim=-1)

                # Compute cosine similarity: [batch_size, num_images]
                logits = z_eeg @ z_img_all.T
                top_idx = logits.argmax(dim=1)
                predicted_labels = img_labels[top_idx]
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

        return total_correct / total_samples
    
    def run_epoch(
        self,
        train=True
    ):
        if train:
            self.eeg_model.train()
            self.flow.train()
            stage = 'train'
        else:
            self.eeg_model.eval()
            self.flow.eval()
            stage = 'test'

        total_clip_loss = 0
        total_flow_loss = 0
        total_cos_sim = 0
        recon_loss = 0
        img_recon_mse = 0
        num_batches = 0

        loop = tqdm(self.dataloader[stage], desc=stage.upper())

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for eeg_data, labels, _, _, _, img_features in loop:
                eeg_data = eeg_data.to(self.device)
                img_features = img_features.to(self.device)
                labels = labels.to(self.device)

                subject_ids = torch.full((eeg_data.size(0),), extract_id_from_string(self.subject), dtype=torch.long).to(self.device)

                if train:
                    self.optimizer.zero_grad()

                z_eeg = self.eeg_model(eeg_data, subject_ids)
                z_img = self.flow._transform(img_features)[0]

                # reconstruction
                

                z_eeg = F.normalize(z_eeg, dim=-1)
                z_img = F.normalize(z_img, dim=-1)
                eeg_recon = self.flow._transform.inverse(z_eeg)[0]

                loss_clip = self.contrastive_clip_loss(z_img, z_eeg)
                loss_flow = -self.flow.log_prob(img_features).mean()
                loss_recon = F.mse_loss(eeg_recon, img_features)

                if train:
                    loss = loss_clip
                    loss.backward()
                    self.optimizer.step()
                
                with torch.no_grad():
                    img_recon = self.flow._transform.inverse(z_img)[0]
                    mse_img = F.mse_loss(img_recon, img_features)
                    cos_sim = F.cosine_similarity(z_img, z_eeg).mean().item()

                total_clip_loss += loss_clip.item()
                total_flow_loss += loss_flow.item()
                recon_loss += loss_recon.item()
                img_recon_mse += mse_img.item()
                total_cos_sim += cos_sim
                num_batches += 1

                loop.set_postfix({
                    "clip": total_clip_loss / num_batches,
                    "flow": total_flow_loss / num_batches,
                    "recon": recon_loss / num_batches,
                    "cosine": total_cos_sim / num_batches,
                    "img_recon": img_recon_mse / num_batches
                })

        retrieval_top1 = self.compute_retrieval_accuracy(stage)

        return {
            "avg_clip_loss": total_clip_loss / num_batches,
            "avg_flow_loss": total_flow_loss / num_batches,
            "avg_cosine_similarity": total_cos_sim / num_batches,
            "avg_reconstruction_loss": recon_loss / num_batches,
            "avg_img_recon_mse": img_recon_mse / num_batches,
            "retrieval_top1": retrieval_top1
        }

    def train(self):
        for epoch in tqdm(range(self.epoch, args.epochs), desc="Epochs"):

            train_metrics = self.run_epoch(train=True)
            test_metrics  = self.run_epoch(train=False)

            # Logging
            epoch_results = {f"train_{k}": v for k, v in train_metrics.items()}
            epoch_results.update({f"test_{k}": v for k, v in test_metrics.items()})
            epoch_results["epoch"] = epoch + 1
            self.logger.log(epoch_results)

            # Save best
            if test_metrics["retrieval_top1"] > self.best_test_acc:
                self.best_test_acc = test_metrics["retrieval_top1"]
                self.save_model()
                print(f"New best test accuracy: {self.best_test_acc:.4f} at epoch {epoch + 1}")

            # Update resume point
            self.epoch = epoch + 1


    def save_model(self):
        torch.save({
            "epoch": self.epoch,
            "eeg_model": self.eeg_model.state_dict(),
            "flow_model": self.flow.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_test_acc": self.best_test_acc
        }, f"{args.model_dir}/{self.subject}/checkpoint.pt")

    def load_model(self):
        checkpoint = torch.load(f"{args.model_dir}/{self.subject}/checkpoint.pt", map_location=self.device)
        self.eeg_model.load_state_dict(checkpoint["eeg_model"])
        self.flow.load_state_dict(checkpoint["flow_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]
        self.best_test_acc = checkpoint["best_test_acc"]
        print(f"Model loaded from epoch {self.epoch}")

    def generate_images(self):
        if self.generator == None:
            self.generator = Generator4Embeds(num_inference_steps=4, device=self.device, force_download=True)

        for i, data in tqdm(enumerate(self.dataloader['test'])):
            eeg_data, labels, _, _, _, img_features = data
            eeg_data = eeg_data.to(self.device)
            img_features = img_features.to(self.device)

            subject_ids = torch.full((eeg_data.size(0),), extract_id_from_string(self.subject), dtype=torch.long).to(self.device)
            z_eeg = self.eeg_model(eeg_data, subject_ids)

            # Get reconstructed embeddings
            eeg_recon = self.flow._transform.inverse(z_eeg)[0]
            img_recon = self.flow._transform.inverse(self.flow._transform(img_features)[0])[0]

            try:
                # Generate images
                img_original = self.generator.generate(image_embeds=img_features, text_prompt="")
                img_reconstr = self.generator.generate(image_embeds=img_recon, text_prompt="")
                eeg_image = self.generator.generate(image_embeds=eeg_recon.unsqueeze(0), text_prompt="")

                # Ensure all are PIL
                if not isinstance(img_original, Image.Image): img_original = Image.fromarray(np.array(img_original))
                if not isinstance(img_reconstr, Image.Image): img_reconstr = Image.fromarray(np.array(img_reconstr))
                if not isinstance(eeg_image, Image.Image):     eeg_image   = Image.fromarray(np.array(eeg_image))

                # Combine in one plot
                fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                axs[0].imshow(img_original);  axs[0].set_title("Original")
                axs[1].imshow(img_reconstr); axs[1].set_title("Flow-Recon")
                axs[2].imshow(eeg_image);    axs[2].set_title("EEG→Image")
                for ax in axs: ax.axis('off')

                # Save to WandB
                wandb.log({f"Image reconstruction/{i}": wandb.Image(fig)})

                plt.close(fig)

            except Exception as e:
                print(f"Generation failed on sample {i}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')

    parser.add_argument('--data_path', type=str, default="/work3/s184984/repos/EEG_Image_decode/eeg_dataset/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--model_dir', type=str, default='./models/shared_latents', help='Directory to save output results')    
    parser.add_argument('--seed', type=int, default=42, help='Number of epochs')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cuda', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')  
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs') 
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--loss_fn', type=str, choices=['clip', 'vicreg', 'softContrastive', 'softHybridContrastive', 'None'], default='clip', help='loss function, see loss.py for more info')
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--test_repetition_method', type=str, choices=['average', 'first'], default='average', help='Method to use for testing repetition')


    # WandB args
    parser.add_argument('--project', type=str, default="EEG_image_reconstruction", help='WandB project name')
    parser.add_argument('--name', type=str, default="shared_latents", help='Experiment name')
    parser.add_argument('--entity', type=str, default="philliphoejbjerg", help='WandB entity name')
    os.environ["WANDB_API_KEY"] = "b0c5da2aac89929c85f768b56e5f260e287064ab"
    os.environ["WANDB_MODE"] = 'online'

    args = parser.parse_args()

    for subject in args.subjects:
        print(f"Training for subject: {subject}")
        
        model = shared_latents(args, subject)
        # model.load_model()
        model.train()


    