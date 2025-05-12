import torch
import argparse
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchdiffeq import odeint # For flow solving
import torch.nn as nn
from flow_matching.path import CondOTProbPath

#from flow_unet import FlowUNet1D
#from flow_mlp import ResMLPFlow
from flow_transformer import TransformerFlow

import itertools

from tqdm import tqdm

# Modules
from util import wandb_logger
from utils_phil import extract_id_from_string, set_seed
from eegdatasets_leaveone import EEGDataset

import matplotlib.pyplot as plt

from models import ATMS
from normalizer import Normalizer

import io


def load_ATMS_loaders(args, sub, device):
    # Placeholder function to load EEG dataset
    # Replace with actual data loading logic
    print(f"Loading EEG dataset for subject {sub} on device {device}")
    # Implement your data loading logic here

            # Load datasets 
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

def load_FLOW_loaders(args, sub, device, clip_loaders, eeg_model, split=None, reconstruction = False):
    def build_dataset(split_key):
        dataset = clip_loaders[split_key].dataset

        eeg_embs = torch.cat([
            eeg_model(
                ele[0].unsqueeze(0).to(device),
                torch.tensor([extract_id_from_string(sub)], dtype=torch.long).to(device)
            ) for ele in tqdm(dataset)
        ], axis=0)

        if args.diffusion_target == 'image':
            clip_embs = dataset.img_features.view(1654, 10, 1, 1024).repeat(1, 1, 4, 1).view(-1, 1024) if split_key == 'train' else dataset.img_features
        else:
            clip_embs = dataset.text_features.view(1654, 1, 1, 1024).repeat(1, 10, 4, 1).view(-1, 1024) if split_key == 'train' else dataset.text_features

        return EmbeddingDataset(
            clip_eeg_embeddings=eeg_embs,
            clip_embeddings=clip_embs,
            labels=dataset.labels,
            img_paths = dataset.img if split_key == 'test' else np.repeat(dataset.img, 4).tolist() # Repeat images per training instance
        )

    stage2_dataset = {}
    stage2_loaders = {}

    with torch.no_grad():
        
        splits = ['train', 'test'] if split is None else [split]

        for s in splits:
            stage2_dataset[s] = build_dataset(s)
            stage2_loaders[s] = DataLoader(
                stage2_dataset[s],
                batch_size= 1 if reconstruction else args.flow_batch_size,
                shuffle=(s == 'train'),
                num_workers=0
            )

    return stage2_loaders
    


# Train model function
def train_clip_aligner(sub, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, args):
    
    eeg_model.train()
    
    # init loss func
    mse_loss_fn = nn.MSELoss()

    # For grabbing correct embeddings
    all_clip_emb = {'text': text_features_all.to(device).float(), # (n_cls, d) # prev: text_features_all,
                    'image': (img_features_all[::10]).to(device).float() # prev: img_features_all 
                    # TODO: There are 10 images per class, so we take one of them - but perhaps, prediction should be okay if correct class is chosen? - the first at any rate is not enough, cause it might not be the specific one in the batch - though, this is just for logging scores anywats
                    }

    # Define a mapping for the feature types
    feature_mapping = {
        'text': 'text_features',
        'image': 'img_features'
    }

    # initialize 
    total_loss, total_size, correct, total_MSE = 0, 0, 0, 0 

    for batch_idx, (eeg_data, labels, _, text_features, _, img_features) in enumerate(dataloader):
        
        # TODO: possibly do within dataloader? 
        eeg_data = eeg_data.to(device)

        # Grabbing either img_features or text features from the locals within the function
        clip_emb = locals()[feature_mapping[args.atms_target]].to(device).float()
        labels = labels.to(device)
        
        batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
        subject_ids = torch.full((batch_size,), extract_id_from_string(sub), dtype=torch.long).to(device)

        # get outs
        clip_eeg_emb = eeg_model(eeg_data, subject_ids).float()
        logit_scale = eeg_model.logit_scale # a learnable parameter

        # MSE calculation
        total_MSE += mse_loss_fn(clip_eeg_emb, clip_emb).item()

        # loss_func: --> clip_loss()
        if args.loss_fn == 'clip':
            clip_loss = eeg_model.loss_func(clip_eeg_emb, clip_emb, logit_scale)

            mse_loss =  mse_loss_fn(clip_eeg_emb, clip_emb)
    
            loss = (10 * (args.alpha * mse_loss) + (10 * ((1 - args.alpha) * clip_loss)))
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Measure similarity between EEG embeddings and CLIP image embeddings - to get actual prediction, and thereby accuracy. NOT needed for training
            logits_img = logit_scale * clip_eeg_emb @ all_clip_emb[args.atms_target].T

        elif args.loss_fn == 'vicreg' or args.loss_fn == 'softContrastive' or args.loss_fn == 'softHybridContrastive':
            loss = eeg_model.loss_func(clip_eeg_emb, clip_emb)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Measure similarity between EEG embeddings and CLIP image embeddings - to get actual prediction, and thereby accuracy. NOT needed for training
            # Perhaps use logit_scale
            logits_img = clip_eeg_emb @ all_clip_emb[args.atms_target].T

        predicted = torch.argmax(logits_img, dim=1) # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
    
        # update loss and accuracy
        total_loss += loss.item()
        total_size += batch_size
        correct += (predicted == labels).sum().item()

        del eeg_data, clip_eeg_emb, clip_emb

    # Calculate loss and accuracy
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total_size
    average_MSE = total_MSE / (batch_idx+1)

    return average_loss, accuracy, average_MSE


def validate_clip_aligner(sub, eeg_model, test_loader, device, test_dataset, args, epoch):
    
    eeg_model.eval()

    # init loss func
    mse_loss_fn = nn.MSELoss()

    # For grabbing correct embeddings
    all_clip_emb = {'text': test_dataset.text_features.to(device).float(), 
                    'image': test_dataset.img_features.to(device).float()
                    }
    
    # initialize 
    all_labels = set(range(all_clip_emb['text'].size(0)))
    total_loss, total_size, correct, total_MSE = 0, 0, 0, 0,

    # Define a mapping for the feature types
    feature_mapping = {
        'text': 'text_features',
        'image': 'img_features'
    }    

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _, text_features, _, img_features) in enumerate(test_loader):

            eeg_data = eeg_data.to(device)

            clip_emb = locals()[feature_mapping[args.atms_target]].to(device).float()

            labels = labels.to(device)
            
            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_ids = torch.full((batch_size,), extract_id_from_string(sub), dtype=torch.long).to(device)
            
            # get outs
            clip_eeg_emb = eeg_model(eeg_data, subject_ids).float()
            logit_scale = eeg_model.logit_scale

            # MSE calculation
            total_MSE += mse_loss_fn(clip_eeg_emb, clip_emb).item()
                
            # loss_func: --> clip_loss()
            if args.loss_fn == 'clip':
                # calculate loss
                clip_loss = eeg_model.loss_func(clip_eeg_emb, clip_emb, logit_scale)
                mse_loss =  mse_loss_fn(clip_eeg_emb, clip_emb)
                loss = (10 * (args.alpha * mse_loss) + (10 * ((1 - args.alpha) * clip_loss)))
                    
                total_loss += loss.item()
                
            elif args.loss_fn == 'vicreg' or args.loss_fn == 'softContrastive' or args.loss_fn == 'softHybridContrastive':
                loss = eeg_model.loss_func(clip_eeg_emb, clip_emb)

                total_loss += loss.item()
        

            logits_img = logit_scale * clip_eeg_emb @ all_clip_emb[args.atms_target].T if args.loss_fn == 'clip' else clip_eeg_emb @ all_clip_emb[args.atms_target].T
            predicted = torch.argmax(logits_img, dim=1) # (n_batch, ) ∈ {0, 1, ..., n_cls-1}

            # update loss and accuracy
            total_size += batch_size
            correct += (predicted == labels).sum().item()            


            # del eeg_data, eeg_features, img_features

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total_size
    average_MSE = total_MSE / (batch_idx+1)
    
    # Append results for this epoch
    epoch_results = {
        "epoch": epoch + 1,
        "test_loss": average_loss,
        "test_accuracy": accuracy,
        "test_MSE": average_MSE
    }

    return epoch_results

# TODO: REMEMBER WE'RE NOW DOING BEST VALIDATION NOT BEST RETRIEVAL!!!
def load_best_model(eeg_model, sub, device, args):

    # Getting output from the best model
    PATH = f"{args.model_dir}/{args.name}/{sub}" if args.insubject else f"{args.model_dir}/{args.name}/across"
    eeg_model.load_state_dict(torch.load(f"{PATH}/{args.pth_name}.pth", weights_only=False, map_location=torch.device(device)))
    # Freezing the original embedder
    if args.freeze_ATMS:
        # Freeze the parameters of the original model
        for param in eeg_model.parameters():
            param.requires_grad = False
        eeg_model.eval()

    return eeg_model


# -- Flow matching --
def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


from torch import Tensor

class EmbeddingDataset(Dataset):

    def __init__(self, clip_eeg_embeddings, clip_embeddings, labels, img_paths):
        self.clip_eeg_embeddings = clip_eeg_embeddings
        self.clip_embeddings = clip_embeddings
        self.labels = labels,
        self.img_paths = img_paths

    def __len__(self):
        return len(self.clip_eeg_embeddings)

    def __getitem__(self, idx):
        return {
            "clip_eeg_embeddings": self.clip_eeg_embeddings[idx],
            "clip_embeddings": self.clip_embeddings[idx],
            "labels": self.labels[0][idx],
            "img_paths": self.img_paths[idx]
        } 
    
def process_batch(batch, is_train, epoch=None):
    x_0 = batch['clip_eeg_embeddings'].to(device)
    x_1 = batch['clip_embeddings'].to(device)
    labels = batch.get('labels', None)
    if labels is not None:
        labels = labels.to(device)

    # Normalize
    if args.use_normalization and eeg_normalizer is not None and clip_normalizer is not None:
        x_0 = eeg_normalizer.normalize(x_0)
        x_1 = clip_normalizer.normalize(x_1)

    if is_train:
        batch_size = x_0.size(0)
        t = skewed_timestep_sample(batch_size, device=device) if args.skewed_timesteps else torch.rand(batch_size, device=device)
        sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        pred_dx = flow(sample.x_t, t)
        loss = F.mse_loss(pred_dx, sample.dx_t)

        # Optional: cosine similarity loss on endpoint
        if args.cosine_loss_weight > 0 or args.mse_loss_weight > 0:

            x_1_pred = solver.sample(
                x_init=x_0,
                step_size=None,
                time_grid=torch.linspace(0, 1, steps=21).to(device),
                method="rk4",
                return_intermediates=False,
                enable_grad = True
            )

            # TODO: Denormalize? - also try transformer with normalize
            if args.cosine_loss_weight > 0:
                cosine_sim = F.cosine_similarity(x_1_pred, x_1, dim=-1).mean()
                cosine_loss = 1 - cosine_sim
                loss += args.cosine_loss_weight * cosine_loss

            if args.mse_loss_weight > 0:
                mse_loss = F.mse_loss(x_1_pred, x_1)
                loss += args.mse_loss_weight * mse_loss

        return loss, {}

    else:
        # Validation mode
        t = torch.rand(x_0.size(0), device=device)
        sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        pred_dx = flow(sample.x_t, t)
        val_loss = F.mse_loss(pred_dx, sample.dx_t)

        x_1_pred = solver.sample(
            x_init=x_0,
            step_size=None,
            time_grid=torch.linspace(0, 1, steps=21).to(device),
            method="rk4",
            return_intermediates=False,
        )

        if args.cosine_loss_weight > 0 or args.mse_loss_weight > 0:
            if args.cosine_loss_weight > 0:
                cosine_sim = F.cosine_similarity(x_1_pred, x_1, dim=-1).mean()
                cosine_loss = 1 - cosine_sim
                val_loss += args.cosine_loss_weight * cosine_loss

            if args.mse_loss_weight > 0:
                mse_loss = F.mse_loss(x_1_pred, x_1)
                val_loss += args.mse_loss_weight * mse_loss  

        if args.use_normalization and eeg_normalizer is not None and clip_normalizer is not None:
            x_0 = eeg_normalizer.denormalize(x_0)
            x_1_pred = clip_normalizer.denormalize(x_1_pred)
            x_1 = clip_normalizer.denormalize(x_1)

        return val_loss, {
            'x_0': x_0.cpu(),
            'x_1_true': x_1.cpu(),
            'x_1_hat': x_1_pred.cpu(),
            'labels': labels,
            'x_1_hat_raw': x_1_pred,
            'x_1_true_raw': x_1,
        }

def train_one_epoch(epoch):
    flow.train()
    loss_total, train_batches = 0.0, 0

    for batch in flow_loaders['train']:
        loss, _ = process_batch(batch, is_train=True, epoch=epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        train_batches += 1

    return loss_total / train_batches


def validate_one_epoch(epoch):
    flow.eval()
    val_loss_total = 0.0
    inference_mse_total = 0.0
    inference_cos_total = 0.0
    correct, total_size, val_batches = 0, 0, 0
    x_0_list, x_1_list, x_1_hat_list, labels = [], [], [], []

    with torch.no_grad():
        for batch in flow_loaders['test']:
            val_loss, result = process_batch(batch, is_train=False)
            val_loss_total += val_loss.item()
            val_batches += 1

            mse = F.mse_loss(result['x_1_hat_raw'], result['x_1_true_raw'])
            cosine_sim = F.cosine_similarity(result['x_1_hat_raw'], result['x_1_true_raw'], dim=-1).mean()
            inference_mse_total += mse.item()
            inference_cos_total += cosine_sim.item()

            logits = result['x_1_hat_raw'] @ clip_loaders['test'].dataset.img_features.to(device).float().T
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == result['labels']).sum().item()
            total_size += result['labels'].size(0)

            x_0_list.append(result['x_0'])
            x_1_list.append(result['x_1_true'])
            x_1_hat_list.append(result['x_1_hat'])
            labels.append(result['labels'])

    return {
        "val_loss": val_loss_total / val_batches,
        "inference_mse": inference_mse_total / val_batches,
        "inference_cos": inference_cos_total / val_batches,
        "retrieval_acc": correct / total_size,
        "x_0_all": torch.cat(x_0_list, dim=0),
        "x_1_all": torch.cat(x_1_list, dim=0),
        "x_1_hat_all": torch.cat(x_1_hat_list, dim=0),
        "labels": torch.cat(labels, dim=0),
    }


def plot_PCA(train_pca, val_results, pca, epoch, save_to_wandb):
    # PCA + plot
    x_0_2d = pca.transform(val_results['x_0_all'].numpy())
    x_1_2d = pca.transform(val_results['x_1_all'].numpy())
    x_1_hat_2d = pca.transform(val_results['x_1_hat_all'].numpy())

    save_dir = f'./phd_results/flow_PCA/epoch_{epoch:03d}'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c='gray', label='CLIP training features', alpha=0.1)
    plt.scatter(x_1_2d[:, 0], x_1_2d[:, 1], c='black', label='CLIP target', alpha=0.5)
    plt.scatter(x_0_2d[:, 0], x_0_2d[:, 1], c='blue', label='EEG (before)', alpha=0.5)
    plt.scatter(x_1_hat_2d[:, 0], x_1_hat_2d[:, 1], c='red', label='EEG → CLIP (after)', alpha=0.5)

    # Draw lines between corresponding x_1 and x_1_hat points
    for (x1, x1_hat) in zip(x_1_2d, x_1_hat_2d):
        plt.plot([x1[0], x1_hat[0]], [x1[1], x1_hat[1]], c='gray', linewidth=0.5, alpha=0.4)

    plt.legend()
    plt.title(f'Flow Epoch {epoch}: {val_results["retrieval_acc"]:.4f}')
    plt.axis('off')
    plt.tight_layout()
    # Save locally
    pca_path = f'{save_dir}/step.png'
    plt.savefig(pca_path)

    # Save to WandB if requested
    if save_to_wandb:
        wandb.log({f'PCA/pca_w_score': wandb.Image(pca_path)})

    plt.close()


import plotly.graph_objects as go
import wandb
"""
def plot_PCA(train_pca, val_results, pca, epoch, save_to_wandb=False):
    # Project into PCA space
    x_0_2d = pca.transform(val_results['x_0_all'].cpu().numpy())
    x_1_2d = pca.transform(val_results['x_1_all'].cpu().numpy())
    x_1_hat_2d = pca.transform(val_results['x_1_hat_all'].cpu().numpy())

    # Create Plotly figure
    fig = go.Figure()

    # Training features
    fig.add_trace(go.Scatter(
        x=train_pca[:, 0], y=train_pca[:, 1],
        mode='markers',
        marker=dict(color='gray', opacity=0.1),
        name='CLIP training features',
        hoverinfo='skip'
    ))

    # EEG before
    fig.add_trace(go.Scatter(
        x=x_0_2d[:, 0], y=x_0_2d[:, 1],
        mode='markers',
        marker=dict(color='blue', opacity=0.7),
        name='EEG (before)',
        text=val_results['labels'].cpu().tolist() if isinstance(val_results['labels'], torch.Tensor) else val_results['labels'],
        hovertemplate='EEG (before)<br>%{text}<extra></extra>'
    ))

    # EEG after (predicted CLIP)
    fig.add_trace(go.Scatter(
        x=x_1_hat_2d[:, 0], y=x_1_hat_2d[:, 1],
        mode='markers',
        marker=dict(color='red', opacity=0.7),
        name='EEG → CLIP (after)',
        text=val_results['labels'].cpu().tolist() if isinstance(val_results['labels'], torch.Tensor) else val_results['labels'],
        hovertemplate='EEG → CLIP (after)<br>%{text}<extra></extra>'
    ))

    # CLIP target
    fig.add_trace(go.Scatter(
        x=x_1_2d[:, 0], y=x_1_2d[:, 1],
        mode='markers',
        marker=dict(color='black', opacity=0.7),
        name='CLIP target',
        text=val_results['labels'].cpu().tolist() if isinstance(val_results['labels'], torch.Tensor) else val_results['labels'],
        hovertemplate='CLIP target<br>%{text}<extra></extra>'
    ))

    # Connecting lines
    for i in range(len(x_0_2d)):
        fig.add_trace(go.Scatter(
            x=[x_0_2d[i, 0], x_1_hat_2d[i, 0], x_1_2d[i, 0]],
            y=[x_0_2d[i, 1], x_1_hat_2d[i, 1], x_1_2d[i, 1]],
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.4)', width=1),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=f'Flow Epoch {epoch}: Retrieval Acc {val_results["retrieval_acc"]:.4f}',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=700,
        height=700
    )

    # Save as HTML
    save_dir = f'/tmp/pca_epoch_{epoch:03d}'
    os.makedirs(save_dir, exist_ok=True)
    html_path = os.path.join(save_dir, 'pca_interactive.html')
    fig.write_html(html_path)

    # Log to WandB
    if save_to_wandb:
        wandb.log({f"PCA/Interactive_Epoch": wandb.Html(html_path)})
"""

def load_best_flow_model(flow, path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved flow model found at: {path}")
    
    state_dict = torch.load(path, map_location=device)
    flow.load_state_dict(state_dict)
    flow.eval()  # Set to eval mode just in case
    print(f"Loaded best flow model from: {path}")
    return flow


import wandb
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

from PIL import Image

def load_and_correct_image(image_path):
    """
    Loads an image, corrects its orientation using EXIF data, converts to RGB format, 
    and center crops it to the largest possible square.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        PIL.Image.Image: A corrected and square-cropped PIL Image object.
    """
    with Image.open(image_path) as image:
        # Correct orientation using EXIF metadata if available
        if hasattr(image, "_getexif") and image._getexif():
            exif = image._getexif()
            orientation_key = 274  # Key for orientation tag
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                # Apply transformations based on orientation value
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        
        # Convert to RGB to ensure consistent processing
        image = image.convert("RGB")
        
        # Perform center square cropping
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        image = image.crop((left, top, right, bottom))
    
    return image


from PIL import Image, ImageDraw

def generate_and_log_flow_reconstructions(
    flow_loader,
    flow,
    solver,
    generator,
    args,
    load_and_correct_image,
    eeg_normalizer=None,
    clip_normalizer=None,
    pca=None,
    train_data_for_pca=None,
):
    """
    Reconstruct and log images using flow + ODE and stable diffusion.
    Logs a static 3-panel plot (Original, Stage 1, Final) and optionally a GIF.
    """
    flow.eval()


    from torchvision.models.inception import inception_v3
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_stg1 = FrechetInceptionDistance(feature=2048, normalize=True).to('cuda' if torch.cuda.is_available() else 'cpu')
    fid_flow = FrechetInceptionDistance(feature=2048, normalize=True).to('cuda' if torch.cuda.is_available() else 'cpu')
    import torchvision.transforms as transforms
    transform_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])


    all_stg1_metrics = []
    all_flow_metrics = []

    from metrics import calculate_inception_score, compute_lpips, compute_ssim, color_histogram_distance

    from itertools import islice

    with torch.no_grad():
        for i, batch in enumerate(flow_loader):
        # for i, batch in enumerate(islice(flow_loader, 10)):  # For testing
            x_0 = batch['clip_eeg_embeddings'].to(device)
            x_1 = batch['clip_embeddings'].to(device)
            original_img_path = batch['img_paths'][0]

            try:
                original_img = load_and_correct_image(original_img_path)
            except Exception as e:
                print(f"Failed to load image at {original_img_path}: {e}")
                # continue

            if args.use_normalization and eeg_normalizer is not None:
                # PCA is fitted on normalized space
                x_0 = eeg_normalizer.normalize(x_0)
                x_1 = clip_normalizer.normalize(x_1)

            # Solve ODE flow
            x_intermediates = solver.sample(
                x_init=x_0,
                step_size=None,
                time_grid=torch.linspace(0, 1, steps=21).to(device),
                method="rk4",
                return_intermediates=args.log_gif_to_wandb
            )

            if not args.log_gif_to_wandb:
                x_intermediates = [x_intermediates]

            # PCA + plot
            x_0_pca = pca.transform(x_0.cpu().numpy())
            x_1_pca = pca.transform(x_1.cpu().numpy())
            x_intermediates_pca = pca.transform(np.array([ele.cpu().numpy() for ele in x_intermediates]).squeeze(1))

            # Generate images from latent embeddings
            if args.use_normalization and clip_normalizer is not None:
                # Denormalize for visualization
                x_intermediates = [clip_normalizer.denormalize(x) for x in x_intermediates]

            gif_frames = []
            for step_idx, x_embed in enumerate(x_intermediates):
                set_seed(args.seed)
                img = generator.generate(image_embeds=x_embed, text_prompt="")
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                gif_frames.append(img)
            
            # ----- Calculating metrics -------
            # --- Compute metrics for both Stage 1 and Final
            metrics_stage1 = {}
            metrics_final = {}

            stage1_img = gif_frames[0].resize((500, 500), Image.BILINEAR)
            final_img = gif_frames[-1].resize((500, 500), Image.BILINEAR)

            metrics_stage1['IS'] = calculate_inception_score(stage1_img, inception_model, device=device)
            metrics_stage1['cos_sim'] = F.cosine_similarity(x_0, x_1, dim=-1).item()
            metrics_stage1['lpips'] = compute_lpips(original_img, stage1_img, lpips_model=lpips_model, device=device)
            metrics_stage1['ssim'] = compute_ssim(original_img, gif_frames[0])
            metrics_stage1['color_dist'] = color_histogram_distance(original_img, gif_frames[0])

            metrics_final['IS'] = calculate_inception_score(final_img, inception_model, device=device)
            metrics_final['cos_sim'] = F.cosine_similarity(x_intermediates[-1], x_1, dim=-1).item()
            metrics_final['lpips'] = compute_lpips(original_img, final_img, lpips_model=lpips_model, device=device)
            metrics_final['ssim'] = compute_ssim(original_img, gif_frames[-1])
            metrics_final['color_dist'] = color_histogram_distance(original_img, gif_frames[-1])

            # Optionally add to a list for logging/export
            all_stg1_metrics.append(metrics_stage1)
            all_flow_metrics.append(metrics_final)

            # FID update
            img_real = transform_fid(original_img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            img_stg1 = transform_fid(gif_frames[0]).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            img_flow = transform_fid(gif_frames[-1]).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            fid_stg1.update(img_real, real=True)
            fid_stg1.update(img_stg1, real=False)
            fid_flow.update(img_real, real=True)
            fid_flow.update(img_flow, real=False)

            # --- Plotting
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(original_img)
            axs[0].set_title("Original")
            axs[0].axis('off')

            axs[1].imshow(gif_frames[0])
            axs[1].set_title("Stage 1")
            axs[1].axis('off')

            axs[2].imshow(gif_frames[-1])
            axs[2].set_title("Flow-Reconstructed")
            axs[2].axis('off')

            def format_metrics(m):
                keys = list(m.keys())
                lines = []
                for i in range(0, len(keys), 2):
                    k1 = keys[i]
                    v1 = f"{m[k1]:.4f}"
                    if i + 1 < len(keys):
                        k2 = keys[i + 1]
                        v2 = f"{m[k2]:.4f}"
                        lines.append(f"{k1}: {v1}    {k2}: {v2}")
                    else:
                        lines.append(f"{k1}: {v1}")
                return "\n".join(lines)

            axs[1].text(0.5, -0.02, format_metrics(metrics_stage1), transform=axs[1].transAxes,
                        fontsize=10, ha='center', va='top', wrap=True)
            axs[2].text(0.5, -0.02, format_metrics(metrics_final), transform=axs[2].transAxes,
                        fontsize=10, ha='center', va='top', wrap=True)

            # Plot title
            title = original_img_path.split("/")[-1].replace(".jpg", "")
            fig.suptitle(f"{title}", fontsize=14)
            #fig.subplots_adjust(top=0.85, bottom=0.1)
            fig.tight_layout()

            wandb.log({f"Flow Reconstruction/{i}": wandb.Image(fig)})
            plt.close(fig)

            # --- Animated GIF (Left: Original, Right: Flow Steps)
            if args.log_gif_to_wandb:
                gif_with_original = []

                # Resize all images to same size
                target_size = (500, 500)
                original_resized = original_img.resize(target_size)

                for step_idx, frame in enumerate(gif_frames):
                    # Reconstruction
                    frame_resized = frame.resize(target_size)
                    combined = Image.new("RGB", (target_size[0] * 2, target_size[1]))
                    combined.paste(original_resized, (0, 0))
                    combined.paste(frame_resized, (target_size[0], 0))
                    
                    # PCA plot
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.scatter(train_data_for_pca[:, 0], train_data_for_pca[:, 1], c='lightgray', alpha=0.1, label='CLIP training features')
                    ax.scatter(x_0_pca[:, 0], x_0_pca[:, 1], c='blue', label='EEG (x₀)', s=50)
                    ax.scatter(x_1_pca[:, 0], x_1_pca[:, 1], c='black', label='CLIP target (x₁)', s=50)
                    ax.scatter(x_intermediates_pca[:step_idx+1, 0], x_intermediates_pca[:step_idx+1, 1], c='red', label=f'Step {step_idx}', s=50)
                    ax.legend(loc='lower left', fontsize=6)
                    ax.set_title('PCA Flow Trajectory')
                    ax.axis('off')

                    # Convert matplotlib fig to image
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    pca_img = Image.open(buf).convert("RGB").resize((target_size[0] * 2, target_size[1]))
                    plt.close(fig)

                    # ----- Final Frame: Stacked Image + PCA Plot -----
                    final_frame = Image.new("RGB", (target_size[0] * 2, target_size[1] * 2))
                    final_frame.paste(combined, (0, 0))
                    final_frame.paste(pca_img, (0, target_size[1]))
                    gif_with_original.append(final_frame)


                gif_buffer = io.BytesIO()
                gif_with_original[0].save(
                    gif_buffer,
                    format='GIF',
                    save_all=True,
                    append_images=gif_with_original[1:],
                    duration=300,
                    loop=0
                )
                gif_buffer.seek(0)
                wandb.log({f"Flow Trajectory GIF/{i}": wandb.Video(gif_buffer, format="gif")})
                

    def average_metrics(metric_list):
        return {k: np.mean([m[k] for m in metric_list]) for k in metric_list[0].keys()}
        
    avg_stage1 = average_metrics(all_stg1_metrics)
    avg_flow = average_metrics(all_flow_metrics)

    # Add FID scores
    avg_stage1['FID'] = fid_stg1.compute().item()
    avg_flow['FID'] = fid_flow.compute().item()

    for metric in avg_stage1:
        # Data format: [ [group, value] ]
        data = [
            ["Stage 1", avg_stage1[metric]],
            ["Flow", avg_flow[metric]]
        ]

        # Create table
        table = wandb.Table(data=data, columns=["group", "value"])

        # Log individual bar chart for this metric
        wandb.log({
            f"Metric/{metric}": wandb.plot.bar(
                table,
                "group",   # x-axis: Stage 1 vs Flow
                "value",   # y-axis: metric value
                title=f"{metric} Comparison"
            )
        })

    print(f"Logged flow reconstruction for {original_img_path} to WandB.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/work3/s184984/repos/EEG_Image_decode/eeg_dataset/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--model_dir', type=str, default='./models/EEG_encoder', help='Directory to save output results')    
    parser.add_argument('--seed', type=int, default=42, help='Number of epochs')

    parser.add_argument('--pth_name', type=str, default='best', help='Name of the ATMS model to load')

    # Model params
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')  
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--alpha', type=float, default=0.90, help='alpha value to weigh the loss')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs') 
    parser.add_argument('--flow_epochs', type=int, default=100, help='Number of epochs')    
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs for warmup')  
    parser.add_argument('--use_normalization', action='store_true', help='Use normalization for training')
    parser.add_argument('--reconstruction', action='store_true', help='Use normalization for training')

    # Params
    parser.add_argument('--skewed_timesteps', action='store_true', help='Use skewed timesteps for training')
    parser.add_argument('--train_ATMS', action='store_true', help='train ATMS model')
    parser.add_argument('--train_flow', action='store_true', help='train flow model')

    # Data params
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--flow_batch_size', type=int, default=64, help='Batch size')

    # Loss parameters
    parser.add_argument('--loss_fn', type=str, choices=['clip', 'vicreg', 'softContrastive', 'softHybridContrastive'], default='clip', help='loss function, see loss.py for more info')
    parser.add_argument('--uniformity_loss_weight', type=float, default=0, help='Add terms to CLIP loss for uniformity and alignment')
    parser.add_argument('--cosine_loss_weight', type=float, default=0, help='Add cosine similarity loss to the flow model')
    parser.add_argument('--mse_loss_weight', type=float, default=0, help='Add MSE loss to the flow model')

    # Model targets
    parser.add_argument('--atms_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')
    parser.add_argument('--diffusion_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')

    # Wandb
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--project', type=str, default="EEG_image_reconstruction", help='WandB project name')
    parser.add_argument('--name', type=str, default="flow_training", help='Experiment name')
    parser.add_argument('--entity', type=str, default="philliphoejbjerg", help='WandB entity name')
    parser.add_argument('--log_gif_to_wandb', action='store_true', help='Log GIFs to WandB')
    parser.add_argument('--pca', action='store_true', help='Log PCA to WandB')

    # Freeze ATMS
    parser.add_argument('--freeze_ATMS', action='store_true', help='Freeze ATMS model parameters')

    os.environ["WANDB_API_KEY"] = "b0c5da2aac89929c85f768b56e5f260e287064ab"
    os.environ["WANDB_MODE"] = 'online'


    args = parser.parse_args()

    set_seed(args.seed)


    # Set device based on the argument
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device(args.device)        
    else:
        device = torch.device('cpu')

    for sub in args.subjects:

        # Path for model
        aligner_model_path = f"{args.model_dir}/{args.name}/{sub}" if args.insubject else f"{args.model_dir}/{args.name}/across"
        os.makedirs(aligner_model_path, exist_ok=True)             

        # init wandb logger
        logger = wandb_logger(args, sub) if args.logger else None

        # instantiate ATMS model
        eeg_model = ATMS(args = args) # globals()[args.encoder_type]()
        eeg_model.to(device)
        logger.watch(eeg_model,logger) 

        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)

        clip_dataset, clip_loaders = load_ATMS_loaders(args, sub, device) # Load EEG dataset
            
        # ----------- ATMS training -----------
        if args.train_ATMS:

            # Train and validation loop
            best_accuracy_val = 0.0
            best_accuracy_train = 0.0
            best_val_loss = float('inf')
            best_epoch = 0
            for epoch in tqdm(range(args.epochs), desc = "Epoch"):
                
                # Train one epoch
                train_loss, train_accuracy, average_MSE = train_clip_aligner(sub, eeg_model, clip_loaders['train'], optimizer, device, clip_dataset['train'].text_features, clip_dataset['train'].img_features, args=args)

                # Evaluate model
                epoch_results = validate_clip_aligner(sub, eeg_model, clip_loaders['test'], device, clip_dataset['test'], args, epoch)
                epoch_results['train_loss'], epoch_results['train_accuracy'], epoch_results['train_MSE'] = train_loss, train_accuracy, average_MSE
                logger.log(epoch_results)

                # If the test accuracy of the current epoch is the best, save the model and related information
                if epoch_results['test_accuracy'] > best_accuracy_val:
                    best_accuracy_val = epoch_results['test_accuracy']
                    best_epoch = epoch + 1

                    torch.save(eeg_model.state_dict(), f"{aligner_model_path}/best.pth")
                    print(f"Model saved in {aligner_model_path}!, Epoch: {best_epoch}, Accuracy: {best_accuracy_val:.4f}, MSE: {epoch_results['test_MSE']:.4f}, Loss: {epoch_results['test_loss']:.4f}")

                # if validation loss is better than before
                if epoch_results['test_loss'] < best_val_loss:
                    best_val_loss = epoch_results['test_loss']
                    torch.save(eeg_model.state_dict(), f"{aligner_model_path}/best_val.pth")
                    print(f"Model saved in {aligner_model_path}!, Epoch: {best_epoch}, Validation Loss: {best_val_loss:.4f}")

                # Save model with best train accuracy
                if epoch_results['train_accuracy'] > best_accuracy_train:
                    best_accuracy_train = epoch_results['train_accuracy']
                    torch.save(eeg_model.state_dict(), f"{aligner_model_path}/best_train.pth")
                    print(f"Model saved in {aligner_model_path}!, Epoch: {best_epoch}, Train Accuracy: {best_accuracy_train:.4f}, MSE: {epoch_results['train_MSE']:.4f}, Loss: {epoch_results['train_loss']:.4f}")

                print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {epoch_results['train_loss']:.4f}, Train Accuracy: {epoch_results['train_accuracy']:.4f}, Train MSE: {epoch_results['train_MSE']:.4f}")
                print(f"Epoch {epoch + 1}/{args.epochs} - Test Loss: {epoch_results['test_loss']:.4f}, Test Accuracy: {epoch_results['test_accuracy']:.4f}, Test MSE: {epoch_results['test_MSE']:.4f}")
            
        # ---------------- FLOW MATCHING -------------------
        if args.train_flow or args.reconstruction:

            eeg_model = load_best_model(eeg_model, sub, device, args)

            # flow = ResMLPFlow(dim=1024, hidden_dim=1024, num_blocks=4).to(device)

            flow = TransformerFlow(
                dim=1024,
                time_dim=128,
                hidden_dim=2048,
                num_heads=8,
                num_blocks=4  # Can increase later
            ).to(device)

            from flow_matching.solver.ode_solver import ODESolver
            solver = ODESolver(velocity_model=flow)

            # probability path for the flow model
            path = CondOTProbPath()

            # ----- Normalizer -----
            eeg_normalizer, clip_normalizer = None, None

            # We need training set if training or using normalization
            if args.use_normalization or args.pca or args.train_flow:

                flow_loaders = load_FLOW_loaders(args, sub, device, clip_loaders, eeg_model) # Load EEG dataset
                # Fit normalizers
                eeg_train_feats = flow_loaders['train'].dataset.clip_eeg_embeddings.view(-1, 1024)
                clip_train_feats = clip_loaders['train'].dataset.img_features.view(1654, 10, 1, 1024)
                clip_train_feats = clip_train_feats.repeat(1, 1, 4, 1).view(-1, 1024)

                eeg_normalizer = Normalizer()
                clip_normalizer = Normalizer()

                eeg_normalizer.fit(eeg_train_feats)
                clip_normalizer.fit(clip_train_feats)

                # Move to GPU
                eeg_normalizer.to(device)
                clip_normalizer.to(device)

                # Fit PCA
                # PCA for sanity checking
                from sklearn.decomposition import PCA

                if args.use_normalization: # Normalize before fitting PCA
                    clip_train_feats = clip_normalizer.normalize(clip_train_feats)

                pca = PCA(n_components=2)
                train_pca = pca.fit_transform(clip_train_feats.cpu().numpy()) 
            # ----------------------


        if args.train_flow:
            print("Training flow matching model...")
            del clip_dataset

            # optimizer = torch.optim.Adam(flow.parameters(), 1e-4)
            optimizer = torch.optim.AdamW(flow.parameters(), lr=3e-4, weight_decay=1e-4)

            loss_fn = nn.MSELoss()

            best_val_loss = float('inf')
            best_retrieval = 0.0

            # Train and validate
            for epoch in range(args.flow_epochs):

                # Train one epoch
                train_loss = train_one_epoch(epoch)
                # Validate one epoch
                val_results = validate_one_epoch(epoch)

                # Print results
                print(f"Epoch {epoch + 1}/{args.flow_epochs}")
                print(f"▶ Train Loss: {train_loss:.4f}")
                print(f"▶ Val Flow Loss: {val_results['val_loss']:.4f}")
                print(f"▶ Val Embedding MSE: {val_results['inference_mse']:.4f}")
                print(f"▶ Val Cosine Similarity: {val_results['inference_cos']:.4f}")
                print(f"▶ Val Retrieval Accuracy: {val_results['retrieval_acc']:.4f}")

                plot_PCA(train_pca, val_results, pca, epoch, save_to_wandb=(val_results['val_loss'] < best_val_loss))

                # Save model if it's the best so far
                if val_results['val_loss'] < best_val_loss:
                    best_val_loss = val_results['val_loss']
                    torch.save(flow.state_dict(), f"{aligner_model_path}/flow_best_loss.pth")
                    print(f"Saved new best model at epoch {epoch+1}: Val loss")

                if val_results['retrieval_acc'] > best_retrieval:
                    best_retrieval = val_results['retrieval_acc']
                    torch.save(flow.state_dict(), f"{aligner_model_path}/flow_best_retrieval.pth")
                    print(f"Saved new best model at epoch {epoch+1}: Retrieval accuracy")

                # Log results
                logger.log({
                    "flow_epoch": epoch,
                    "flow_train/loss": train_loss,
                    "flow_val/loss": val_results['val_loss'],
                    "flow_val/inference_mse": val_results['inference_mse'],
                    "flow_val/inference_cosine": val_results['inference_cos'],
                    "flow_val/retrieval_accuracy": val_results['retrieval_acc'],
                })


        # Reconstruct images
        if args.reconstruction:     

            from custom_pipeline_phil import Generator4Embeds
            generator = Generator4Embeds(num_inference_steps=4, device=device, force_download=True)       

            # Image reconstruction:
            # Load the best flow model
            best_flow_path = f"{aligner_model_path}/flow_best_loss.pth"
            flow = load_best_flow_model(flow, best_flow_path)

            flow_loaders = load_FLOW_loaders(args, sub, device, clip_loaders, eeg_model, split='test', reconstruction = True)

            generate_and_log_flow_reconstructions(
                flow_loader=flow_loaders['test'],
                flow=flow,
                solver=solver,
                generator=generator,
                args=args,
                load_and_correct_image=load_and_correct_image,
                eeg_normalizer = eeg_normalizer if args.use_normalization else None,
                clip_normalizer=clip_normalizer if args.use_normalization else None,
                pca=pca if args.pca else None,
                train_data_for_pca=train_pca if args.pca else None,
            )