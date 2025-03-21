import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import urllib.request
proxies = urllib.request.getproxies()
#os.environ['HTTP_PROXY'] = proxies['http']
#os.environ['HTTPS_PROXY'] = proxies['https']

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from eegdatasets_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from PIL import Image

from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW

from models import Config, iTransformer, PatchEmbedding, ResidualAdd, FlattenHead, Enc_eeg, Proj_eeg, ATMS
from utils_phil import extract_id_from_string, plot_color_histograms, plot_lbp_histograms, compare_textures, compare_colors, save_img

from custom_pipeline_phil import Generator4Embeds
from diffusion_prior import *

import datetime

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Function to extract the object name
def extract_object_name(s):
    # Split by the first underscore
    parts = s.split('_', 1)
    # Replace any underscores in the object name with spaces
    object_name = parts[1].replace('_', ' ')
    return object_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/work3/s184984/repos/EEG_Image_decode/eeg_dataset/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')    
    parser.add_argument('--model_dir', type=str, default='./models/EEG_encoder', help='Directory to save output results')    
    parser.add_argument('--experiment_name', type=str, default='', help='Directory to save output results')    
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings/EEG_encoder', help='Directory to save output results')        
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--diffusion_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--diffusion_epochs', type=int, default=150, help='Number of epochs') 
    parser.add_argument('--seed', type=int, default=42, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='ATMS', help='Encoder type')
    parser.add_argument('--atms_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')
    parser.add_argument('--diffusion_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')    
    parser.add_argument('--alpha', type=float, default=0.90, help='alpha value to weigh the loss')

    parser.add_argument('--alpha_scheduler', action='store_true', help='Slowly transitions from CLIP loss to MSE')

    parser.add_argument('--train_EEG_aligner', action='store_true', help='Trains the EEG embedder')
    parser.add_argument('--use_diffusion_prior', action='store_true', help='Trains the diffusion prior to the pipeline')    
    parser.add_argument('--use_text_prompt', action='store_true', help='Uses the original class name within the generation as a text prompt')
    parser.add_argument('--generate_from_CLIP_feats', action='store_true', help='Uses the original class name within the generation as a text prompt')

    args = parser.parse_args()

    if args.generate_from_CLIP_feats:
        model = 'from_img_feats'

    else:
        model = f"{args.atms_target}_{args.diffusion_target}_Diffusion_prior" if args.use_diffusion_prior else f"{args.atms_target}_ATMS"
        model += "_w_text_prompt" if args.use_text_prompt else ""
        if args.alpha_scheduler:
            model += '_alpha_scheduler'

    generated_image_dir = f"./generated_imgs/{model}" 

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    # Initialize torch cosine similarity
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Loading for sub 8
    for sub in args.subjects:

        # init wandb logger
        logger = wandb_logger(args) if args.logger else None


        # instantiate model
        eeg_model = ATMS() # globals()[args.encoder_type]()
        eeg_model.to(device)
        PATH = f"{args.model_dir}/{args.atms_target}_{args.encoder_type}/{sub}/lr{args.lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}" if args.insubject else f"{args.model_dir}/across/{args.atms_target}_{args.encoder_type}/lr{args.lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}"
        if args.alpha_scheduler:
            PATH += '_alpha_scheduler'        
        eeg_model.load_state_dict(torch.load(f"{PATH}/best.pth", weights_only=False, map_location=torch.device(device)))
        eeg_model.eval()
        logger.watch(eeg_model,logger) 

        if args.use_diffusion_prior:
            diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
            pipe = Pipe(diffusion_prior, device=device)
            file_path = f"{args.model_dir}/{args.atms_target}_{args.diffusion_target}_Diffusion_prior/{sub}/lr{args.diffusion_lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}" if args.insubject else f"{args.model_dir}/across/{args.atms_target}_{args.diffusion_target}_Diffusion_prior/lr{args.diffusion_lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}"
            if args.alpha_scheduler:
                file_path += '_alpha_scheduler'               
            save_path = f"{file_path}/best.pth"
            pipe.diffusion_prior.load_state_dict(torch.load(save_path, map_location=device))        

        # Setup optimizer
        # optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)

        # Load datasets 
        if args.insubject: # per subject
            # train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True, device=device)
            test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False, device=device)
        else:
            # Leave one subject out
            # train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=True)
            test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=False)

        # Loaders
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        # Add rest of pipeline..
        print("Loading Generator")

        # GET TOKEN ON HUGGINGFACE
        # from huggingface_hub import login
        # login(token="YOUR PERSONAL HF TOKEN, WILL BE SAVED IN CACHE")
        generator = Generator4Embeds(num_inference_steps=4, device=device, force_download=True)

        scores = {'img_path': [], 'Cosine Similarity': [], 'LBP Correlation': [], 'Color Correlation': []}

        with torch.no_grad():
            for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(test_loader):

                print(f"Generating batch: {batch_idx}")

                # Load the original image from the file path
                original_img_path = img[0]  # Assuming batch size is 1, get the first path in img tuple
                filename = f"comparison_{batch_idx}_{'___'.join(original_img_path.split('/')[-2:]).split('.')[0]}.png"
                
                eeg_data = eeg_data.to(device)
                labels = labels.to(device) 
                
                batch_size = eeg_data.size(0)
                subject_ids = torch.full((batch_size,), extract_id_from_string(sub), dtype=torch.long).to(device)
                
                # Generate based on original CLIP embeddings
                if args.generate_from_CLIP_feats:
                    clip_eeg_emb = img_features

                else:
                    # Get model outputs
                    clip_eeg_emb = eeg_model(eeg_data, subject_ids)

                    if args.use_diffusion_prior:
                        clip_eeg_emb = pipe.generate(c_embeds = clip_eeg_emb, num_inference_steps=50, guidance_scale=5.0).to(dtype=torch.float16)

                # Generate image
                prompt = f"A photorealistic image of a {extract_object_name(original_img_path.split('/')[-2])}, centered and isolated, with a regular background" if args.use_text_prompt else ''
                generated_image = generator.generate(image_embeds = clip_eeg_emb, text_prompt=prompt) if (args.use_diffusion_prior or args.generate_from_CLIP_feats) else generator.generate(image_embeds = clip_eeg_emb, text_prompt=prompt)

                try:
                    original_img = Image.open(original_img_path).convert("RGB")  # Convert to RGB for consistency
                    img_np = np.array(original_img)  # Convert to numpy array for matplotlib
                    print(f"Original image loaded with shape: {img_np.shape}")
                except Exception as e:
                    print(f"Error loading original image from path {original_img_path}: {e}")
                    continue  # Skip this batch if image loading fails

                # Convert generated image to numpy if it's a PIL Image
                try:
                    if isinstance(generated_image, Image.Image):
                        gen_img_np = np.array(generated_image)
                        print(f"Generated image loaded with shape: {gen_img_np.shape}")
                    else:
                        raise TypeError("Generated image is not a PIL Image.")
                except Exception as e:
                    print(f"Error processing generated image: {e}")
                    continue   

                # and save generated image
                save_img(gen_img_np, f"{generated_image_dir}/{sub}/alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}/{'/'.join(img[0].split('/')[-2:])}")             

                # Calculate scores
                scores['img_path'].append(img[0])
                scores['Cosine Similarity'].append(cosine_sim(clip_eeg_emb, img_features).cpu().item()) # Cosine similarity
                scores['Color Correlation'].append(compare_colors(img_np, gen_img_np, measure='corr'))  # Assuming this returns a dictionary
                scores['LBP Correlation'].append(compare_textures(img_np, gen_img_np))

                # Create a 1x3 subplot grid
                fig = make_subplots(rows=1, cols=3, column_widths=[0.4, 0.4, 0.2], specs=[[{"type": "image"}, {"type": "image"}, {"type": "xy"}]])

                # Original Image in the first column
                fig.add_trace(go.Image(z=img_np), row=1, col=1)
                fig.update_xaxes(visible=False, showticklabels=False, row=1, col=1)
                fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1)

                # Generated Image in the second column
                fig.add_trace(go.Image(z=gen_img_np), row=1, col=2)
                fig.update_xaxes(visible=False, showticklabels=False, row=1, col=2)
                fig.update_yaxes(visible=False, showticklabels=False, row=1, col=2)

                # Score Text in the third column as an annotation
                score_text = "<b>Scores:</b>"
                for k, v in {key:scores[key] for key in scores if key!='img_path'}.items():
                    if k == 'Color Correlation':
                        score_text += f"<br><b>{k}:</b><br>"
                        for color, corr in v[-1].items():
                            score_text += f"{color}: {corr:.2f}, "
                    else:
                        # print(f"<b>{k}:</b> {v[-1].item():.2f}<br>")
                        score_text += f"<br><b>{k}:</b> {v[-1]:.2f}"

                # Add text as an annotation within the third subplot area
                fig.add_annotation(
                    x=0.95,  # Adjust for alignment in the third column
                    y=0.90,  # Center vertically in the third column
                    text=score_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    font=dict(size=14),
                    align="left"
                )

                # Update layout for spacing, centering, and lowering the title
                fig.update_layout(
                    title_text=f"Label: {text[0][16:].capitalize()}",
                    title_x=0.4,          # Adjust to center above the first two images
                    title_y=0.85,         # Lower title closer to images
                    title_xanchor="center",
                    font=dict(
                        family="Editorial New Italic",  # Consistent font family
                        size=14,
                    ),
                    width=1000,
                    height=500,
                    margin=dict(t=80, b=50, l=50, r=50)  # Adjust top margin for closer title
                )                

                # Save the figure as HTML or static image
                os.makedirs(f"{generated_image_dir}/{sub}/alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}", exist_ok=True)
                fig.write_image(f"{generated_image_dir}/{sub}/alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}/{filename}")                
                
        # Save scores..!
        df = pd.DataFrame({key: pd.Series(value) for key, value in scores.items()})
        os.makedirs(f'outputs/{model}/{sub}/alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}', exist_ok=True)
        df.to_csv(f'outputs/{model}/{sub}/alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}/scores.csv', encoding='utf-8', index=False, )

        logger.finish()