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

from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
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
import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW

from models import Config, iTransformer, PatchEmbedding, ResidualAdd, FlattenHead, Enc_eeg, Proj_eeg, ATMS
from utils_phil import extract_id_from_string, set_seed

from custom_pipeline_phil import Generator4Embeds
from diffusion_prior import *

# For WandB login
from dotenv import load_dotenv
# Load environment variables from .env file
import datetime

# Train model function
def train_clip_aligner(sub, eeg_model, dataloader, optimizer, epoch, device, text_features_all, img_features_all, args):
    
    eeg_model.train()
    
    # init loss func
    mse_loss_fn = nn.MSELoss()

    # For grabbing correct embeddings
    all_clip_emb = {'text': text_features_all.to(device).float(), # (n_cls, d) # prev: text_features_all,
                    'image': (img_features_all[::10]).to(device).float() # prev: img_features_all
                    }

    # Define a mapping for the feature types
    feature_mapping = {
        'text': 'text_features',
        'image': 'img_features'
    }

    # initialize 
    total_loss, total_size, correct, total_MSE = 0, 0, 0, 0 
    features_list = []  # List to store features

    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        
        # TODO: possibly do within dataloader? 
        eeg_data = eeg_data.to(device)

        # Grabbing either img_features or text features
        clip_emb = locals()[feature_mapping[args.atms_target]].to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
        subject_ids = torch.full((batch_size,), extract_id_from_string(sub), dtype=torch.long).to(device)

        # get outs
        clip_eeg_emb = eeg_model(eeg_data, subject_ids).float()
        logit_scale = eeg_model.logit_scale # a learnable parameter
        features_list.append(clip_eeg_emb)

        # MSE calculation
        total_MSE += mse_loss_fn(clip_eeg_emb, clip_emb).item()

        # loss_func: --> clip_loss()
        if args.loss_fn == 'clip':
            clip_loss = eeg_model.loss_func(clip_eeg_emb, clip_emb, logit_scale)

            mse_loss =  mse_loss_fn(clip_eeg_emb, clip_emb)
    
            loss = (10 * (args.alpha * mse_loss) + (10 * ((1 - args.alpha) * clip_loss)))
            
            # backprop
            loss.backward()
            optimizer.step()
            
            # Measure similarity between EEG embeddings and CLIP image embeddings - to get actual prediction, and thereby accuracy. NOT needed for training
            logits_img = logit_scale * clip_eeg_emb @ all_clip_emb[args.atms_target].T

        elif args.loss_fn == 'vicreg' or args.loss_fn == 'softContrastive' or args.loss_fn == 'softHybridContrastive':
            loss = eeg_model.loss_func(clip_eeg_emb, clip_emb)

            # backprop
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

    return average_loss, accuracy, torch.cat(features_list, dim=0), average_MSE

def train_diffusion_prior(sub, diffusion_prior, dataloader, device, args, logger):

    learning_rate=1e-3

    # TODO: OBS, this part doesn't include possibility of training multiple subjects...!

    # number of parameters
    pipe = Pipe(diffusion_prior, device=device, logger=logger)

    # Train model, and save the best one
    pipe.train(dataloader, sub, args, num_epochs=args.diffusion_epochs) # to 0.142 
    
    # Or load a trained one:
    # file_path = f"{args.model_dir}/Diffusion_prior/{sub}/lr{learning_rate}" if args.insubject else f"{args.model_dir}/across/{args.encoder_type}/lr{learning_rate}"
    # save_path = f"{file_path}/best.pth"
    # pipe.diffusion_prior.load_state_dict(torch.load(save_path, map_location=device))


def evaluate_model(sub, eeg_model, dataloader, device, all_clip_text_emb, all_clip_img_emb, k, args):
    
    eeg_model.eval()

    # init loss func
    mse_loss_fn = nn.MSELoss()

    # For grabbing correct embeddings
    all_clip_emb = {'text': all_clip_text_emb.to(device).float(), 
                    'image': all_clip_img_emb.to(device).float()
                    }
    
    # initialize 
    all_labels = set(range(all_clip_emb['text'].size(0)))
    total_loss, total_size, correct, top5_acc, top5_correct_count, total_MSE = 0, 0, 0, 0, 0, 0

    features_list = []

    # Define a mapping for the feature types
    feature_mapping = {
        'text': 'text_features',
        'image': 'img_features'
    }    

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            # TODO: possibly do within dataloader? 
            eeg_data = eeg_data.to(device)

            clip_emb = locals()[feature_mapping[args.atms_target]].to(device).float()

            labels = labels.to(device)
            
            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_ids = torch.full((batch_size,), extract_id_from_string(sub), dtype=torch.long).to(device)
            
            # get outs
            clip_eeg_emb = eeg_model(eeg_data, subject_ids).float()
            logit_scale = eeg_model.logit_scale
            features_list.append(clip_eeg_emb)

            # MSE calculation
            total_MSE += mse_loss_fn(clip_eeg_emb, clip_emb).item()
                
            # loss_func: --> clip_loss()
            if args.loss_fn == 'clip':
                # calculate loss
                clip_loss = eeg_model.loss_func(clip_eeg_emb, clip_emb, logit_scale)
                mse_loss =  mse_loss_fn(clip_eeg_emb, clip_emb)
                loss = (10 * (args.alpha * mse_loss) + (10 * ((1 - args.alpha) * clip_loss)))
                    
                total_loss += loss.item()
                
                # Measure similarity between EEG embeddings and CLIP image embeddings - to get actual prediction, and thereby accuracy. NOT needed for training

            elif args.loss_fn == 'vicreg' or args.loss_fn == 'softContrastive' or args.loss_fn == 'softHybridContrastive':
                loss = eeg_model.loss_func(clip_eeg_emb, clip_emb)

                total_loss += loss.item()

                # Measure similarity between EEG embeddings and CLIP image embeddings - to get actual prediction, and thereby accuracy. NOT needed for training
                # Perhaps use logit_scale

            
            for idx, label in enumerate(labels):
                # First select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})

                # sample classes
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = all_clip_emb[args.atms_target][selected_classes]

                # find predicted labels for specific k labels
                # TODO: check for logit_scale
                logits_single = logit_scale * clip_eeg_emb[idx] @ selected_img_features.T if args.loss_fn == 'clip' else clip_eeg_emb[idx] @ selected_img_features.T

                # predict by argmax
                predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                if predicted_label == label.item():
                    correct += 1

                if k > 10:
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1  

                total_size += 1

            # del eeg_data, eeg_features, img_features

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total_size
    top5_acc = top5_correct_count / total_size
    average_MSE = total_MSE / (batch_idx+1)

    return average_loss, accuracy, top5_acc, features_list, average_MSE


def evaluate_and_append_results(sub, eeg_model, test_loader, device, test_dataset, args, 
                                test_losses, test_accuracies, v2_accs, v4_accs, v10_accs, epoch,
                                logger):
    
    # Evaluate the model for various values of k
    test_loss, test_accuracy, top5_acc, features_list, average_MSE = evaluate_model(sub, eeg_model, test_loader, device, 
                                                        test_dataset.text_features, 
                                                        test_dataset.img_features, k=200, args=args)
    
    # TODO: Possibly just reuse "features_list" for remaining scripts
    _, v2_acc, _, _, _ = evaluate_model(sub, eeg_model, test_loader, device, 
                                  test_dataset.text_features, 
                                  test_dataset.img_features, k=2, args=args)
    
    _, v4_acc, _, _, _ = evaluate_model(sub, eeg_model, test_loader, device, 
                                  test_dataset.text_features, 
                                  test_dataset.img_features, k=4, args=args)
    
    _, v10_acc, _, _, _ = evaluate_model(sub, eeg_model, test_loader, device, 
                                   test_dataset.text_features, 
                                   test_dataset.img_features, k=10, args=args)
    
    _, v50_acc, v50_top5_acc, _, _ = evaluate_model(sub, eeg_model, test_loader, device, 
                                              test_dataset.text_features, 
                                              test_dataset.img_features, k=50, args=args)
    
    _, v100_acc, v100_top5_acc, _, _ = evaluate_model(sub, eeg_model, test_loader, device, 
                                                test_dataset.text_features, 
                                                test_dataset.img_features, k=100, args=args)

    # Append scores
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    v2_accs.append(v2_acc)
    v4_accs.append(v4_acc)
    v10_accs.append(v10_acc)

    # Append results for this epoch
    epoch_results = {
        "epoch": epoch + 1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc": top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc": v50_top5_acc,
        "v100_top5_acc": v100_top5_acc,
        "test_MSE": average_MSE
    }

    logger.log(epoch_results)

    print(f"Epoch {epoch + 1}/{args.epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
    print(f"Epoch {epoch + 1}/{args.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")

    return epoch_results, features_list

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, 
                v2_accs, v4_accs, v10_accs, best_epoch_info, plot_name):
    
    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss plot
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy plot
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you have calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information you want to annotate
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle(plot_name, fontsize=16, y=1.05)
    plt.savefig(plot_name)


def load_and_freeze_best_model(eeg_model, args):

    # Getting output from the best model
    PATH = f"{args.model_dir}/{args.atms_target}_{args.encoder_type}/{sub}/{args.name}" if args.insubject else f"{args.model_dir}/across/{args.atms_target}_{args.encoder_type}/{args.name}"
    eeg_model.load_state_dict(torch.load(f"{PATH}/best.pth", weights_only=False, map_location=torch.device(device)))
    # Freezing the original embedder
    eeg_model.eval()
    for param in eeg_model.parameters():
        param.requires_grad = False

    return eeg_model


if __name__ == '__main__':

    proxies = urllib.request.getproxies()


    # What has to be trained? 
    # --> CLIP EEG encoder
    # --> VAE encoder for low levels
    # --> IP adapter? 
    #   (X: ViT-H-14_features_train)
    #   (y: ViT-L-14_features_GIT_train.pt) <-- TODO: where are these generated? They're just loaded within image_adapter.ipynb....

    # CLIP encoder

    # --data_path /Users/pchho/Documents/repos/EEG_Image_decode/eeg_dataset/Preprocessed_data_250Hz --gpu mps --train_EEG_aligner --train_diffusion_prior --atms_target image --diffusion_target image --name vicreg_v2 --batch_size 64 --loss_fn vicreg --subjects sub-08

    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/work3/s184984/repos/EEG_Image_decode/eeg_dataset/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')    
    parser.add_argument('--model_dir', type=str, default='./models/EEG_encoder', help='Directory to save output results')    
    parser.add_argument('--experiment_name', type=str, default='', help='Directory to save output results')    
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings/EEG_encoder', help='Directory to save output results')        
    parser.add_argument('--project', type=str, default="EEG_image_reconstruction", help='WandB project name')
    parser.add_argument('--entity', type=str, default="philliphoejbjerg", help='WandB entity name')
    parser.add_argument('--name', type=str, default="modified_loss", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--diffusion_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs') 
    parser.add_argument('--diffusion_epochs', type=int, default=150, help='Number of epochs') 
    parser.add_argument('--seed', type=int, default=42, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'mps'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='ATMS', help='Encoder type')
    parser.add_argument('--atms_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')
    parser.add_argument('--diffusion_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')    
    parser.add_argument('--alpha', type=float, default=0.90, help='alpha value to weigh the loss')

    parser.add_argument('--alpha_scheduler', action='store_true', help='Slowly transitions from CLIP loss to MSE')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs') 

    parser.add_argument('--train_EEG_aligner', action='store_true', help='Trains the EEG embedder')
    parser.add_argument('--train_diffusion_prior', action='store_true', help='Trains the diffusion prior to the pipeline')

    # Loss
    parser.add_argument('--loss_fn', type=str, choices=['clip', 'vicreg', 'softContrastive', 'softHybridContrastive'], default='clip', help='loss function, see loss.py for more info')
    
    # VICReg loss arguments
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    # "Contrastive head"
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # CLIP loss arguments
    parser.add_argument('--uniformity_loss_weight', type=float, default=0, help='Add terms to CLIP loss for uniformity and alignment')

    args = parser.parse_args()

    args.model_dir += args.experiment_name

    load_dotenv()

    os.environ["WANDB_API_KEY"] = "b0c5da2aac89929c85f768b56e5f260e287064ab"
    os.environ["WANDB_MODE"] = 'online'

    # Example usage
    set_seed(args.seed)

    # Define a mapping for the feature types
    feature_mapping = {
        'text': 'text_features',
        'image': 'img_features'
    }

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device(args.gpu)        
    else:
        device = torch.device('cpu')

    for sub in args.subjects:

        #aligner_model_path = f"{args.model_dir}/{args.atms_target}_{args.encoder_type}/{sub}/lr{args.lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}" if args.insubject else f"{args.model_dir}/across/{args.atms_target}_{args.encoder_type}/lr{args.lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}"
        aligner_model_path = f"{args.model_dir}/{args.atms_target}_{args.encoder_type}/{sub}/{args.name}" if args.insubject else f"{args.model_dir}/across/{args.atms_target}_{args.encoder_type}/{args.name}"
        #if args.alpha_scheduler:
        #    aligner_model_path += '_alpha_scheduler'
        os.makedirs(aligner_model_path, exist_ok=True)             
        aligner_model_path = f"{aligner_model_path}/best.pth"

        #aligner_embeddings_path = f"{args.embeddings_dir}/{args.atms_target}_{args.encoder_type}/{sub}/lr{args.lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}" if args.insubject else f"{args.embeddings_dir}/across/{args.atms_target}_{args.encoder_type}/lr{args.lr}_alpha{args.alpha}_epochs{args.epochs}_{args.diffusion_epochs}"
        aligner_embeddings_path = f"{args.embeddings_dir}/{args.atms_target}_{args.encoder_type}/{sub}/{args.name}" if args.insubject else f"{args.embeddings_dir}/across/{args.atms_target}_{args.encoder_type}/{args.name}"
        #if args.alpha_scheduler:
        #    aligner_embeddings_path += '_alpha_scheduler'
        os.makedirs(aligner_embeddings_path, exist_ok=True)       

        # init wandb logger
        logger = wandb_logger(args, sub) if args.logger else None

        # instantiate model
        eeg_model = ATMS(args = args) # globals()[args.encoder_type]()
        eeg_model.to(device)
        logger.watch(eeg_model,logger) 

        # Setup optimizer
        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)

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

        # -------TRAIN CLIP EMBEDDER------------------------------------------
        # train CLIP embedder, if the arg allows or if doesn't exist: 
        if not os.path.exists(aligner_model_path) or args.train_EEG_aligner:
        
            # init
            train_losses, train_accuracies = [], []
            test_losses, test_accuracies = [], []
            v2_accs, v4_accs, v10_accs = [], [], []

            # init
            best_accuracy = 0.0
            best_model_weights = None
            best_epoch_info = {}
            
            for epoch in tqdm(range(args.epochs), desc = "Epoch"):
                
                # Train one epoch
                train_loss, train_accuracy, _, average_MSE = train_clip_aligner(sub, eeg_model, clip_loaders['train'], optimizer, epoch, device, clip_dataset['train'].text_features, clip_dataset['train'].img_features, args=args)
                # save train losses
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                # Evaluate model
                epoch_results, _ = evaluate_and_append_results(sub, eeg_model, clip_loaders['test'], device, clip_dataset['test'], args, test_losses, test_accuracies, v2_accs, v4_accs, v10_accs, epoch, logger=logger)
                epoch_results['train_loss'], epoch_results['train_accuracy'], epoch_results['train_MSE'] = train_loss, train_accuracy, average_MSE
                logger.log(epoch_results)

                # If the test accuracy of the current epoch is the best, save the model and related information
                if epoch_results['test_accuracy'] > best_accuracy:
                    best_accuracy = epoch_results['test_accuracy']

                    # Save the model if better                  
                    torch.save(eeg_model.state_dict(), aligner_model_path)
                    print(f"Model saved in {aligner_model_path}!")

                    # For plot
                    best_epoch_info = epoch_results

                print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {epoch_results['train_loss']:.4f}, Train Accuracy: {epoch_results['train_accuracy']:.4f}")
                print(f"Epoch {epoch + 1}/{args.epochs} - Test Loss: {epoch_results['test_loss']:.4f}, Test Accuracy: {epoch_results['test_accuracy']:.4f}, Top5 Accuracy: {epoch_results['top5_acc']:.4f}")
                print(f"Epoch {epoch + 1}/{args.epochs} - v2 Accuracy:{epoch_results['v2_acc']} - v4 Accuracy:{epoch_results['v4_acc']} - v10 Accuracy:{epoch_results['v10_acc']} - v50 Accuracy:{epoch_results['v50_acc']} - v100 Accuracy:{epoch_results['v100_acc']}")


            # Plot curves
            plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, 
                    v2_accs, v4_accs, v10_accs, best_epoch_info, 'EEG_encoder_results')
            
        # -------END CLIP EMBEDDER------------------------------------------

        # Train and save diffusion prior
        if args.train_diffusion_prior:

            del clip_dataset

            eeg_model = load_and_freeze_best_model(eeg_model, args)
            diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)

            # TODO: OBS: USE logit_scale???

            with torch.no_grad():
                # Add rest of pipeline..!
                diffusion_dataset = {'train': 
                                    # Train was shown 4 times per image, test was 80 times -- this is the reason for the .repeat - however, something is still strange regardless
                                        EmbeddingDataset( 
                                            clip_eeg_embeddings = torch.cat([eeg_model(ele[0].unsqueeze(0).to(device), torch.tensor([extract_id_from_string(sub)], dtype=torch.long).to(device)) for ele in clip_loaders['train'].dataset], axis=0), 
                                            clip_embeddings = clip_loaders['train'].dataset.img_features.view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024) if args.diffusion_target == 'image' else clip_loaders['train'].dataset.text_features.view(1654,1,1,1024).repeat(1,10,4,1).view(-1,1024)
                                            ), # Corresponds to loading ViT-H-14
                                    'test': 
                                        EmbeddingDataset(
                                            clip_eeg_embeddings = torch.cat([eeg_model(ele[0].unsqueeze(0).to(device), torch.tensor([extract_id_from_string(sub)], dtype=torch.long).to(device)) for ele in clip_loaders['test'].dataset], axis=0), 
                                            clip_embeddings = clip_loaders['test'].dataset.img_features if args.diffusion_target == 'image' else clip_loaders['test'].dataset.text_features # TODO: WHY ONLY 20??
                                        ), 
                                    }
                
            print(f"Size of train EEG: {diffusion_dataset['train'].clip_eeg_embeddings.shape}")
            print(f"Size of train CLIP: {diffusion_dataset['train'].clip_embeddings.shape}")
                
            print(f"Size of train EEG: {diffusion_dataset['test'].clip_eeg_embeddings.shape}")
            print(f"Size of train CLIP: {diffusion_dataset['test'].clip_embeddings.shape}")


            del clip_loaders, eeg_model

            diffusion_loaders = { 'train': DataLoader(diffusion_dataset['train'], batch_size=1024, shuffle=True, num_workers=0) ,
                                  'test':  DataLoader(diffusion_dataset['test'],  batch_size=1024, shuffle=False, num_workers=0)}
            
            del diffusion_dataset

            train_diffusion_prior(sub, diffusion_prior, diffusion_loaders['train'], device, args, logger=logger)

        logger.finish()

