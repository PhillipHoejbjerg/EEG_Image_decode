
import torch
import argparse
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
import itertools

from tqdm import tqdm

# Modules
from util import wandb_logger
from utils_phil import extract_id_from_string, set_seed
from eegdatasets_leaveone import EEGDataset


from models import ATMS


def load_CLIP_loaders(args, sub, device):
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
        
        optimizer.zero_grad()
        
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

def load_best_model(eeg_model, args):

    # Getting output from the best model
    PATH = f"{args.model_dir}/{sub}/{args.name}" if args.insubject else f"{args.model_dir}/across/{args.name}"
    eeg_model.load_state_dict(torch.load(f"{PATH}/best.pth", weights_only=False, map_location=torch.device(device)))
    # Freezing the original embedder
    if args.freeze_ATMS:
        # Freeze the parameters of the original model
        for param in eeg_model.parameters():
            param.requires_grad = False
        eeg_model.eval()

    return eeg_model

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, time_embed_dim=64):
        super().__init__()

        # Time embedding: you can use sinusoidal encoding or a learnable MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Output: velocity vector
        )

    def forward(self, x, t):
        """
        x: tensor of shape (batch_size, input_dim) → current x(t)
        t: tensor of shape (batch_size, 1) → time step
        """
        t_embed = self.time_mlp(t)                  # (B, time_embed_dim)
        xt = torch.cat([x, t_embed], dim=-1)        # Concatenate time and input
        return self.net(xt) 

from torch.utils.data import Dataset
class EmbeddingDataset(Dataset):

    def __init__(self, clip_eeg_embeddings, clip_embeddings):
        self.clip_eeg_embeddings = clip_eeg_embeddings
        self.clip_embeddings = clip_embeddings

    def __len__(self):
        return len(self.clip_eeg_embeddings)

    def __getitem__(self, idx):
        return {
            "clip_eeg_embeddings": self.clip_eeg_embeddings[idx],
            "clip_embeddings": self.clip_embeddings[idx]
        } 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/work3/s184984/repos/EEG_Image_decode/eeg_dataset/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--model_dir', type=str, default='./models/EEG_encoder', help='Directory to save output results')    

    # Model params
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'mps'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')  
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--alpha', type=float, default=0.90, help='alpha value to weigh the loss')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs') 
    parser.add_argument('--flow_epochs', type=int, default=100, help='Number of epochs') 


    # Model targets
    parser.add_argument('--atms_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')
    parser.add_argument('--diffusion_target', type=str, choices=['image', 'text'], default='image', help='Encoder type')

    # Wandb
    parser.add_argument('--project', type=str, default="EEG_image_reconstruction", help='WandB project name')
    parser.add_argument('--name', type=str, default="modified_loss", help='Experiment name')
    parser.add_argument('--entity', type=str, default="philliphoejbjerg", help='WandB entity name')

    # Freeze ATMS
    parser.add_argument('--freeze_ATMS', action='store_true', help='Freeze ATMS model parameters')

    os.environ["WANDB_API_KEY"] = "b0c5da2aac89929c85f768b56e5f260e287064ab"
    os.environ["WANDB_MODE"] = 'online'


    args = parser.parse_args()

    set_seed(args.seed)


    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device(args.device)        
    else:
        device = torch.device('cpu')

    for sub in args.subjects:

        # Path for model
        aligner_model_path = f"{args.model_dir}/{sub}/{args.name}" if args.insubject else f"{args.model_dir}/across/{args.name}"
        os.makedirs(aligner_model_path, exist_ok=True)             

        # init wandb logger
        logger = wandb_logger(args, sub) if args.logger else None

        # instantiate ATMS model
        eeg_model = ATMS(args = args) # globals()[args.encoder_type]()
        eeg_model.to(device)
        logger.watch(eeg_model,logger) 

        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)

        clip_dataset, clip_loaders = load_CLIP_loaders(args, sub, device) # Load EEG dataset
        
        # Train and validation loop
        best_accuracy = 0.0
        best_epoch = 0
        for epoch in tqdm(range(args.epochs), desc = "Epoch"):
            
            # Train one epoch
            train_loss, train_accuracy, average_MSE = train_clip_aligner(sub, eeg_model, clip_loaders['train'], optimizer, epoch, device, clip_dataset['train'].text_features, clip_dataset['train'].img_features, args=args)

            # Evaluate model
            epoch_results = validate_clip_aligner(sub, eeg_model, clip_loaders['test'], device, clip_dataset['test'], args, epoch)
            epoch_results['train_loss'], epoch_results['train_accuracy'], epoch_results['train_MSE'] = train_loss, train_accuracy, average_MSE
            logger.log(epoch_results)

            # If the test accuracy of the current epoch is the best, save the model and related information
            if epoch_results['test_accuracy'] > best_accuracy:
                best_accuracy = epoch_results['test_accuracy']
                best_epoch = epoch + 1

                torch.save(eeg_model.state_dict(), aligner_model_path)
                print(f"Model saved in {aligner_model_path}!, Epoch: {best_epoch}, Accuracy: {best_accuracy:.4f}, MSE: {epoch_results['test_MSE']:.4f}, Loss: {epoch_results['test_loss']:.4f}")

            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {epoch_results['train_loss']:.4f}, Train Accuracy: {epoch_results['train_accuracy']:.4f}, Train MSE: {epoch_results['train_MSE']:.4f}")
            print(f"Epoch {epoch + 1}/{args.epochs} - Test Loss: {epoch_results['test_loss']:.4f}, Test Accuracy: {epoch_results['test_accuracy']:.4f}, Test MSE: {epoch_results['test_MSE']:.4f}")
        
        # FLOW MATCHING
            
        del clip_dataset

        # Loading best model (args decide whether to freeze model)
        eeg_model = load_best_model(eeg_model, args)

        # Fine-tune with Matching Flow
        flow = FlowMatchingMLP(input_dim=1024).to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)

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
                 
        diffusion_loaders = { 'train': DataLoader(diffusion_dataset['train'], batch_size=1024, shuffle=True, num_workers=0) ,
                                'test':  DataLoader(diffusion_dataset['test'],  batch_size=1024, shuffle=False, num_workers=0)}
        
        import torch
        import torch.nn.functional as F


        alpha = 1  # Balance factor between MSE and cosine similarity

        for epoch in range(args.flow_epochs):
            flow.train()
            train_mse_total = 0.0
            train_cos_total = 0.0
            train_batches = 0

            for batch in diffusion_loaders['train']:
                x0 = batch['clip_eeg_embeddings'].to(device)
                x1 = batch['clip_embeddings'].to(device)

                t = torch.rand((x0.size(0), 1), device=device)
                xt = (1 - t) * x0 + t * x1

                v_true = x1 - x0
                v_pred = flow(torch.cat([xt, t], dim=-1))

                mse_loss = F.mse_loss(v_pred, v_true)
                cosine_loss = 1 - F.cosine_similarity(v_pred, v_true, dim=-1).mean()
                loss = alpha * mse_loss + (1 - alpha) * cosine_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track training loss
                train_mse_total += mse_loss.item()
                train_cos_total += cosine_loss.item()
                train_batches += 1

            # -------------------
            # Validation
            # -------------------
            flow.eval()
            val_mse_total = 0.0
            val_cos_total = 0.0
            val_batches = 0

            best_val_loss = float('inf')

            with torch.no_grad():
                for batch in diffusion_loaders['test']:
                    x0 = batch['clip_eeg_embeddings'].to(device)
                    x1 = batch['clip_embeddings'].to(device)

                    t = torch.rand((x0.size(0), 1), device=device)
                    xt = (1 - t) * x0 + t * x1

                    v_true = x1 - x0
                    v_pred = flow(torch.cat([xt, t], dim=-1))

                    mse_loss = F.mse_loss(v_pred, v_true)
                    cosine_loss = 1 - F.cosine_similarity(v_pred, v_true, dim=-1).mean()

                    val_mse_total += mse_loss.item()
                    val_cos_total += cosine_loss.item()
                    val_batches += 1

                val_loss = alpha * (val_mse_total / val_batches) + (1 - alpha) * (val_cos_total / val_batches)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(flow.state_dict(), f"{args.model_dir}/{sub}/{args.name}" + "/best_flow.pth")
                print(f"✅ Saved new best model at epoch {epoch} with val_loss = {val_loss:.4f}")

            # Log results to WandB
            epoch_results = {
                "flow_epoch": epoch,
                "flow_train/mse_loss": train_mse_total / train_batches,
                "flow_train/cosine_loss": train_cos_total / train_batches,
                "flow_val/mse_loss": val_mse_total / val_batches,
                "flow_val/cosine_loss": val_cos_total / val_batches
            }

            logger.log(epoch_results)
            print(f"Epoch {epoch + 1}/{args.flow_epochs} - Train MSE Loss: {train_mse_total / train_batches:.4f}, Train Cosine Loss: {train_cos_total / train_batches:.4f}")
            print(f"Epoch {epoch + 1}/{args.flow_epochs} - Val MSE Loss: {val_mse_total / val_batches:.4f}, Val Cosine Loss: {val_cos_total / val_batches:.4f}")