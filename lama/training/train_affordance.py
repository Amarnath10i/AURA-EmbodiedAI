"""
Train the Affordance Predictor Head on pseudo-labels discovered by the clustering module.
"""
import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.models.encoder import VisionRobotEncoder
from lama.affordance.discovery import AffordanceDiscovery
from lama.affordance.predictor import AffordancePredictor

LATENT_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_affordance():
    # ---- Load trained encoder ----
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lama_world_model.pth')
    robot_state_dim = 14
    
    encoder = VisionRobotEncoder(robot_state_dim=robot_state_dim, final_latent_dim=LATENT_DIM).to(DEVICE)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        print("Loaded trained encoder.")
    else:
        print("WARNING: No trained encoder found. Using random weights.")

    # ---- Load episodes and encode ----
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
    ep_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode*")))
    
    if not ep_dirs:
        print(f"ERROR: No episodes found in {dataset_dir}")
        return
    
    discovery = AffordanceDiscovery(method="kmeans", n_clusters=6)
    
    print("Encoding episodes and collecting transitions...")
    encoder.eval()
    with torch.no_grad():
        for ep_dir in tqdm(ep_dirs, desc="Encoding"):
            rgb = np.load(os.path.join(ep_dir, "rgb.npy"))
            state = np.load(os.path.join(ep_dir, "state.npy"))
            action = np.load(os.path.join(ep_dir, "action.npy"))
            
            rgb_t = torch.FloatTensor(rgb).permute(0, 3, 1, 2).to(DEVICE) / 255.0
            state_t = torch.FloatTensor(state).to(DEVICE)
            
            z_all = encoder(rgb_t, state_t).cpu().numpy()  # (T, LATENT_DIM)
            
            # Add transitions (Δz = z_{t+1} - z_t)
            for t in range(len(action)):
                if t < len(z_all) - 1:
                    action_id = int(np.argmax(np.abs(action[t])))  # Rough primitive ID
                    discovery.add_transition(z_all[t], action_id, z_all[t+1])
    
    # ---- Discover Affordances ----
    affordances = discovery.discover(min_transitions=10)
    if not affordances:
        print("No affordances discovered. Collect more diverse data.")
        return
    
    pseudo_labels = discovery.get_pseudo_labels()
    n_affordances = len(affordances)
    
    # ---- Train Predictor ----
    predictor = AffordancePredictor(latent_dim=LATENT_DIM, n_affordances=n_affordances).to(DEVICE)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    
    z_data = torch.FloatTensor(np.array(discovery.z_t_buffer)).to(DEVICE)
    label_data = torch.LongTensor(pseudo_labels).to(DEVICE)
    
    # Filter out noise labels (-1 from DBSCAN)
    valid_mask = label_data >= 0
    z_data = z_data[valid_mask]
    label_data = label_data[valid_mask]
    
    print(f"\nTraining Affordance Predictor on {len(z_data)} samples, {n_affordances} classes...")
    
    predictor.train()
    for epoch in range(30):
        # Shuffle
        perm = torch.randperm(len(z_data))
        z_shuffled = z_data[perm]
        l_shuffled = label_data[perm]
        
        total_loss = 0.0
        correct = 0
        
        bs = 64
        for i in range(0, len(z_data), bs):
            z_batch = z_shuffled[i:i+bs]
            l_batch = l_shuffled[i:i+bs]
            
            logits = predictor(z_batch)
            loss = ce_loss(logits, l_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == l_batch).sum().item()
        
        acc = correct / len(z_data) * 100
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/30  |  Loss: {total_loss:.4f}  |  Acc: {acc:.1f}%")
    
    # ---- Save ----
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "predictor": predictor.state_dict(),
        "n_affordances": n_affordances,
        "affordance_names": [a.label for a in affordances],
    }, os.path.join(save_dir, "affordance_predictor.pth"))
    print(f"Saved affordance predictor ({n_affordances} affordances).")


if __name__ == "__main__":
    train_affordance()
