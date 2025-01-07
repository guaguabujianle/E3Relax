# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import torch
import pandas as pd
from E3Relax_oc22 import E3Relax
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from torch.utils.data import DataLoader
from ema import EMAHelper
from collections import defaultdict
from pymatgen.analysis.structure_matcher import StructureMatcher
import argparse
from utils import create_dir
import warnings
warnings.filterwarnings("ignore")

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--val_type', type=str, default="val_id", help='val_id or val_ood', required=True)
    parser.add_argument('--model_path', type=str, default=None, help='model path', required=True)

    args = parser.parse_args()
    data_root = args.data_root
    val_type = args.val_type
    model_path = args.model_path 

    dataset = data_root.split('/')[-1]

    if val_type == 'val_id':
        test_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'val_id_E3Relax')})
    elif val_type == 'val_ood':
        test_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'val_ood_E3Relax')})

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = torch.device('cuda:0')
    model = E3Relax(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    model = model.to(device)
    model.eval()

    performance_dict = defaultdict(list)
    matcher = StructureMatcher() # Initialize a StructureMatcher object
    for data in test_loader:
        with torch.no_grad():
            data = data.to(device)

            start_time = time.time() # Record the starting time

            pos_pred_list = model(data)

            end_time = time.time()
            elapsed_time = end_time - start_time # Calculate the time elapsed

            pos_u = data.pos
            pos_r = data.pos_relaxed

            mae_pos_Dummy = (pos_u - pos_r).cpu().abs().mean().item()
            mae_pos_E3Relax = (pos_r - pos_pred_list[-1]).cpu().abs().mean().item()

            mae_pos_SA_Dummy = (pos_u[data.tags != 0] - pos_r[data.tags != 0]).cpu().abs().mean().item()
            mae_pos_SA_E3Relax = (pos_r[data.tags != 0] - pos_pred_list[-1][data.tags != 0]).cpu().abs().mean().item() 

            performance_dict['sid'].append(data.sid[0].cpu().item())
            performance_dict['mae_pos_Dummy'].append(mae_pos_Dummy)
            performance_dict['mae_pos_E3Relax'].append(mae_pos_E3Relax)
            performance_dict['mae_pos_SA_Dummy'].append(mae_pos_SA_Dummy)
            performance_dict['mae_pos_SA_E3Relax'].append(mae_pos_SA_E3Relax)
            performance_dict['elapsed_time'].append(elapsed_time)
 
    create_dir(['./results'])
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(f"./results/{dataset}_{val_type}.csv", index=False)


# %%