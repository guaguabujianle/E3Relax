# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import torch
import pandas as pd
from E3Relax import E3Relax
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
    parser.add_argument('--model_path', type=str, default=None, help='model path', required=True)

    args = parser.parse_args()
    data_root = args.data_root
    model_path = args.model_path 

    dataset = data_root.split('/')[-1]

    test_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'test_E3Relax')})
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

            pos_pred_list, cell_pred_list = model(data)

            end_time = time.time()
            elapsed_time = end_time - start_time # Calculate the time elapsed

            pos_r = data.pos_r
            pos_u = data.pos_u
            cell_u = data.cell_u
            cell_r = data.cell_r

            mae_pos_Dummy = (pos_u - pos_r).cpu().abs().mean().item()
            mae_pos_E3Relax = (pos_r - pos_pred_list[-1]).cpu().abs().mean().item()

            mae_cell_Dummy = (cell_u - cell_r).cpu().abs().mean().item()
            mae_cell_E3Relax = (cell_r - cell_pred_list[-1]).cpu().abs().mean().item()

            metric_tensor_unrelaxed = cell_u.squeeze() @ cell_u.squeeze().T
            metric_tensor_relaxed = cell_r.squeeze() @ cell_r.squeeze().T
            metric_tensor_predicted= cell_pred_list[-1].squeeze() @ cell_pred_list[-1].squeeze().T
            mae_metric_tensor_Dummy = torch.norm(metric_tensor_unrelaxed - metric_tensor_relaxed, p='fro').item()
            mae_metric_tensor_E3Relax = torch.norm(metric_tensor_predicted - metric_tensor_relaxed, p='fro').item()

            mae_volume_Dummy = (torch.linalg.det(cell_u) - torch.linalg.det(cell_r)).abs().item()
            mae_volume_E3Relax = (torch.linalg.det(cell_pred_list[-1]) - torch.linalg.det(cell_r)).abs().item()

            performance_dict['cif_id'].append(data.cif_id[0])
            performance_dict['mae_pos_Dummy'].append(mae_pos_Dummy)
            performance_dict['mae_pos_E3Relax'].append(mae_pos_E3Relax)
            performance_dict['mae_metric_tensor_Dummy'].append(mae_metric_tensor_Dummy)
            performance_dict['mae_metric_tensor_E3Relax'].append(mae_metric_tensor_E3Relax)
            performance_dict['mae_volume_Dummy'].append(mae_volume_Dummy)
            performance_dict['mae_volume_E3Relax'].append(mae_volume_E3Relax)
            performance_dict['elapsed_time'].append(elapsed_time)
 
    create_dir(['./results'])
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(f"./results/{dataset}.csv", index=False)


# %%