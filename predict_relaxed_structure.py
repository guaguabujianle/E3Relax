# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import torch
import pandas as pd
from E3Relax import E3Relax
from ase.io import read, write
import argparse
from ema import EMAHelper
from collections import defaultdict
from graph_constructor import AtomsToGraphs
from torch_geometric.data import Batch
from utils import *
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

    test_df = pd.read_csv(os.path.join(data_root, "test.csv"))

    device = torch.device('cuda:0')
    model = E3Relax(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    model = model.to(device)
    model.eval()
    
    a2g = AtomsToGraphs(
        radius=6,
        max_neigh=50
    )

    create_dir(['./predicted_structures'])

    performance_dict = defaultdict(list)
    for i, row in test_df.iterrows():
        atoms_id = row['atoms_id']
        cif_path = os.path.join( data_root, 'CIF', atoms_id + '_unrelaxed.cif')
        atoms_u = read(cif_path)
        data = a2g.convert_single(atoms_u)
        data = Batch.from_data_list([data])
        data = data.to(device)

        # Record the starting time
        start_time = time.time()
        with torch.no_grad():
            pos_pred_list, cell_pred_list = model(data)

        end_time = time.time()
        # Calculate the time elapsed
        elapsed_time = end_time - start_time

        atoms_u.set_positions(pos_pred_list[-1].cpu().numpy())
        atoms_u.set_cell(cell_pred_list[-1].squeeze(0).cpu().numpy())

        predicted_path = os.path.join('./predicted_structures', atoms_id + '_predicted.cif')

        write(predicted_path, atoms_u)

# %%