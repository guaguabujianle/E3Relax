import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import time
import torch
import torch.optim as optim
from E3Relax import E3Relax
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from torch.utils.data import DataLoader
from graph_utils import vector_norm
from collections import defaultdict
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from ema import EMAHelper
from utils import *
import warnings
warnings.filterwarnings("ignore")

# %%
def val(model, dataloader, device):
    model.eval()

    running_loss = AverageMeter()
    running_loss_pos = AverageMeter()
    running_loss_cell = AverageMeter()

    pred_quantity_dict = defaultdict(list)

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pos_pred_list, cell_pred_list = model(data)
            pos_label = data.pos_r
            cell_label = data.cell_r
            
            loss_pos = 0
            loss_cell = 0
            for l, (pos_pred, cell_pred) in enumerate(zip(pos_pred_list, cell_pred_list)):
                loss_pos += vector_norm(pos_pred - pos_label, dim=-1).mean()
                loss_cell += (cell_pred - cell_label).abs().mean()
            loss = loss_pos + loss_cell

            pred_quantity_dict['pos_label'].append(pos_label)
            pred_quantity_dict['pos_pred'].append(pos_pred_list[-1])
            pred_quantity_dict['cell_label'].append(cell_label)
            pred_quantity_dict['cell_pred'].append(cell_pred_list[-1])

            running_loss.update(loss.item()) 
            running_loss_pos.update(loss_pos.item()) 
            running_loss_cell.update(loss_cell.item())

    pos_label = torch.cat(pred_quantity_dict['pos_label'], dim=0)
    pos_pred = torch.cat(pred_quantity_dict['pos_pred'], dim=0)
    cell_label = torch.cat(pred_quantity_dict['cell_label'], dim=0)
    cell_pred = torch.cat(pred_quantity_dict['cell_pred'], dim=0)

    valid_pos_mae = (pos_label - pos_pred).abs().mean().item()
    valid_cell_mae = (cell_label - cell_pred).abs().mean().item()
    
    valid_loss = running_loss.get_average()
    valid_loss_pos = running_loss_pos.get_average()
    valid_loss_cell = running_loss_cell.get_average()

    model.train()

    return valid_loss, valid_loss_pos, valid_loss_cell, valid_pos_mae, valid_cell_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_norm', type=int, default=150, help='max_norm for clip_grad_norm')
    parser.add_argument('--epochs', type=int, default=800, help='epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=800, help='steps_per_epoch')
    parser.add_argument('--early_stop_epoch', type=int, default=15, help='steps_per_epoch')
    parser.add_argument('--save_model', action='store_true', help='Save the model after training')
    parser.add_argument('--transfer', action='store_true')

    args = parser.parse_args()
    data_root = args.data_root
    num_workers = args.num_workers
    batch_size = args.batch_size
    max_norm = args.max_norm
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    early_stop_epoch = args.early_stop_epoch
    save_model = args.save_model
    transfer = args.transfer 

    train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train_E3Relax')})
    valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'val_E3Relax')})

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    dataset = data_root.split('/')[-1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f'E3Relax_{dataset}_{timestamp}'
    wandb.init(project="E3Relax", 
            group=f"{dataset}",
            config={"train_len" : len(train_set), "valid_len" : len(valid_set)}, 
            name=log_name,
            id=log_name
            )

    device = torch.device('cuda:0')
    model = E3Relax(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)
    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    
    if transfer == True:
        print("Loading pretrained model")

        model_path = './trained_model/model.pt'
        ema_helper.load_state_dict(torch.load(model_path))
        ema_helper.ema(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 5, min_lr = 1.e-8)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)

    running_loss = AverageMeter()
    running_loss_pos = AverageMeter()
    running_loss_cell = AverageMeter()
    running_grad_norm = AverageMeter()
    running_best_loss = BestMeter("min")
    running_best_mae = BestMeter("min")

    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    global_step = 0
    global_epoch = 0

    break_flag = False
    model.train()

    for epoch in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1      

            data = data.to(device)
            pos_pred_list, cell_pred_list = model(data)
            pos_label = data.pos_r
            cell_label = data.cell_r
            
            loss_pos = 0
            loss_cell = 0
            for l, (pos_pred, cell_pred) in enumerate(zip(pos_pred_list, cell_pred_list)):
                loss_pos += vector_norm(pos_pred - pos_label, dim=-1).mean()
                loss_cell += (cell_pred - cell_label).abs().mean()
            loss = loss_pos + loss_cell
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
            )
            optimizer.step()
            ema_helper.update(model)

            running_loss.update(loss.item()) 
            running_loss_pos.update(loss_pos.item()) 
            running_loss_cell.update(loss_cell.item())
            running_grad_norm.update(grad_norm.item())

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                train_loss = running_loss.get_average()
                train_loss_pos = running_loss_pos.get_average()
                train_loss_cell = running_loss_cell.get_average()
                train_grad_norm = running_grad_norm.get_average()

                running_loss.reset()
                running_loss_pos.reset()
                running_loss_cell.reset()
                running_grad_norm.reset()

                valid_loss, valid_loss_pos, valid_loss_cell, valid_pos_mae, valid_cell_mae = val(ema_helper.ema_copy(model), valid_loader, device)

                scheduler.step(valid_pos_mae)

                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/epoch' : global_epoch,
                    'train/loss' : train_loss,
                    'train/loss_pos' : train_loss_pos,
                    'train/loss_cell' : train_loss_cell,
                    'train/grad_norm' : train_grad_norm,
                    'train/lr' : current_lr,
                    'val/valid_loss' : valid_loss,
                    'val/valid_loss_pos' : valid_loss_pos,
                    'val/valid_loss_cell' : valid_loss_cell,
                    'val/valid_pos_mae' : valid_pos_mae,
                    'val/valid_cell_mae' : valid_cell_mae,
                }
                wandb.log(log_dict)

                if valid_pos_mae < running_best_mae.get_best():
                    running_best_mae.update(valid_pos_mae)
                    if save_model:
                        torch.save(ema_helper.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
                else:
                    count = running_best_mae.counter()
                    if count > early_stop_epoch:
                        best_mae = running_best_mae.get_best()
                        print(f"early stop in epoch {global_epoch}")
                        print("best_mae: ", best_mae)
                        break_flag = True
                        break

    wandb.finish()