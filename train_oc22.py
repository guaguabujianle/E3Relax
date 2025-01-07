# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import time
import torch
import torch.optim as optim
from E3Relax_oc22 import E3Relax
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
    pred_quantity_dict = defaultdict(list)
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pos_pred_list = model(data)
            pos_label = data.pos_relaxed
   
            loss = 0
            for pos_pred in pos_pred_list:
                loss += vector_norm(pos_pred - pos_label, dim=-1).mean()
   
            pred_quantity_dict['pos_label'].append(pos_label)
            pred_quantity_dict['pos_pred'].append(pos_pred_list[-1])

            running_loss.update(loss.item()) 

    pos_label = torch.cat(pred_quantity_dict['pos_label'], dim=0)
    pos_pred = torch.cat(pred_quantity_dict['pos_pred'], dim=0)

    valid_pos_mae = (pos_label - pos_pred).abs().mean().item()
    valid_loss = running_loss.get_average()

    model.train()

    return valid_loss, valid_pos_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--val_type', type=str, default="val_id", help='val_id or val_ood', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_norm', type=int, default=150, help='max_norm for clip_grad_norm')
    parser.add_argument('--epochs', type=int, default=800, help='epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=1600, help='steps_per_epoch')
    parser.add_argument('--early_stop_epoch', type=int, default=15, help='steps_per_epoch')
    parser.add_argument('--save_model', action='store_true', help='Save the model after training')

    args = parser.parse_args()
    data_root = args.data_root
    val_type = args.val_type
    num_workers = args.num_workers
    batch_size = args.batch_size
    max_norm = args.max_norm
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    early_stop_epoch = args.early_stop_epoch
    save_model = args.save_model

    train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train_E3Relax')})
    valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, f'{val_type}_E3Relax')})

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    dataset = data_root.split('/')[-1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f'E3Relax_{dataset}_{val_type}_{timestamp}'
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
            pos_pred_list = model(data)
            pos_label = data.pos_relaxed

            loss = 0
            for pos_pred in pos_pred_list:
                loss += vector_norm(pos_pred - pos_label, dim=-1).mean()
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=20,
            )
            optimizer.step()
            ema_helper.update(model)

            running_loss.update(loss.item()) 
            running_grad_norm.update(grad_norm.item())

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                train_loss = running_loss.get_average()
                train_grad_norm = running_grad_norm.get_average()

                running_loss.reset()
                running_grad_norm.reset()

                valid_loss, valid_pos_mae = val(ema_helper.ema_copy(model), valid_loader, device)

                scheduler.step(valid_pos_mae)

                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/epoch' : global_epoch,
                    'train/loss' : train_loss,
                    'train/grad_norm' : train_grad_norm,
                    'train/lr' : current_lr,
                    'val/valid_loss' : valid_loss,
                    'val/valid_pos_mae' : valid_pos_mae,
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