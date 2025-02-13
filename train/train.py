import os
import yaml
import time
import torch
import argparse
from torch.utils.data import DataLoader
from disaja import TTbarDileptonDataset
from disaja import TTbarDileptonSAJA
from disaja import object_wise_cross_entropy
from disaja import MinMaxScaler


# Function to load a model checkpoint
def load_checkpoint(model, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        lr_lst = checkpoint['lr']
        t_loss = checkpoint['t_loss']
        v_loss = checkpoint['v_loss']
        if 'scheduler' in checkpoint.keys():
            scheduler = checkpoint['scheduler']
        else:
            scheduler = None

        print(f"=> loaded checkpoint '{filename}' (epoch {start_epoch})")
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model, optimizer, scheduler, start_epoch, lr_lst, t_loss, v_loss

# Function to train the model
def train(model, train_loader, opt, device):
    torch.set_grad_enabled(True)
    model.train()
    train_loss = 0
    for batch in train_loader:
        opt.zero_grad()
        batch = batch.to(device)
        logits = model(batch)
        loss = object_wise_cross_entropy(logits,
                                                batch.target,
                                                torch.logical_not(
                                                    batch.jet_data_mask
                                                    ),
                                                batch.jet_lengths,
                                         )
        loss.backward()
        opt.step()
        train_loss += loss.item() * len(batch.target)
    return train_loss

# Function to validate the model
def validation(model, valid_loader, scheduler, device):
    torch.set_grad_enabled(False)
    model.eval()
    valid_loss = 0
    v_ncorrect = 0
    ntot = 0
    for batch in valid_loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = object_wise_cross_entropy(logits,
                                                batch.target,
                                                torch.logical_not(
                                                    batch.jet_data_mask
                                                    ),
                                                batch.jet_lengths,
                                                reduction='none',
                                         ).sum()
        valid_loss += loss
        assign = logits.argmax(dim=2)
        target = batch.target
        v_ncorrect += (((assign==target)).all(dim=1)).sum().item()
        ntot += len(batch.target.squeeze())

    if scheduler is not None:
        scheduler.step()
    return valid_loss.item(), v_ncorrect, ntot


# Main function to execute training and validation
def main():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='devce_num (default: 0)')
    parser.add_argument('-o', '--output_path', type=str, default='model',
                        help='output_path (default="model")')
    args = parser.parse_args()

    # Load configuration file
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    branches = config['branches']
    hyper_params = config['hyper_params']

    save_path = f"{args.output_path}" 

    # Create output directory
    if os.path.isdir(save_path):
        raise Exception(f"{save_path} exists!")
    else:
        os.makedirs(save_path)
    copy_config = f"cp config.yaml train.py {save_path}/"
    print(">>>", copy_config)
    os.system(copy_config)

    # Allocate memory on GPU
    device_num = args.device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_num}"

    device = torch.device("cuda" if (torch.cuda.is_available()) else False)

    # Initialize model
    model = TTbarDileptonSAJA(dim_jet=len(branches['jet_branches']),
                              dim_lepton=len(branches['lep_branches']),
                              dim_met=len(branches['met_branches']),
                              dim_output=3,  # other, t, tbar
                              dim_ffnn=hyper_params['dim_ffnn'],
                              num_blocks=hyper_params['num_blocks'],
                              num_heads=hyper_params['num_heads'],
                              depth=hyper_params['depth'],
                              dropout_rate=hyper_params['dropout_rate'],
            ).to(device)

    # Load existing model checkpoint if available
    if os.path.isfile(f"{save_path}/last_model.pt"):
        model, opt, scheduler, start_epoch, lr_lst, t_losses, v_losses = load_checkpoint(model, f"{save_path}/last_model.pt")
    else:
        opt = torch.optim.AdamW(model.parameters(),
                           lr=hyper_params['learning_rate'])
        start_epoch = 0
        lr_lst = []
        t_losses = []
        v_losses = []
    scheduler = None

    ### Dataload & Scale ###
    base_path = "DATA_PATH"
    train_bb_sample = f"{base_path}/train_bb.root"
    train_bs_sample = f"{base_path}/train_bs.root"
    valid_bb_sample = f"{base_path}/valid_bb.root"
    valid_bs_sample = f"{base_path}/valid_bs.root"
    tree_path = 'delphes'
    unmatched_tree_path = 'unmatched'

    # data load & preprocessing
    matched_bb_train_dataset = TTbarDileptonDataset(train_bb_sample,
                                                    tree_path,
                                                    **branches)
    unmatched_bb_train_dataset = TTbarDileptonDataset(train_bb_sample, unmatched_tree_path, **branches)
    matched_bs_train_dataset = TTbarDileptonDataset(train_bs_sample,
                                                    tree_path,
                                                    **branches)
    matched_bb_valid_dataset = TTbarDileptonDataset(valid_bb_sample,
                                                    tree_path,
                                                    **branches)
    unmatched_bb_valid_dataset = TTbarDileptonDataset(valid_bb_sample, unmatched_tree_path, **branches)
    matched_bs_valid_dataset = TTbarDileptonDataset(valid_bs_sample,
                                                    tree_path,
                                                    **branches)

    train_dataset = matched_bb_train_dataset\
            + matched_bs_train_dataset\
            + unmatched_bb_train_dataset


    valid_dataset = matched_bb_valid_dataset\
            + matched_bs_valid_dataset\
            + unmatched_bb_valid_dataset

    scaler_path = f'{save_path}/scaler.pt'
    if os.path.isfile(scaler_path):
        print('>>> using fitted scaler')
        scaler = torch.load(scaler_path)
    else:
        print('>>> fitting scaler')
        scaler = MinMaxScaler(
                branches['jet_branches'],
                branches['lep_branches'],
                branches['met_branches'],
                )
        scaler.fit(train_dataset)
        torch.save(scaler, f'{save_path}/scaler.pt')

    scaler.transform(train_dataset)
    scaler.transform(valid_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=hyper_params['batch_size'],
                              collate_fn=matched_bs_train_dataset.collate,
                              **config['loader_args'],
                              )

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1024,
                              collate_fn=matched_bs_valid_dataset.collate,
                              )

    v_acc = []
    v_nCorrect = []
    if len(v_losses)==0:
        min_loss = 1e+10
    else:
        min_loss = min(v_losses)

    # Training and validation loop
    print(">>> training start")
    for epoch in range(start_epoch, start_epoch + config['n_epoch']):
        start = time.time()

        # training
        train_loss = train(model, train_loader, opt, device)
        t_losses.append(train_loss / len(train_dataset))

        # validation
        valid_loss, v_ncorrect, ntot = validation(model,
                                                  valid_loader,
                                                  scheduler,
                                                  device)

        v_losses.append(valid_loss / len(valid_dataset))
        v_acc.append(v_ncorrect / ntot)
        v_nCorrect.append(v_ncorrect)
        lr_lst.append(opt.param_groups[0]['lr'])

        save_dic = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': opt,
            'scheduler': scheduler,
            'lr': lr_lst,
            't_loss': t_losses,
            'v_loss': v_losses,
            'v_nCorrect': v_nCorrect,
            'v_acc': v_acc,
            }
        end = time.time()

        # save model
        status = f'Epoch {epoch + 1}/{config["n_epoch"]} \t' \
                + ' Train loss = {:.4f}\t'.format(t_losses[-1]) \
                + ' Valid loss = {:.4f}\t'.format(v_losses[-1]) \
                + f' time : {(end-start)//60}min'

        torch.save(save_dic, f'{save_path}/last_model.pt')

        if v_losses[-1] < min_loss:
            min_loss = v_losses[-1]
            torch.save(save_dic, f'{save_path}/best_model.pt')
            status += " >>> saved"
        print(status)


if __name__ == '__main__':
    main()
