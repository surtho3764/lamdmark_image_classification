import os
import sys
import time
import ssl
import numpy as np
import torch
import amp
import torch.optim
import torch.utils.data
from src.utils import get_config_info,check_save_model_folde_exit,check_context_fold_exits
from src.preporcess import get_df
from src.dataset import get_transforms,Dataset_GLDv2
from src.loss_optimier import criterion_cla,optimizer_fun
from lib.pytorch_lib.warmup_scheduler.scheduler import GradualWarmupScheduler
from src.trin_fun import train_epoch_fun
from src.valid_fun import val_epoch_fun
from src.model import Effnet_GLDv2,ResNet101_GLDv2
from src.utils import set_seed


# Addition my lib path
lib_path =[os.getcwd(),os.path.join(os.getcwd(),'..')]
sys.path.append(lib_path)

# config
config_path = os.path.join(os.getcwd(),'..','config','config.yaml')
config = get_config_info(config_path)

# init
# data dir config
data_dir = config['data_dir']
train_file_name = config['train_file_name']

# context record dir
context_dir = config['context_dir']
check_context_fold_exits(context_dir)

# choose model
model_type = config['model_type']

# pre_trained_model_name
pre_trained_model_name = config[model_type]['pre_trained_name']

# save training model dir
model_save_dir = config[model_type]["save_model_dir"]
model_save_name = config[model_type]['save_model_name']
check_save_model_folde_exit(model_save_dir)

# save finished model
save_finish_model_name = config[model_type]["save_finish_model_name"]

# input image size
IMAGE_SIZE = config[model_type]['image_size'][config["image_size_type"]]

# n_epochs
N_EPOCHS = config[model_type]["epochs"][config["n_epochs_type"]]



# training init
K_FOLD = config['k_fold']
START_FROM_EPOCH = config["start_from_epoch"]
BATCH_SIZE = config['batch_size']
NUM_WORKERS = config['num_workers']
LR_INIT = config['lr_int']
USE_AMP = config['use_amp']
IS_GPU = config['is_gpu']
GET_VAL_OUTPUT = config["get_val_output"]
EXIT_PRETRAINED_MODEL_WEIGHT = config['exit_pretrained_model_weight']


torch.cuda.empty_cache()
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    df, out_dim = get_df(data_dir, train_file_name)

    tmp = np.sqrt(1 / np.sqrt(df['landmark_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    transforms_train, transforms_val = get_transforms(IMAGE_SIZE)
    df_train = df[df['fold'] != K_FOLD]
    df_valid = df[df['fold'] == K_FOLD]
    dataset_train = Dataset_GLDv2(df_train, 'train', transform=transforms_train)
    dataset_valid = Dataset_GLDv2(df_valid, 'val', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # model
    model = ModelClass(pre_trained_model_name, out_dim=out_dim)
    if IS_GPU:
        model = model.cuda()

    criterion = criterion_cla(margin=margins, s=80, out_dim=out_dim)
    optimizer = optimizer_fun(model.parameters(), lr=LR_INIT)

    if USE_AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, N_EPOCHS - 1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    model_file = os.path.join(model_save_dir, model_save_name)
    model_finish_file = os.path.join(model_save_dir, model_finish_save_name)

    if EXIT_PRETRAINED_MODEL_WEIGHT:
        if os.path.isfile(model_file):
            checkpoint = torch.load(model_finish_file, map_location=device)
            # load exits model weights file
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict, strict=True)
            del checkpoint
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        else:
            raise Exception("{} not exits.".format(model_file))

    # train & valid loop
    gap_m_max = 0.

    for epoch in range(START_FROM_EPOCH, N_EPOCHS + 1):
        print(time.ctime(), 'Epoch:', epoch)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        scheduler_warmup.step(epoch - 1)

        # run train epochs
        train_loss = train_epoch_fun(model, train_loader, optimizer, criterion, device, USE_AMP)
        print("epoch_train_loss:", train_loss)

        # run validation epochs
        val_loss, acc_m, gap_m = val_epoch_fun(model, valid_loader, criterion, device, USE_AMP, GET_VAL_OUTPUT=False)
        print("epoch:", val_loss)
        print("epoch acc_m:", acc_m)
        print("epoch gap_m", gap_m)

        # save training records content
        content = time.ctime() + ' ' + f'Epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
        print(content)
        with open(os.path.join(context_dir, 'context.txt'), 'a') as appender:
            appender.write(content)

        print('gap_m_max:{} -->{}.'.format(gap_m_max, gap_m))

        print("Save model ...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_file)

        gap_m_max = gap_m

    # Loop finish save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_finish_file)


if __name__ == '__main__':
    if model_type == 'EfficientNet_B7':
        ModelClass = Effnet_GLDv2
    elif model_type == 'ResNeSt101':
        ModelClass = ResNet101_GLDv2

    set_seed(0)
    main()

