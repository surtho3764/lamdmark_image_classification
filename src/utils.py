import yaml
import os


def get_config_info(config_path):
    with open(config_path,'r') as stream:
        return yaml.load(stream)


def is_save_model_folder_exits(model_save_dir):
    return os.path.exists(model_save_dir)



def check_save_model_folde_exit(model_save_dir):
    if is_save_model_folder_exits(model_save_dir):
        print("{} dir exit".format(model_save_dir))
    else:
        print("{} dir no exit".format(model_save_dir))
        print("Create {} dir".format(model_save_dir))
        os.makedirs(model_save_dir)



def is_context_fold_exits(context_dir):
    return os.path.exists(context_dir)



def check_context_fold_exits(context_dir):
    if is_context_fold_exits(context_dir):
        print("{} dir exit.".format(context_dir))
    else:
        print("{} dir no exit".format(context_dir))
        print("Create {} dir".format(context_dir))
        os.makedirs(context_dir)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True




