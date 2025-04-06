import os
import torch.backends.cudnn as cudnn
import yaml

from utils import AttrDict
import pandas as pd
from model import Model
import torch
cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


def remove_old_weight(opt):
    ################## setting layer
    opt.character = opt.number + opt.symbol + opt.lang_char
    opt.num_class = len(opt.character)
    # Load pretrained model
    model = Model(opt)
    state_dict = torch.load('saved_models/en_filtered/english_g2.pth')

    # Remove incompatible final prediction layer weights
    for key in list(state_dict.keys()):
        if 'Prediction' in key:
            print(f"Removing: {key}")
            del state_dict[key]

    # Load weights except final layer
    model.load_state_dict(state_dict, strict=False)  # strict False to allow missing keys
    return model


if __name__ == '__main__':
    clear_old_weight = False
    opt = get_config("config_files/en_filtered_config.yaml")

    model = None
    if clear_old_weight:
        model = remove_old_weight(opt)
    from train import train
    train(opt, show_number=2,amp=False,model=model)