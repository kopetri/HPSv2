import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import requests
from clint.textui import progress

warnings.filterwarnings("ignore", category=UserWarning)

environ_root = os.environ.get('HPS_ROOT')
root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root == None else environ_root

model_dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val

def score(img_path: list, prompt: str, cp: str = os.path.join(root_path, 'HPS_v2_compressed.pt')) -> float:

    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp == os.path.join(root_path, 'HPS_v2_compressed.pt') and not os.path.exists(cp):
        print('Downloading HPS_v2_compressed.pt ...')
        url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
        r = requests.get(url, stream=True)
        with open(os.path.join(root_path,'HPS_v2_compressed.pt'), 'wb') as HPSv2:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    HPSv2.write(chunk)
                    HPSv2.flush()
        print('Download HPS_2_compressed.pt to {} sucessfully.'.format(root_path+'/'))
    
    checkpoint = torch.load(cp)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model.eval()
    
    result = []
    for one_img_path in img_path:
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        result.append(hps_score[0])
    return result

def score_numpy(images: list, prompt: str, cp: str = os.path.join(root_path, 'HPS_v2_compressed.pt')) -> float:

    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp == os.path.join(root_path, 'HPS_v2_compressed.pt') and not os.path.exists(cp):
        print('Downloading HPS_v2_compressed.pt ...')
        url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
        r = requests.get(url, stream=True)
        with open(os.path.join(root_path,'HPS_v2_compressed.pt'), 'wb') as HPSv2:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    HPSv2.write(chunk)
                    HPSv2.flush()
        print('Download HPS_2_compressed.pt to {} sucessfully.'.format(root_path+'/'))
    
    checkpoint = torch.load(cp)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model.eval()
    
    result = []
    for one_img_path in images:
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocess_val(Image.fromarray(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        result.append(hps_score[0])
    return result

def initHPSv2Model(cp: str = os.path.join(root_path, 'HPS_v2_compressed.pt')):

    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp == os.path.join(root_path, 'HPS_v2_compressed.pt') and not os.path.exists(cp):
        print('Downloading HPS_v2_compressed.pt ...')
        url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
        r = requests.get(url, stream=True)
        with open(os.path.join(root_path,'HPS_v2_compressed.pt'), 'wb') as HPSv2:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    HPSv2.write(chunk)
                    HPSv2.flush()
        print('Download HPS_2_compressed.pt to {} sucessfully.'.format(root_path+'/'))
    
    checkpoint = torch.load(cp)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model.eval()
    
    return model, tokenizer, preprocess_val

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(root_path,'HPS_v2_compressed.pt'), help='Path to the model checkpoint')

    args = parser.parse_args()

    hps_score = score(args.image_path, args.prompt, args.checkpoint)
    print('HPSv2 score:', hps_score)