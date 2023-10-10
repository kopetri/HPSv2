try:
    from typing import Iterable

    import torch
    from torch import nn
    import numpy as np
    from PIL import Image
    from hpsv2.src.open_clip import get_tokenizer, create_model_and_transforms
    from drlx.reward_modelling import RewardModel


    class HPSv2(RewardModel):
        """
        Reward model that rewards images with higher aesthetic score. Uses CLIP and an MLP (not put on any device by default)

        :param device: Device to load model on
        :type device: torch.device
        """
        def __init__(self, device = None):
            super().__init__()
            self.model_dict = {}
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.initialize_model()
            self.model = self.model_dict['model']
            self.preprocess_val = self.model_dict['preprocess_val']

            checkpoint = torch.load("../HPS_v2_compressed.pt")
            self.model.load_state_dict(checkpoint['state_dict'])
            self.tokenizer = get_tokenizer('ViT-H-14')
            self.model.eval()

            if self.device is not None:
                self.model.to(self.device)
        
        def initialize_model(self):
            if not self.model_dict:
                model, preprocess_train, preprocess_val = create_model_and_transforms(
                    'ViT-H-14',
                    'laion2B-s32B-b79K',
                    precision='amp',
                    device=self.device,
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
                self.model_dict['model'] = model
                self.model_dict['preprocess_val'] = preprocess_val
                
        def hpsv2_scoring(self, imgs, preprocess, prompts, model):    
            result = []
            for img in imgs:
                # Load your image and prompt
                with torch.no_grad():
                    # Process the image
                    image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device=self.device, non_blocking=True)
                    # Process the prompt
                    text = self.tokenizer([prompts]).to(device=self.device, non_blocking=True)
                    # Calculate the HPS
                    with torch.cuda.amp.autocast():
                        outputs = model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T

                        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(hps_score[0])
            return result


        def forward(self, images : np.ndarray, prompts : Iterable[str]):
            return self.hpsv2_scoring(
                images,
                self.preprocess,
                prompts,
                self.model
            )
except:
    pass