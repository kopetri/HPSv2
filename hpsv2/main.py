from hpsv2_reward import HPSv2
from drlx.pipeline.pickapic_prompts import PickAPicPrompts
from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig

# We import a reward model, a prompt pipeline, the trainer and config

pipe = PickAPicPrompts()
config = DRLXConfig.load_yaml("../ddpo_sd_imagenet.yml")
trainer = DDPOTrainer(config)

trainer.train(pipe, HPSv2())