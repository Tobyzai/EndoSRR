# EndoSRR
<<<<<<< HEAD
Endoscopic Specular Reflection Removal 

## Demo
https://github.com/Tobyzai/EndoSRR/assets/53550360/b99706ac-2275-4771-a99e-435b5e468d57

## Setup
This code was implemented with Python 3.8.16 and Pytorch 1.13.0+cu116.You can install all the requirements via:

`pip install -r requirements.txt`

## Fine-tuning SAM-Adapter for Reflection Detection
<img width="549" alt="SAM_Adapter(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/0a4ac283-daf3-4d4f-9293-aa433a305761">

Download the [vit-b](https://github.com/facebookresearch/segment-anything) pretrained model of SAM and place it in the pretrained folder.

After configuring the yaml file, run the following command to fine-tune the SAM-Adapter.

`CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py --config configs/cod-sam-vit-b.yaml`

## LaMa for Reflection region inpainting
<img width="534" alt="LaMa(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/08b1e410-8ed6-4da5-bb7d-d92f2339e14d">

Download the [Big-LaMa](https://github.com/advimman/lama) pretrained model of LaMa and place it in the pretrained folder.

## EndoSRR pre-trained model for Endoscopic Specular Reflection Removal

<img width="530" alt="flowchat" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/74f9526f-eb50-4af9-84eb-e3711efcb3e6">

Specular reflection removal using the [EndoSRR pre-trained model](https://github.com/Tobyzai/EndoSR).

```
CUDA_VISIBLE_DEVICES=0 python EndoSRR.py 
--config configs/cod-sam-vit-b.yaml  
--lama_config lama/configs/prediction/default.yaml   
--lama_ckpt /pretrained/big-lama/   
--model /pretrained/SAM_Adapter/model_epoch_best.pth   
--input_path /image   
--save_mask_path 'EndoSRR/mask'   
--save_inpaint_path 'EndoSRR/inpaint_15'   
--final_mask_path 'EndoSRR/final_mask'  
--final_inpaint_path 'EndoSRR/final_inpaint'   
--dilate_kernel_size 15 
```

## Application
<img width="529" alt="application(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/cde4f9e4-bb2f-464b-af83-5a58ec22ff15">

## Dataset
Process for creating endoscopic specular reflection weakly labeled dataset.

<img width="545" alt="creation_dataset(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/d2970b93-6820-4a11-95d2-096f42ab0b0c">

The whole [Reflection Dataset](https://github.com/Tobyzai/EndoSR) will be released soon, samples can be seen at EndoRR_Dataset.

## Acknowledgement
Our code is based on [SAM-Adapter](https://github.com/facebookresearch/segment-anything) and [LaMa](https://github.com/advimman/lama).









=======
Endoscopy Specular Reflection Removal 
>>>>>>> 3aab9a05a482658aba8b30164c4586ddfba2c102
