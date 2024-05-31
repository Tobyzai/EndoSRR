# EndoSRR
[EndoSRR: a comprehensive multi-stage approach for endoscopic specular reflection removal](https://link.springer.com/article/10.1007/s11548-024-03137-8)

## Demo


https://github.com/Tobyzai/EndoSRR/assets/53550360/3d28f2e9-a43b-417f-9d33-33110378be4d



## Setup
This code was implemented with Python 3.8.16 and Pytorch 1.13.0+cu116.You can install all the requirements via:

`pip install -r requirements.txt`

## Fine-tuning SAM-Adapter for Reflection Detection
<img width="549" alt="SAM_Adapter(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/85f265d4-1522-4383-a64a-01b90db601ab">

Download the [vit-b](https://github.com/facebookresearch/segment-anything) pretrained model of SAM and place it in the pretrained folder.

After configuring the yaml file, run the following command to fine-tune the SAM-Adapter.

`CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py --config configs/cod-sam-vit-b.yaml`

## LaMa for Reflection region inpainting
<img width="534" alt="LaMa(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/ef62dde6-67ab-4535-a2b7-dcb552577095">

Download the [Big-LaMa](https://github.com/advimman/lama) pretrained model of LaMa and place it in the pretrained folder.

## Visualization of optimization strategy


https://github.com/Tobyzai/EndoSRR/assets/53550360/de219b16-6868-477f-9f25-f0957eebe5bc


## EndoSRR pre-trained model for Endoscopic Specular Reflection Removal

<img width="530" alt="flowchat" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/c7d05e59-403e-40ef-b924-b9b44c284aa6">

Specular reflection removal using the [EndoSRR pre-trained model](https://drive.google.com/drive/folders/12htzO9lLFakWj0WjblmxwEZwrW2NyO6z?usp=sharing).

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
<img width="529" alt="application(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/f6554787-003f-41a3-a3bc-b7dcbffa3740">

## Dataset
Process for creating endoscopic specular reflection weakly labeled dataset.

<img width="545" alt="creation_dataset(1)" src="https://github.com/Tobyzai/EndoSRR/assets/53550360/db795987-6b76-4e23-b041-679f5dcac8c9">

The whole [Reflection Dataset](https://drive.google.com/drive/folders/12htzO9lLFakWj0WjblmxwEZwrW2NyO6z?usp=sharing) is released.

## Acknowledgement
Our code is based on [SAM-Adapter](https://github.com/facebookresearch/segment-anything) and [LaMa](https://github.com/advimman/lama).

