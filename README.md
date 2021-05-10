# connect-caption-and-trace
This repository contains the reference code for our paper ``Connecting What to Say With Where to Look by Modeling Human Attention Traces'' (CVPR2021).

![example results](images/example_coco_tasks_1_2_3.png)

## Requirements
* Python 3
* PyTorch 1.5+ (along with torchvision)
* [coco-caption](https://github.com/tylin/coco-caption) (Remember to follow initialization steps in coco-caption/README.md)

## Prepare data
Our experiments cover all four datasets included in Localized Narratives: COCO2017, Flickr30k, Open Images and ADE20k. For each dataset, we need four things: (1) json file containing image info and word tokens. (DATASET_LN.json) (2) h5 file containing caption labels (DATASET_LN_label.h5) (3) The trace labels extracted from Localized Narratives (DATASET_LN_trace_box/) (4) json file for coco-caption evaluation (captions_DATASET_LN_test.json) (5) Image features (with bounding boxes) extracted by a Mask-RCNN pretrained on Visual Genome. 

You can download (1--4) from [here](https://drive.google.com/drive/folders/1DnaJu7lZc1dmyJyr-pxdK4l2sWjWUo03?usp=sharing):  (make a folder named `data` and put (1--3) in it, and put (4) under coco-caption/annotaions/)

To get (5), you can use Detectron2. First, install [Detectron2](https://github.com/lichengunc/detectron2/tree/genome_obj+attr), then follow [Prepare COCO-style annotations for Visual Genome](https://github.com/lichengunc/detectron2/blob/genome_obj%2Battr/NOTES.md) (We use the pre-trained Resnet101-C4 model provided there). After that you can utilize tools/extract_feats.py in [Detectron2](https://github.com/lichengunc/detectron2/tree/genome_obj+attr) to extract features. Finally, run scripts/prepare_feats_boxes_from_npz.py in this repo to prepare features and bounding boxes in seperate folders for training.

For COCO dataest you can also directly use the features provided by Peter Anderson [here](https://github.com/peteanderson80/bottom-up-attention). The performance is almost the same (with around 0.2% difference.)

## Training 
The dataset can be chosen from the four datasets. The `--task` can be chosen from `trace`, `caption`, `c_joint_t` and `pred_both`. The `--eval_task` can be chosen from `trace`, `caption`, and `pred_both`. 

### COCO: joint training of controlled caption generation and trace generation (N=2 layers, evaluated on caption generation)
```python
python tools/train.py --language_eval 0 --id transformer_LN_coco  --caption_model transformer --input_json data/coco_LN.json --input_att_dir Dir_to_image_features_vg --input_box_dir Dir_to_bounding_boxes_vg --input_label_h5 data/coco_LN_label.h5 --batch_size 30 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3  --save_checkpoint_every 1000 --max_epochs 30 --max_length 225 --seq_per_img 1 --use_box 1   --use_trace 1  --input_trace_dir data/coco_LN_trace_box --use_trace_feat 0 --beam_size 1 --val_images_use -1 --num_layers 2 --task c_joint_t --eval_task caption --dataset_choice=coco
```

### Open image: training of generating caption and trace at the same time (N=1 layers, evaluated on predicting both)
```python
python tools/train.py --language_eval 0 --id transformer_LN_openimg  --caption_model transformer --input_json data/openimg_LN.json --input_att_dir Dir_to_image_features_vg --input_box_dir Dir_to_bounding_boxes_vg --input_label_h5 data/openimg_LN_label.h5 --batch_size 30 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3  --save_checkpoint_every 1000 --max_epochs 30 --max_length 225 --seq_per_img 1 --use_box 1   --use_trace 1  --input_trace_dir data/openimg_LN_trace_box --use_trace_feat 0 --beam_size 1 --val_images_use -1 --num_layers 1 --task pred_both --eval_task pred_both --dataset_choice=openimg
```

### Flickr30k: training of controlled caption generation alone (N=1 layer)
```python
python tools/train.py --language_eval 0 --id transformer_LN_flk30k  --caption_model transformer --input_json data/flk30k_LN.json --input_att_dir Dir_to_image_features_vg --input_box_dir Dir_to_bounding_boxes_vg --input_label_h5 data/flk30k_LN_label.h5 --batch_size 30 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3  --save_checkpoint_every 1000 --max_epochs 30 --max_length 225 --seq_per_img 1 --use_box 1   --use_trace 1  --input_trace_dir data/flk30k_LN_trace_box --use_trace_feat 0 --beam_size 1 --val_images_use -1 --num_layers 1 --task caption --eval_task caption --dataset_choice=flk30k
```

### ADE20k: training of controlled trace generation alone (N=1 layer)
```python
python tools/train.py --language_eval 0 --id transformer_LN_ade20k  --caption_model transformer --input_json data/ade20k_LN.json --input_att_dir Dir_to_image_features_vg --input_box_dir Dir_to_bounding_boxes_vg --input_label_h5 data/ade20k_LN_label.h5 --batch_size 30 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3  --save_checkpoint_every 1000 --max_epochs 30 --max_length 225 --seq_per_img 1 --use_box 1   --use_trace 1  --input_trace_dir data/ade20k_LN_trace_box --use_trace_feat 0 --beam_size 1 --val_images_use -1 --num_layers 1 --task trace --eval_task trace --dataset_choice=ade20k
```

## Evaluating
### COCO: joint training of controlled caption generation and trace generation (N=2 layers, evaluated on caption generation)
```python
python tools/train.py --language_eval 1 --id transformer_LN_coco  --caption_model transformer --input_json data/coco_LN.json --input_att_dir Dir_to_image_features_vg --input_box_dir Dir_to_bounding_boxes_vg --input_label_h5 data/coco_LN_label.h5 --batch_size 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3  --save_checkpoint_every 1000 --max_epochs 30 --max_length 225 --seq_per_img 1 --use_box 1   --use_trace 1  --input_trace_dir data/coco_LN_trace_box --use_trace_feat 0 --beam_size 5 --val_images_use -1 --num_layers 2 --task c_joint_t --eval_task caption --dataset_choice=coco
```
### COCO: joint training of controlled caption generation and trace generation (N=2 layers, evaluated on trace generation)
```python
python tools/train.py --language_eval 1 --id transformer_LN_coco  --caption_model transformer --input_json data/coco_LN.json --input_att_dir Dir_to_image_features_vg --input_box_dir Dir_to_bounding_boxes_vg --input_label_h5 data/coco_LN_label.h5 --batch_size 30 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3  --save_checkpoint_every 1000 --max_epochs 30 --max_length 225 --seq_per_img 1 --use_box 1   --use_trace 1  --input_trace_dir data/coco_LN_trace_box --use_trace_feat 0 --beam_size 1 --val_images_use -1 --num_layers 2 --task c_joint_t --eval_task trace --dataset_choice=coco
```
### Open image: training of generating caption and trace at the same time (N=1 layers, evaluated on predicting both)
```python
python tools/train.py --language_eval 1 --id transformer_LN_openimg  --caption_model transformer --input_json data/openimg_LN.json --input_att_dir Dir_to_image_features_vg --input_box_dir Dir_to_bounding_boxes_vg --input_label_h5 data/openimg_LN_label.h5 --batch_size 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 100 --learning_rate_decay_every 3  --save_checkpoint_every 1000 --max_epochs 30 --max_length 225 --seq_per_img 1 --use_box 1   --use_trace 1  --input_trace_dir data/openimg_LN_trace_box --use_trace_feat 0 --beam_size 5 --val_images_use -1 --num_layers 1 --task pred_both --eval_task pred_both --dataset_choice=openimg
```


## Acknowledgements
Some components of this repo were built from Ruotian Luo's [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch).



