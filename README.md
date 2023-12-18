# StyleVSG
Codes for Paper: Attractive Storyteller: Stylized Visual Storytelling with Unpaired Text

## Features
The features can be downloaded from: https://drive.google.com/drive/folders/1gmSPserBvSMR8sVp3_8UKtsjgEwpnHsj?usp=drive_link

## Data Process
```
bash process_multi.sh
bash process_vist.sh
```

## Train
e.g
```
train.sh fairy_model/ fairy
```

## Inference
e.g
```
evaluate.sh fairy_model/ fairy_model/checkpoint_best.pt fairy 10
```
