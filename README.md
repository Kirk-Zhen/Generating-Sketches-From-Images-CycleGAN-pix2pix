# Generating Sketches From Images(CycleGAN, pix2pix)


<!-- |||
|:---|:---|
|Student Name|Zijian Zhen
|Student ID|101087006 -->


# Files & Folders

1. `loss_records`: stores `.csv` files that records loss of each iterations during the training process.
2. `data` (initially empty): stores the training data from Sketchy Database.
3. `models_plot`: stores the `.png` file of model structure produced by Keras
4. `models`: stores the models and checkpoints generated during training process.
5. `process`: stores `.png` files which investigated the generated sketch after each epoch during the training process.
6. `project_figure`: includes all figures used in the report
7. `LoadData.py`: includes class for loading data
8. `train_pix2pix.py`: executable to train pix2pix in the project
9. `train_CycleGAN.py`: executable to train CycleGAN in the project
10. `train_con_CycleGAN.py`: executable to train con-CycleGAN in the project
11. `train_InceptionV3.py`: executable to train InceptionV3 model for performance evaluation
12. `README.md`: instruction for the project
13. `requirements.txt`: includes all the required environment information for quick installation.


Notice: Except for `data`, all folder are empty currently. After executing the program following the instruction, files will automatically stored in the folders.

# Environment and Dependencies

## Hardware dependencies
To ensure that the program can run smoothly.

The program must run on a device with RAM of at least `20G` and a `GPU` to ensure enough space for local storage and GPU acceleration on a TensorFlow backend.

## Python Packages
|||
|:---|:---|
|Package|Version
|||
|Python|3.7.10
|numpy|1.19.5
|Keras|2.4.3
|matplotlib|3.2.2
|pandas|1.1.5
|PIL|7.1.2
|scikit-learn|0.22.2.post1
|scipy|1.4.1
|tensorflow|2.4.1

`requirements.txt` contains all the required environment information for quick installation. You can activate your virtual environment and install all required packages:

 `python -m pip install -r requirements.txt`

If you failed to install `keras-contrib` by the above comment. You can try to install `keras-contrib` separately:

`pip install git+https://www.github.com/farizrahman4u/keras-contrib.git`

If you are still not able to install it, please check https://github.com/keras-team/keras-contrib for more detailed instruction.


## Prepare Data: Sketchy Database

1. Download `rendered_256x256.7z` at this [Link](https://goo.gl/SNpMmK).
2. Extract `rendered_256x256.7z`, results in `256x256` directory.


Notice that `stats.csv`, which has been contained in `data` folder, is necessary for our programs.

The final folder structure is like:
```
./
|
| data/
|___
|   |stats.csv
|   |____
|   
|___
|   |256x256/
|   |___    
|   |   |photo/
|   |   |___
|   |       |tx_00000000000/
|   |       |___
|   |           |airplane/
|   |           |alarm_clock/
|   |           |...
|   |___
|       |sketch/
|       |___
|           |tx_00000000000/
|           |___
|               |airplane/
|               |alarm_clock/
|               |...
|           
```


# Running Instructions
All default training parameters are exact the same as that stated in the report. You may check the `Methodology` part of the report to help understanding to the code.


## Train InceptionV3 classifier



Quick Command to train:
`python train_InceptionV3.py`

The command enable a training with `100` epochs over the selected data in Sketchy Database.

A structure plot of InceptionV3 will be generated at `models_plot/InceptionV3.png`

The finalized InceptionV3 model would be store at `models/InceptionV3.h5`

The accuracy and the training process of the model will be shown in the terminal.


## Train pix2pix model

Quick Command to train: `python train_pix2pix.py`

A structure plot of generator and discriminator will the generated in `models_plot` folder:
1. Generator: `models_plot/gen_pix2pix.png`
2. Discriminator: `models_plot/dis_pix2pix.png`

Loss for each iteration will be recorded in a `.csv` file: `loss_records/loss_pix2pix.csv`

After the `i`-th epochs finished:

1. The program will automatically generate checkpoints models into `.h5` file: `models/pix2pix_Sketch_Generator_i.h5`

2. Sketch samples will be generated in `process/pix2pix_plot_i.png`





## Train CycleGAN model

Quick Command to train: `python train_CycleGAN.py`

A structure plot of generator and discriminator will the generated in `models_plot` folder:
1. Generator: `models_plot/gen_CycleGAN.png`
2. Discriminator: `models_plot/dis_CycleGAN.png`

Classification accuracy evaluated by a pre-trained InceptionV3 model will be printed in the terminal.

Loss for each iteration will be recorded in a `.csv` file: `loss_records/loss_CycleGAN.csv`

After the `i`-th epochs finished:

1. The program will automatically generate checkpoints models into `.h5` file: `models/CycleGAN_Sketch_Generator_i.h5`

2. Sketch samples will be generated in `process/CycleGAN_AtoB_plot_i.png`




## Train con-CycleGAN model

Quick Command to train: `python train_con_CycleGAN.py`

A structure plot of generator and discriminator will the generated in `models_plot` folder:
1. Generator: `models_plot/gen_con_CycleGAN.png`
2. Discriminator: `models_plot/dis_con_CycleGAN.png`

Classification accuracy evaluated by a pre-trained InceptionV3 model will be printed in the terminal.

Loss for each iteration will be recorded in a `.csv` file: `loss_records/loss_con_CycleGAN.csv`

After the `i`-th epochs finished:

1. The program will automatically generate checkpoints models into `.h5` file: `models/con_CycleGAN_Sketch_Generator_i.h5`

2. Sketch samples will be generated in `process/con_CycleGAN_AtoB_plot_i.png`
