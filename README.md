# Multi-task Interaction learning for Spatiospectral Image Super-Resolution


```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--threads THREADS] [--seed SEED]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
```
This example trains a spatiospectral super-resolution network on the [CAVE dataset](https://www.cs.columbia.edu/CAVE/databases/), using the first 22 HSIs for training, the remaining 10 HSIs for testing. A trained model can be downloaded at https://drive.google.com/file/d/1x60giSaTPZWZoG25sRhEbhFxVN8C_3u_/view?usp=sharing
## Example Usage:

### Train

`python main.py --upscale_factor 2 --batchSize 4 --nEpochs 30 --lr 0.001`

### Test
`python main.py --mode 0 --nEpochs 100`


## The code is for the work:

```
@article{ma2022,
  title={Multi-task Interaction learning for Spatiospectral Image Super-Resolution},
  author={Qing Ma, Junjun Jiang, Xianming Liu, and Jiayi Ma},
  journal={IEEE Transactions on Image Processing},
  volume={},
  number={},
  pages={},
  year={2022},
}
```
