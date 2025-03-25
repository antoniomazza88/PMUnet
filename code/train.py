import torch
import os
from experiments_utility import trainloop
import argparse




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Sentinel-5p training for PM 2.5 estimation")
    parse.add_argument("--dataset_path", default=r"..\dataset",
                       help="Folder where the files Train.npy, Val.npy, Test.npy and Gen.npy are stored")
    parse.add_argument("--out_folder", default=r"..\output",
                       help="Folder where the experiments will be stored")
    parse.add_argument("--model", default='PMUnet', help="Model you want to use", choices=['PMUnet', 'PMRes', 'PMSlim'])
    parse.add_argument("--exp_name", default=None, help="Name of the experiment, if None, the experiment will have the name of the chosen model", type=str)
    parse.add_argument("--gpu_number", default='0', help="GPU ID if present")

    args = parse.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    torch.set_num_threads(1)
    trainloop(args)

