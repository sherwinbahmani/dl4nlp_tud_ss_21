import os
from argparse import ArgumentParser
import pathlib
import torch
from datetime import datetime

from rank_questions.model_wrapper import ModelWrapper

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--cuda-devices", default="0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--cpu", default=False, action="store_true",
                    help="Binary flag. If set all operations are performed on the CPU.")
parser.add_argument("--data-parallel", default=False, action="store_true")
parser.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"))
parser.add_argument("--train-file-name", type=str, default="training.tsv")
parser.add_argument("--test-file-name", type=str, default="test_set.tsv")
parser.add_argument("--question-bank-name", type=str, default="question_bank.tsv")
parser.add_argument("--checkpoints-root", type=str, default=os.path.join(os.getcwd(), "checkpoints"))
parser.add_argument("--checkpoint-name", type=str, default=None, help="File name of timestamp.pth")
parser.add_argument("--runs-root", type=str, default=os.path.join(os.getcwd(), "runs"))
parser.add_argument("--txt-root", type=str, default=os.path.join(os.getcwd(), "txt"))
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--betas", nargs="+", type=int, default=[0.9, 0.999])
parser.add_argument("--weight-decay", type=float, default=1e-2)
parser.add_argument("--val-start", type=int, default=0)
parser.add_argument("--val-step", type=int, default=1)
parser.add_argument("--val-split", type=float, default=0.005, help="Take given percentage of train dataset for val")
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--top-k-accuracy", type=int, default=50)
parser.add_argument("--true-label", type=int, default=1)
parser.add_argument("--false-label", type=int, default=0)
parser.add_argument("--train", default=True, action="store_true")
parser.add_argument("--val", default=True, action="store_true")
parser.add_argument("--test", default=True, action="store_true")

# Get arguments
args = parser.parse_args()

# Set device type
args.device = "cpu" if args.cpu else "cuda"

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

# Create directories and time stamp of session
pathlib.Path(args.checkpoints_root).mkdir(parents=True, exist_ok=True)
args.time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(f"Run {args.time_stamp} has started!")

# Create model wrapper
model_wrapper = ModelWrapper(args)

# Load model from checkpoint if given
if os.path.exists(args.checkpoints_root) and args.checkpoint_name != None:
    checkpoint = os.path.join(args.checkpoints_root, args.checkpoint_name)
    model_wrapper.model.load_state_dict(torch.load(checkpoint, map_location=model_wrapper.device), strict=True)
else:
    print("Model initialized based on not fine-tuned parameters")

# Train, validate and test model
if args.train:
    model_wrapper.train()
if args.test:
    model_wrapper.test()

if __name__ == '__main__':
    pass
