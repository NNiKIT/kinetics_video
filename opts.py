import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of 3D ResNet")
parser.add_argument('--dataset', type=str, default='KINETICS')
parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--model', type=str, default='Res3D')
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--pretrained', action='store_true', default=False)



# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--val-interval', default=5, type=int)

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')








