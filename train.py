import argparse
from model.sctcr import SCTCR
from model.trainer.default import Trainer
from data.dataset import generate_dataset
from utils.tools import GetArg
from utils.loss import SmoothL1SSIMLoss


parser = argparse.ArgumentParser()
# common arguments
parser.add_argument('--debug',      type=bool,  default=False,  help='debug mode')
parser.add_argument('--dataset',    type=str,   default='whu',  help='default, x, rgb, whu')
parser.add_argument('--RGB',        type=bool,  default=False,  help='use only RGB bands')
parser.add_argument('--batch_size', type=int,   default=8,      help='batch size')
parser.add_argument('--epochs',     type=int,   default=100,    help='train epochs')
parser.add_argument('--step_size',  type=int,   default=5,      help='StepLR step size')
parser.add_argument('--step_decay', type=int,   default=0.5,    help='StepLR step decay')
parser.add_argument('--lr',         type=float, default=8e-5,   help='initial learning rate')
parser.add_argument('--ckpts',      type=str,   default='',     help='load checkpoints from ${ckpts} if is valid.')
parser.add_argument('--log_step',   type=int,   default=50,     help='log step interval')
parser.add_argument('--save_step',  type=int,   default=25,     help='save every ${save_interval} epochs')
parser.add_argument('--save_root',  type=str,   default='/public/home/cornhut/mjs/ckpts', help='directory for saving checkpoints')

# local arguments
parser.add_argument('--opt_dim',    type=int,   default=128,    help='dim of optical feature in the network')
parser.add_argument('--sar_dim',    type=int,   default=128,    help='dim of SAR feature in the network')
parser.add_argument('--num_trans',  type=int,   default=12,     help='num of transformerlayers')
parser.add_argument('--num_heads',  type=int,   default=8,      help='num of heads')
parser.add_argument('--use_res',    type=bool,  default=False,  help='use resblock or not')
opts = parser.parse_args()

sets, c1, c2 = generate_dataset(GetArg(opts, 'dataset'), GetArg(opts, 'RGB'), GetArg(opts, 'debug'))

net = SCTCR(
    sar_channels=c1,
    opt_channels=c2,
    opt_hid_dim=GetArg(opts, 'opt_dim'),
    sar_hid_dim=GetArg(opts, 'sar_dim'),
    num_trans_layer=GetArg(opts, 'num_trans'),
    num_heads=GetArg(opts, 'num_heads'),
    use_resblock=GetArg(opts, 'use_res')
)

trainer = Trainer(
    name='SCTCR',
    model=net,
    lossfunc=SmoothL1SSIMLoss(),
    datasets=sets,
    batch_size=GetArg(opts, 'batch_size'),
    num_epochs=GetArg(opts, 'epochs'),
    log_interval=GetArg(opts, 'log_step'),
    save_interval=GetArg(opts, 'save_step'),
    lr=GetArg(opts, 'lr'),
    step_lr_step_size=GetArg(opts, 'step_size'),
    step_lr_decay=GetArg(opts, 'step_decay'),
    save_root=GetArg(opts, 'save_root')
)

trainer.load(GetArg(opts, 'ckpts'))
trainer.train()
