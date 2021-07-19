from module import register_model
from simplecv.util import config
from simplecv.module.model_builder import make_model
import argparse
import torch
import time
from simplecv.util import param_util
import prettytable as pt

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')


def run(args):
    cfg = config.import_config(args.config_path)

    model = make_model(cfg['model']).cuda()
    cnt = param_util.count_model_parameters(model)
    model.eval()

    inputs = torch.ones(1, 3, 896, 896).cuda()
    start = time.time()
    with torch.no_grad():
        for i in range(100):
            model(inputs)
            torch.cuda.synchronize()

    total_time = (time.time() - start) / 100.
    tb = pt.PrettyTable()
    tb.field_names = ['#params', 'speed']
    tb.add_row(['{} M'.format(round(cnt / float(1e6), 3)), '{} s'.format(total_time)])
    print(tb)
    print('FPS', 1/total_time)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
