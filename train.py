from __future__ import print_function
from options import TrainOptions
import torch
from models import TETGAN
from utils import (load_trainset_batchfnames_dualnet, prepare_batch, weights_init)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # data loader
    print('--- load parameter ---')
    # outer_iter = opts.outer_iter
    # fade_iter = max(1.0, float(outer_iter / 2))
    epochs = opts.epoch
    batchsize = opts.batchsize
    # datasize = opts.datasize
    # datarange = opts.datarange
    augementratio = opts.augementratio
    centercropratio = opts.centercropratio

    # model
    print('--- create model ---')
    tetGAN = TETGAN(gpu=(opts.gpu != 0))
    if opts.gpu != 0:
        tetGAN.cuda()
    tetGAN.init_networks(weights_init)
    tetGAN.train()

    print('--- training ---')
    dataset_path = os.path.join(opts.train_path, opts.dataset_class, 'train')
    train_size = os.listdir(dataset_path)
    print('List of %d styles:' % (len(train_size)))

    fnames = load_trainset_batchfnames_dualnet(dataset_path, batchsize)
    for epoch in range(epochs):
        for idx, fname in enumerate(fnames):
            x, y_real, y = prepare_batch(fname, 1, 1, centercropratio, augementratio, opts.gpu)
            losses = tetGAN.one_pass(x[0], None, y[0], None, y_real[0], None, 1, 0)
        if (idx+1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d]' %
                  (epoch+1, epochs, idx+1, len(fnames)))
            print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f'
                  % (losses[0], losses[1], losses[2], losses[3], losses[4]))

    print('--- save ---')
    torch.save(tetGAN.state_dict(), opts.save_model_name)


if __name__ == '__main__':
    main()
