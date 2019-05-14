from __future__ import print_function
from options import TrainOptions
import torch
from models import TETGAN
from utils import (load_trainset_batchfnames_dualnet, prepare_batch, weights_init,
                   load_image_dualnet, to_data, save_image)
import os
import random


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

    num_params = 0
    for param in tetGAN.parameters():
        num_params += param.numel()
    print('Total number of parameters in TET-GAN: %.3f M' % (num_params / 1e6))

    print('--- training ---')
    texture_class = 'base_gray_texture' in opts.dataset_class or 'skeleton_gray_texture' in opts.dataset_class
    if texture_class:
        tetGAN.load_state_dict(torch.load(opts.model))
        dataset_path = os.path.join(opts.train_path, opts.dataset_class, 'style')
        val_dataset_path = os.path.join(opts.train_path, opts.dataset_class, 'val')
        if 'base_gray_texture' in opts.dataset_class:
            few_size = 6
            batchsize = 2
            epochs = 1500
        elif 'skeleton_gray_texture' in opts.dataset_class:
            few_size = 30
            batchsize = 10
            epochs = 300
        fnames = load_trainset_batchfnames_dualnet(dataset_path, batchsize, few_size=few_size)
        val_fnames = sorted(os.listdir(val_dataset_path))
        style_fnames = sorted(os.listdir(dataset_path)[:few_size])
    else:
        dataset_path = os.path.join(opts.train_path, opts.dataset_class, 'train')
        fnames = load_trainset_batchfnames_dualnet(dataset_path, batchsize)

    tetGAN.train()

    train_size = os.listdir(dataset_path)
    print('List of %d styles:' % (len(train_size)))

    result_dir = os.path.join(opts.result_dir, opts.dataset_class)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for epoch in range(epochs):
        for idx, fname in enumerate(fnames):
            x, y_real, y = prepare_batch(fname, 1, 1, centercropratio, augementratio, opts.gpu)
            losses = tetGAN.one_pass(x[0], None, y[0], None, y_real[0], None, 1, 0)
            if (idx+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d]' %
                      (epoch+1, epochs, idx+1, len(fnames)))
                print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f'
                      % (losses[0], losses[1], losses[2], losses[3], losses[4]))
        if texture_class and ((epoch+1) % (epochs/10)) == 0:
            for val_idx, val_fname in enumerate(val_fnames):
                v_fname = os.path.join(val_dataset_path, val_fname)
                random.shuffle(style_fnames)
                sty_fname = style_fnames[0]
                s_fname = os.path.join(dataset_path, sty_fname)
                with torch.no_grad():
                    val_content = load_image_dualnet(v_fname, load_type=1)
                    val_sty = load_image_dualnet(s_fname, load_type=0)
                    if opts.gpu != 0:
                        val_content = val_content.cuda()
                        val_sty = val_sty.cuda()
                    result = tetGAN(val_content, val_sty)
                    if opts.gpu != 0:
                        result = to_data(result)
                    result_filename = os.path.join(result_dir, str(val_idx))
                    save_image(result[0], result_filename)

        print('--- save ---')
        if texture_class:
            outname = 'save/' + 'val_epoch' + str(epoch+1) + '_' + opts.save_model_name
        else:
            outname = 'save/' + 'epoch' + str(epoch+1) + '_' + opts.save_model_name
        torch.save(tetGAN.state_dict(), outname)


if __name__ == '__main__':
    main()
