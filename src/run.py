import torch
from options import Options
from dataset import dataset_makeup
from model import LADN
import datetime
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

# from saver import Saver

'''
opts.phase
    This option set whether the network is running in training mode or in testing mode.
    if opts.phase == "train", train normally.
    if opts.phase == "test", only run one epoch of testing and exit.

opts.test_forward
    If opts.phase == "test", this flag will be set.
    if opts.test_forward is True, run over the testing set every opts.test_interval epochs.
    if opts.test_forward is False, train the network only.

opts.interpolate_forward
    It is independent of the above two options.
    If opts.interpolate_forward is True, interpolation will also be run in every test forward
'''


def tensor2img(img):
    img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)


def create_grid_plot(imgs_XY, imgs_YX) -> plt.Figure:
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(100, 50), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

    for idx, image in enumerate(imgs_XY):
        # we compute our current position in the subplot
        #  image = convert_tensor_to_image(image)
        axes[0, idx].axis("off")
        axes[0, idx].imshow(image, aspect=1)
    for idx, image in enumerate(imgs_YX):
        # we compute our current position in the subplot
        # image = convert_tensor_to_image(image)
        axes[1, idx].axis("off")
        axes[1, idx].imshow(image, aspect=1)
    return fig


# save the individual result of test_forward during the progress of training
def save_test_img(ep, index, model, index_a, index_b):
    names = ['source', 'transfer', 'random_makeup', 'source_recon', 'source_cycle_recon',
             'reference', 'demakeup', 'random_demakeup', 'reference_recon', 'reference_cycle_recon',
             'blend']
    images = ['real_A_encoded', 'fake_B_encoded', 'fake_B_random', 'fake_AA_encoded', 'fake_A_recon',
              'real_B_encoded', 'fake_A_encoded', 'fake_A_random', 'fake_BB_encoded', 'fake_B_recon',
              'real_C_encoded']
    imgs = []

    # path = '%s/test_%05d_%05d_%05d' % (self.test_image_dir, index_b, index_a, ep)
    # if not os.path.exists(path):
    #    os.mkdir(path)

    for i in range(11):
        img = model.normalize_image(getattr(model, images[i])).detach()[0:1, ::]
        img = tensor2img(img)
        imgs.append(img)

    results_plot = create_grid_plot([imgs[0], imgs[1], imgs[4], imgs[3]],[imgs[5], imgs[6], imgs[9], imgs[8]])
    wandb.log({"results for image %d" % (index): results_plot}, step=ep)
    plt.close(results_plot)

    # img = Image.fromarray(img)
    # img.save(os.path.join(path, names[i] + '.jpg'))

# save model
def write_model(ep, total_it, model, model_dir):
    print('--- save the model @ ep %d ---' % (ep))
    if ep in [30, 100, 300, 500, 700, 1000]:
        model.save('%s/%05d.pth' % (model_dir, ep), ep, total_it)
    else:
        model.save('%s/%05d.pth' % (model_dir, ep%2), ep, total_it)

def main():
    # parse options
    parser = Options()
    opts = parser.parse()

    name = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    wandb.init(project="LADN-official", entity="daliana-st", name=name)

    test_indeces_list = [0, 11, 22, 33]

    # If the overall mode is testing
    if opts.phase == "test":
        opts.test_forward = True

    # daita loader
    print('\n--- load dataset ---')
    dataset = dataset_makeup(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.nThreads)

    # A separate dataset for test forward
    if opts.test_forward:
        print("creating dataset_test")
        dataset_test = dataset_makeup(opts, mode='test')
    # Another separate dataset for interpolation forwarding
    if opts.interpolate_forward:
        print("creating dataset_interpolate")
        dataset_interpolate = dataset_makeup(opts, mode="interpolate")

    # model
    print('\n--- load model ---')
    model = LADN(opts)
    if opts.resume is None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)

    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    # saver = Saver(opts, len(dataset))

    # Run only one epoch when testing
    if opts.phase == "test":
        opts.n_ep = ep0 + 1
        opts.test_interval = 1

    model_dir = os.path.join("results", name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        # Run the testing set every test_interval epochs
        if opts.test_forward and ep > 0 and (ep + 1) % opts.test_interval == 0:
            print("starting forward for testing images")
            for i in test_indeces_list:
                print("forwarding images %d testing." % (i))
                data = dataset_test[i]
                device = torch.device('cuda:{}'.format(opts.backup_gpu)) if opts.gpu >= 0 else torch.device('cpu')
                images_a = data['img_A'].to(device).detach().unsqueeze(0)
                images_b = data['img_B'].to(device).detach().unsqueeze(0)
                images_c = data['img_C'].to(device).detach().unsqueeze(0)
                index_a = int(data['index_A'])
                index_b = int(data['index_B'])

                model.test_forward(images_a, images_b, images_c)
                # saver.write_test_img(ep, i, model, index_a = index_a, index_b = index_b)
                save_test_img(ep, i, model, index_a=index_a, index_b=index_b)

        if opts.interpolate_forward and (ep + 1) % opts.test_interval == 0:
            print("starting forward for interpolated images")
            for i in range(len(dataset_interpolate)):
                if (i + 1) % 10 == 0:
                    print("forwarding %d/%d images for interpolating." % (i, len(dataset_interpolate)))
                data = dataset_interpolate[i]
                device = torch.device('cuda:{}'.format(opts.backup_gpu)) if opts.gpu >= 0 else torch.device('cpu')
                images_a = data['img_A'].to(device).detach().unsqueeze(0)
                images_b1 = data['img_B'].to(device).detach().unsqueeze(0)
                images_b2 = data['img_C'].to(device).detach().unsqueeze(0)
                images_b = torch.cat([images_b1, images_b2], dim=0)
                index_a = int(data['index_A'])
                index_b = int(data['index_B'])

                model.interpolate_forward(images_a, images_b1, images_b2)
            # saver.save_interpolate_img(ep, i, model, opts.interpolate_num, index_a = index_a, index_b = index_b)

        if opts.phase == "train":
            for it, data in enumerate(train_loader):
                device = torch.device('cuda:{}'.format(opts.backup_gpu)) if opts.gpu >= 0 else torch.device('cpu')
                images_a = data['img_A'].to(device).detach()
                images_b = data['img_B'].to(device).detach()
                images_c = data['img_C'].to(device).detach()

                if images_a.size(0) != opts.batch_size:
                    continue

                # update model
                if (it + 1) % opts.d_iter != 0 and not it == len(train_loader) - 1:
                    model.update_D_content(images_a, images_b)
                    if opts.style_dis:
                        model.update_D_style(images_a, images_b, images_c)
                    if opts.local_style_dis:
                        model.update_D_local_style(data)
                    continue
                else:
                    model.update_D(data)
                    model.update_EG()

                    wandb.log({'gan_loss_a': model.gan_loss_a, 'gan_loss_b': model.gan_loss_b}, step=total_it)
                    wandb.log({'disA_loss': model.disA_loss, 'disB_loss': model.disB_loss}, step=total_it)
                    wandb.log({'disEYE_loss': model.disEYEStyle_loss, 'G_GAN_lefteye_loss': model.G_GAN_eye_style}, step=total_it)

                    if opts.contrastive_loss:
                        wandb.log({'contrastive_loss': model.total_patch_nce_loss}, step=total_it)

                # save to display file
                # if not opts.no_display_img:
                # saver.write_display(total_it, model)

                print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
                total_it += 1
                if total_it >= max_it:
                    # saver.write_img(-1, model)
                    # saver.write_model(-1, model)
                    break

            # decay learning rate
            if opts.n_ep_decay > -1:
                model.update_lr()

        # save result image
        # saver.write_img(ep, model)

        # Save network weights
        write_model(ep, total_it, model, model_dir)

    return


if __name__ == '__main__':
    main()
