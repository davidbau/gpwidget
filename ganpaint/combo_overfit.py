# Combines:
#   Overfit the selected layers to the image
#   And also add final activations to fit the image

import torch, multiprocessing, itertools, os, shutil, PIL, argparse, numpy
import copy, skimage
from collections import OrderedDict
from numbers import Number
from torch.nn.functional import mse_loss, l1_loss
from encoder.progress import default_progress, print_progress, verbose_progress
from encoder.progress import post_progress, desc_progress
from encoder import zdataset, seededsampler, zipdataset
from encoder import proggan, subsequence, customnet, parallelfolder
from encoder import encoder_net, encoder_loss, imagedata
from encoder import edit_algs
from encoder.imagedata import load_images, numpy_from_tensor
from torchvision import transforms, models
from torchvision.models.vgg import model_urls
from encoder.pidfile import exit_if_job_done, mark_job_done
from encoder import nethook
from encoder.pidfile import exit_if_job_done, mark_job_done
from encoder.encoder_loss import cor_square_error
from encoder.nethook import InstrumentedModel
from encoder import easydict
from scipy.io import savemat

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--wlr', type=float, help='Weight learning rate',
        default=0.01)
parser.add_argument('--alr', type=float, help='Residual learning rate',
        default=0.01)
parser.add_argument('--w_layers', nargs='*', default=None,
        help='layers to overfit: any list of layer5 and up')
parser.add_argument('--a_layers', nargs='*', default=None,
        help='layers to add residual: any list of layer5 and up')
parser.add_argument('--w_steps', type=int, default=1000,
        help='number of steps for w optimization')
parser.add_argument('--a_steps', type=int, default=1000,
        help='number of steps for a optimization')
parser.add_argument('--image_number', type=int, help='Image number',
        default=95)
parser.add_argument('--image_source', #choices=['val', 'train', 'gan', 'test'],
        default='train')
parser.add_argument('--redo', type=int, help='Nonzero to delete done.txt',
        default=0)
parser.add_argument('--model', type=str, help='Dataset being modeled',
        default='church')
parser.add_argument('--halfsize', type=int,
        help='Set to 1 for half size enoder',
        default=0)
parser.add_argument('--check_edit', type=int,
        help='Set to 1 to generate editing tests',
        default=1)
parser.add_argument('--snapshot_every', type=int,
        help='only generate snapshots every n iterations',
        default=1000)
parser.add_argument('--inpaint', dest='inpaint', default=False,
        action='store_true', help='Pure inpaint test, z unchanged.')
args = parser.parse_args()

variant = None
if args.w_layers is None and args.a_layers is None:
    variant = ''
    args.w_layers = ['layer6', 'layer8']
    args.a_layers = ['layer10', 'layer12']
if variant is None:
    variant = (
            '_' +
            '_'.join([k.replace('layer', 'w') for k in args.w_layers]) +
            '_' +
            '_'.join([k.replace('layer', 'a') for k in args.a_layers]))
if len(args.w_layers) == 0:
    args.w_steps = 0
if len(args.a_layers) == 0:
    args.a_steps = 0


# lr_milestones = [800, 1200, 1800]
global_seed = 1
w_learning_rate = args.wlr
a_learning_rate = args.alr
image_number = args.image_number
expgroup = 'combo' + variant
imagetypecode = (dict(val='i', train='n', gan='z', test='t')
        .get(args.image_source, args.image_source[0]))
expname = 'opt_%s_%d' % (imagetypecode, image_number)
expdir = os.path.join('results', args.model, expgroup, 'cases', expname)
sumdir = os.path.join('results', args.model, expgroup,
        'summary_%s' % imagetypecode)
os.makedirs(expdir, exist_ok=True)
os.makedirs(sumdir, exist_ok=True)

# First load single image optimize, result of optimize_residual
# Then overfit model layers 5-15 to the specific image.
# Then apply edit.

def main():
    verbose_progress(True)
    progress = default_progress()
    print_progress('Running %s' % expdir)
    delete_log()

    # Grab a target image
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    loaded_image, loaded_z = load_images([image_number],
            args.image_source, model=args.model)
    visualize_results((image_number, 0, 'target'),
            loaded_image[0], summarize=True)

    # Get the learned representation at r4
    rep = imagedata.load_representation(image_number, image_type=imagetypecode,
            model=args.model)

    # Load editor and load the intervention.
    # set squash_units to 418 for church model
    squash_units = imagedata.load_squash_units(args.model)
    editor = edit_algs.BaseEditAlgorithm(model=args.model, num_units=512,
            squash_units=squash_units)
    iv = [] if not args.check_edit else (
            imagedata.load_edit_intervention(image_number, model=args.model))
    iv_seg = editor.merge_edit_layers(iv)
    edit_mask = (torch.from_numpy(iv_seg) > 0).max(1, keepdim=True
            )[0].float().cuda()
    edit_mask_list = [edit_mask]
    for i in range(3):
        edit_mask = torch.nn.functional.interpolate(edit_mask,
                scale_factor=0.5, mode='bilinear', align_corners=False)
        edit_mask_list.append(edit_mask)
    edit_mask = edit_mask_list[0]

    # Visualize the edit region
    yellow = torch.tensor([1, 1, 0])[None,:,None,None].float().cuda()
    orig_x = loaded_image[0].cuda()
    visualize_results((image_number, 0, 'edit'),
        orig_x * (1 - edit_mask) +
        yellow * edit_mask,
        summarize=True)

    # Load the pretrained generator model.
    gan_generator = nethook.InstrumentedSequence(
            imagedata.load_generator(args.model))

    # We will work on overfitting the fine-grained layers
    original_F = gan_generator.subsequence(first_layer='layer5')
    overfit_F = copy.deepcopy(original_F)

    # Also make a conv features model from pretrained VGG
    vgg = models.vgg16(pretrained=True)
    VF = subsequence.Subsequence(vgg.features, last_layer='20')

    # Move models and data to GPU
    for m in [original_F, overfit_F, VF]:
        m.cuda()

    # Some constants for the GPU
    with torch.no_grad():
        # Our starting r is constant
        init_r = rep['r4'].clone().cuda()
        # Our true image is constant
        true_x = loaded_image.cuda()
        # Compute our features once!
        true_v = VF(true_x)

    # Visualize baseline edit
    if args.check_edit:
        with torch.no_grad():
            visualize_results(
                    (image_number, 0, 'initedit'),
                    original_F(editor.apply_edit_to_layer4(iv, init_r)),
                    summarize=True)

    # Set up optimizer
    set_requires_grad(False, original_F, overfit_F, VF)
    parameters = OrderedDict(
            (n, p) for n, p in overfit_F.named_parameters()
            if any(lowlayer in n for lowlayer in args.w_layers))
    for n, p in parameters.items():
        p.requires_grad = True
    w_optimizer = (torch.optim.Adam(parameters.values(), lr=w_learning_rate)
            if parameters else None)
    # w_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         w_optimizer, milestones=lr_milestones, gamma=0.5)

    # Phase 1: Tune the weights of the overfit_F network.
    for step_num in progress(range(args.w_steps + 1)):
        current_x = overfit_F(init_r)
        current_v = VF(current_x)

        loss_x = l1_loss(true_x, current_x)
        loss_v = l1_loss(true_v, current_v)
        # Ordinary regularizer: cannot drift far from the initial weights
        loss_theta = sum(l1_loss(param, init_param)
                for param, init_param
                in zip(overfit_F.parameters(), original_F.parameters()))
        loss = (loss_x + loss_v + loss_theta)

        all_loss = dict(loss=loss, loss_v=loss_v, loss_x=loss_x,
                loss_theta=loss_theta)
        all_loss = { k: v.item() for k, v in all_loss.items()
                if v is not 0 }

        if (step_num % args.snapshot_every == 0) or (step_num == args.w_steps):
            with torch.no_grad():
                psnr = skimage.measure.compare_psnr(
                        numpy_from_tensor(true_x),
                        numpy_from_tensor(current_x))
                all_loss['err_x'] = (current_x - true_x).pow(2).mean() * 100
                all_loss['psnr'] = psnr
                log_progress('%d ' % step_num + ' '.join(
                    '%s=%.3f' % (k, all_loss[k])
                    for k in sorted(all_loss.keys())), phase='w')
                visualize_results((image_number, '2w', step_num), current_x,
                        summarize=(step_num in [0, args.w_steps]))
                visualize_results(
                    (image_number, '2w', step_num, 'res'),
                    10 * (current_x - true_x))
                checkpoint_dict = OrderedDict(all_loss)
                if args.check_edit:
                    visualize_results(
                            (image_number, '2w', step_num, 'edit'),
                            overfit_F(editor.apply_edit_to_layer4(iv, init_r)),
                            summarize=(step_num in [0, args.w_steps]))
            save_checkpoint(
                phase='w',
                step=step_num,
                current_x=current_x,
                true_x=true_x,
                lr=w_learning_rate,
                optimizer=w_optimizer.state_dict() if w_optimizer else None,
                generator=overfit_F.state_dict(),
                **checkpoint_dict)

        if step_num < args.w_steps and w_optimizer:
            w_optimizer.zero_grad()
            loss.backward()
            w_optimizer.step()
            # w_scheduler.step()

    # Phase 2: tune fine-grained activations to perfect the image
    tuned_F = encoder_net.BaselineTunedDirectGenerator(overfit_F, init_r,
            tune_layers=args.a_layers)
    tuned_F.cuda()
    a_parameters = [p for n, p in tuned_F.named_parameters()
            if n.startswith('d')]
    set_requires_grad(False, tuned_F)
    set_requires_grad(True, *a_parameters)
    a_optimizer = (torch.optim.Adam(a_parameters, lr=a_learning_rate)
            if a_parameters else None)
    # a_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     a_optimizer, milestones=act_lr_milestones, gamma=0.1)
    for step_num in progress(range(args.a_steps + 1)):
        out = easydict.EasyDict(tuned_F())
        current_x = out.x
        current_v = VF(current_x)
        loss_x = l1_loss(true_x, current_x) # Pixels should match
        loss_v = 0 # l1_loss(true_v, current_v) # VGG features should match
        loss_d = OrderedDict([('loss_%s' % n, v.pow(2).mean())
            for n, v in out.items() if n.startswith('d')])
        loss = loss_x + loss_v + sum(loss_d.values())
        all_loss = dict(loss=loss, loss_v=loss_v, loss_x=loss_x, **loss_d)
        all_loss = { k: v.item() for k, v in all_loss.items() if v is not 0 }

        if (step_num % args.snapshot_every == 0) or (step_num == args.a_steps):
            with torch.no_grad():
                psnr = skimage.measure.compare_psnr(
                        numpy_from_tensor(true_x),
                        numpy_from_tensor(current_x))
                all_loss['err_x'] = (current_x - true_x).pow(2).mean() * 100
                all_loss['psnr'] = psnr
                log_progress('%d ' % step_num + ' '.join(
                    '%s=%.3f' % (k, all_loss[k])
                    for k in sorted(all_loss.keys())), phase='a')
                visualize_results((image_number, '3a', step_num), current_x,
                        summarize=(step_num in [0, args.a_steps]))
                visualize_results(
                        (image_number, '3a', step_num, 'res'),
                        10 * (current_x - true_x))
                if iv:
                    visualize_results(
                        (image_number, '3a', step_num, 'edit'), editor.edit(
                        iv, dict(r4=init_r), None,
                        generator=tuned_F)['x'])
                    masked_d = apply_mask(edit_mask_list, tuned_F)
                    visualize_results(
                        (image_number, '3a', step_num, 'masked'), editor.edit(
                        iv, dict(r4=init_r), None,
                        generator=lambda z: tuned_F(z, **masked_d))['x'],
                        summarize=(step_num == args.a_steps))
                    visualize_results(
                        (image_number, '3a', step_num, 'maskonly'),
                        tuned_F(init_r, **masked_d)['x'],
                        summarize=(step_num == args.a_steps))
            save_checkpoint(
                phase='a',
                step=step_num,
                current_x=current_x,
                true_x=true_x,
                lr=a_learning_rate,
                optimizer=a_optimizer.state_dict(),
                generator=tuned_F.state_dict(),
                **all_loss)
        if step_num < args.a_steps:
            a_optimizer.zero_grad()
            loss.backward()
            a_optimizer.step()
            # a_scheduler.step()

def apply_mask(edit_mask_list, tuned_F):
    with torch.no_grad():
        masked_d = {}
        for n, p in tuned_F.named_parameters():
            if not n.startswith('d') or n == 'dz':
                continue
            layernum = int(n[1:])
            numscale = (14 - layernum) // 2
            masked_d[n] = p * (1 - edit_mask_list[numscale])
    return masked_d

def delete_log():
    try:
        os.remove(os.path.join(expdir, 'log.txt'))
    except:
        pass

def log_progress(s, phase='o'):
    with open(os.path.join(expdir, 'log.txt'), 'a') as f:
        f.write(phase + ' ' + s + '\n')
    print_progress(s)

def save_checkpoint(**kwargs):
    dirname = os.path.join(expdir, 'snapshots')
    os.makedirs(dirname, exist_ok=True)
    filename = 'step_%s_%d.pth.tar' % (kwargs['phase'], kwargs['step'])
    torch.save(kwargs, os.path.join(dirname, filename))
    # Also save as .mat file for analysis.
    if False:
        numeric_data = {
                k: v.detach().cpu().numpy()
                     if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
                if isinstance(v, (Number, numpy.ndarray, torch.Tensor))}
        filename = 'step_%s_%d.mat' % (kwargs['phase'], kwargs['step'])
        savemat(os.path.join(dirname, filename), numeric_data)

def visualize_results(step, img, summarize=False):
    # TODO: add editing etc.
    if isinstance(step, tuple):
        filename = '%s.png' % ('_'.join(str(i) for i in step))
    else:
        filename = '%s.png' % str(step)
    dirname = os.path.join(expdir, 'images')
    os.makedirs(dirname, exist_ok=True)
    save_tensor_image(img, os.path.join(dirname, filename))
    lbname = os.path.join(dirname, '+lightbox.html')
    if not os.path.exists(lbname):
        shutil.copy('encoder/lightbox.html', lbname)
    if summarize:
        save_tensor_image(img, os.path.join(sumdir, filename))
        lbname = os.path.join(sumdir, '+lightbox.html')
        if not os.path.exists(lbname):
            shutil.copy('encoder/lightbox.html', lbname)

def save_tensor_image(img, filename):
    if len(img.shape) == 4:
        img = img[0]
    np_data = ((img.permute(1, 2, 0) / 2 + 0.5) * 255
            ).clamp(0, 255).byte().cpu().numpy()
    PIL.Image.fromarray(np_data).save(filename)

def set_requires_grad(requires_grad, *models):
    for model in models:
        if model is not None:
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    param.requires_grad = requires_grad
            else:
                model.requires_grad = requires_grad

#unit_level99 = {}
#for cls in ablation_units:
#    corpus = numpy.load('reltest/churchoutdoor/layer4/ace/%s/corpus.npz' % cls)


if __name__ == '__main__':
    exit_if_job_done(expdir, redo=args.redo)
    main()
    mark_job_done(expdir)
