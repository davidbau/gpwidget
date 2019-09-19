import torch, numpy, numbers, os, shutil, PIL, argparse
from torch.nn.functional import mse_loss, l1_loss
from encoder.progress import default_progress, print_progress, verbose_progress
from encoder.progress import post_progress, desc_progress
from encoder.imagedata import save_tensor_image, load_greenmask
from encoder import zdataset, seededsampler, zipdataset
from encoder import proggan, customnet, parallelfolder, easydict
from torchvision import transforms, models
from torchvision.models.vgg import model_urls
from encoder.pidfile import exit_if_job_done, mark_job_done
from encoder import encoder_net, imagedata, subsequence, edit_algs, nethook
from encoder.encoder_loss import cor_square_error
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.io import savemat
from encoder.lpb import laplacian_pyramid_blend
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate', default=0.1)
parser.add_argument('--image_number', type=int, help='Image number',
        default=133)
parser.add_argument('--image_source',
        default='train')
parser.add_argument('--residuals', nargs='*', help='Residuals to adjust',
        default=['layer1', 'layer2'])
parser.add_argument('--redo', type=int, help='Nonzero to delete done.txt',
        default=0)
parser.add_argument('--style', type=int, help='Index of sytle image (needs to be one of the 1000 images with .mat files). '
                                              'Leave at -1 to disable.',
        default=-1)
parser.add_argument('--style_str', type=float, help='Weight of styled units',
        default=1.0)
parser.add_argument('--model', type=str, help='Dataset being modeled',
        default='church')
parser.add_argument('--use_noise', action='store_true', help='Inject noise into editing process')
parser.add_argument('--inj_mode', type=str, help='Method of injecting noise [add|mul|strength_add]',
        default='add')
parser.add_argument('--seed', type=int, help='Seed for noise injection (leave at 0 for new random seed)',
        default=0)
parser.add_argument('--edit_strength', type=float,
        help='Strength of edit in loss', default=1.0)
parser.add_argument('--manifold_strength', type=float,
        help='Strength of gan manifold in loss', default=10.0)
parser.add_argument('--noise_penalty', type=float,
        help='Penalty for adding noise', default=0.1)
parser.add_argument('--snapshot_every', type=int, default=100)
parser.add_argument('--vgg_update_freq', type=int, default=10000)
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--quiet', dest='quiet', default=False, action='store_true')
parser.add_argument('--noedit', dest='noedit', default=False,
        action='store_true', help='Pure reconstruction test.')
parser.add_argument('--inpaint', dest='inpaint', default=False,
        action='store_true', help='Pure inpaint test, z unchanged.')
parser.add_argument('--expgroup', default=None)
args = parser.parse_args()

verbose_progress(not args.quiet)

image_number = args.image_number
global_seed = 1
torch.manual_seed(global_seed)
learning_rate = args.lr
lr_milestones = []
seed = 0
if args.expgroup is not None:
    expgroup = args.expgroup
elif args.inpaint:
    args.edit_strength = 0.0
    args.manifold_strength = 1000.0
    expgroup = 'optimize_fixed_gp_inpaint'
elif args.noedit:
    args.edit_strength = 0.0
    expgroup = 'optimize_fixed_gp_noedit'
else:
    expgroup = 'optimize_fixed_ganprior'
imagetypecode = imagedata.imagetypecode(args.image_source)
expname = 'opt_%s_%d_%d' % (imagetypecode, image_number, seed)
expdir = os.path.join('results', args.model, expgroup, 'cases', expname)
sumdir = os.path.join('results', args.model, expgroup,
        'summary_%s' % imagetypecode)
os.makedirs(expdir, exist_ok=True)
os.makedirs(sumdir, exist_ok=True)

# This code combines the optimize_realedit ideas with
# the optimize_r_ganprior ideas.

def main():
    progress = default_progress()
    print_progress('Running %s' % expdir)
    delete_log()

    # Grab a target image, an edit mask, and an edit
    editor = edit_algs.BaseEditAlgorithm(model=args.model, num_units=512)
    if args.noedit:
        iv = []
    else:
        iv = imagedata.load_edit_intervention(image_number,
           'editwild' if args.image_source == 'editwild' else args.model)
    uned_rep = imagedata.load_representation(image_number, model=args.model,
            image_type=imagetypecode)
    uned_img = imagedata.load_images(
            [image_number], image_source=args.image_source, model=args.model)[0]
    uned_img = uned_img.cuda()
    # For masking pixel loss
    seg = editor.merge_edit_layers(iv)

    # Visualization of edit mask, no border
    yellow = torch.tensor([1, 1, 0])[None,:,None,None].float().cuda()
    sharp_edit_mask = (torch.from_numpy(seg) > 0).max(1, keepdim=True
            )[0].float().cuda()
    visualize_results((image_number, 'a_uned_sm'),
        uned_img * (1 - sharp_edit_mask) +
        yellow * sharp_edit_mask,
        summarize=True)

    inflate_mask = not args.inpaint
    if inflate_mask:
        dilated_mask = gaussian_filter(
                binary_dilation((seg > 0).any(axis=1), iterations=16).
                astype(numpy.float), sigma=4)
        edit_mask = (torch.from_numpy(dilated_mask).float().cuda())
    else:
        edit_mask = sharp_edit_mask

    # Inflate this mask
    keep_mask = 1 - edit_mask
    # For masking VGG features
    vgg_keep_mask = torch.nn.functional.adaptive_avg_pool2d(
            keep_mask, (32, 32))
    # For masking target
    r4_edit_mask = 1 - torch.nn.functional.adaptive_avg_pool2d(
            keep_mask, (8, 8))
    masked_uned_img = uned_img * keep_mask
    visualize_results((image_number, 'a_uned'), uned_img[0],
            summarize=True)
    visualize_results((image_number, 'a_uned_m'), masked_uned_img[0],
            summarize=True)

    # Load the pretrained generator model.
    unwrapped_G = imagedata.load_generator(model=args.model)
    # Get the lower levels H.
    H = subsequence.Subsequence(unwrapped_G, last_layer='layer4')
    # Get the higher levels F.
    F = subsequence.Subsequence(unwrapped_G, first_layer='layer5')

    # Load a pretrained gan inverter
    encoder = imagedata.load_hybrid_inverter(model=args.model)
    E = encoder.subsequence(last_layer='resnet')
    D = encoder.subsequence(first_layer='inv4')

    # Also make a conv features model from pretrained VGG
    vgg = models.vgg16(pretrained=True)
    VF = subsequence.Subsequence(vgg.features, last_layer='20')

    # Move models and data to GPU
    for m in [H, F, E, D, VF]:
        m.cuda()

    # Load the baseline editing model, and perform the edit, and make a seed z
    with torch.no_grad():
        test_rep = imagedata.load_representation(image_number, model=args.model)
        test_orig = imagedata.load_images([image_number], model=args.model)[0]
        test_edit = editor.edit(iv, test_rep, test_orig, None)
        visualize_results(
                (image_number, 'b_tested_r4'),
                test_edit, summarize=True)
        visualize_results(
                (image_number, 'b_uned_r4'),
                F(uned_rep['r4']), summarize=True)
        uned_r4 = uned_rep['r4']
        edit_seg = editor.merge_edit_layers(iv)
        mask_r4, sum_paint_r4 = editor.sum_edit_features(edit_seg)
        # Testing bad_units - notes.
        # Got squarer and emptier with: 279;
        # A little bigger with 236
        # bigger with: 418 definitely; 443, 288
        # (If inpainting without editing, do not edit any channels.)
        bad_units = [418] if not args.inpaint else slice(None)
        mask_r4[:,bad_units,:,:] = 0      # Try eliminating this channel.
        sum_paint_r4[:,bad_units,:,:] = 0 # Try eliminating this channel.

        mask_r4, sum_paint_r4 = [d.cuda() for d in [mask_r4, sum_paint_r4]]
        # Pull this stuff apart
        # sum_paint_r4 *= 3
        init_r4 = (1 - mask_r4) * uned_r4 + sum_paint_r4
        paint_r4 = sum_paint_r4 / (mask_r4 + 1e-12)

        init_r4_image = F(init_r4)
        visualize_results(
                (image_number, 'c_init_r4'),
                init_r4_image, summarize=True)
        # Also make the initial guess for VGG features based on the masked img
        # TODO: goal_v should keep on being adjusted during training for
        # better realism.
        goal_v_img = (masked_uned_img + init_r4_image * (1 - keep_mask))
        goal_v = VF(goal_v_img)
        uned_v = VF(uned_img) # For benchmarking inpainting and reconst.
        visualize_results((image_number, 'd_init_gv'), goal_v_img)

    # Make the u-shaped random generator model.
    # TODO: maybe unwrapped_F can be pre-trained for each image (sans edits)
    # for faster convergence?
    generator = encoder_net.FixedGANPriorGenerator(F, init_r4)
    generator.cuda()

    # Set up optimizer
    parameters = []
    # For now, let's only learn top-level parameters
    for n, p in generator.named_parameters():
        # Here are the trainable parameters:
        # d_N for the z-derived bottom layers.
        # and downN for the noise-generated layers.
        if n.startswith('down'): # down_6, down_12.
            parameters.append(p)
        elif n.startswith('d') and not args.noedit: # d1, d2
            parameters.append(p)
        else:
            p.requires_grad = False
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.1)

    # Do the training now.
    for step_num in progress(range(args.num_steps + 1)):

        out = easydict.EasyDict(generator()) # make some pixels!
        v = VF(out.x) # VGG features of generated pixels

        # To steer the GAN dragon, we tie it down from head to tail.
        loss_8 = out.s8.pow(2).mean()
        loss_10 = out.s10.pow(2).mean()
        loss_12 = out.s12.pow(2).mean()
        loss_14 = out.s14.pow(2).mean()
        loss_p = l1_loss(out.x * keep_mask, uned_img * keep_mask)
        loss_v = l1_loss(v * vgg_keep_mask, goal_v * vgg_keep_mask)
        loss = (args.noise_penalty * (
                    loss_8 + loss_10 + loss_12 + loss_14)
                + loss_p
                # + loss_v
                )

        if step_num % args.snapshot_every == 0 or step_num == args.num_steps:
            # For logging, pull these off the GPU.
            report = {k: v.item() for k, v in dict(
                    loss=loss,
                    loss_8=loss_8,
                    loss_10=loss_10,
                    loss_12=loss_12,
                    loss_14=loss_14,
                    loss_p=loss_p,
                    loss_v=loss_v,
                    # Reconstruction benchmark
                    recon_p=(l1_loss(out.x, uned_img)),
                    recon_v=(l1_loss(v, uned_v)),
                    # Inpainting benchmark
                    inpaint_p=(l1_loss(out.x * (1 - keep_mask),
                                   uned_img * (1 - keep_mask))),
                    inpaint_v=(l1_loss(v * (1 - vgg_keep_mask),
                                   uned_v * (1 - vgg_keep_mask))),
                    ).items()}
            log_progress('i %d ' % step_num + ' '.join(
                '%s=%.3f' % (k, report[k]) for k in sorted(report.keys())))
            visualize_results(
                    (image_number, 'i', step_num),
                    out.x,
                    summarize=(step_num in [0, args.num_steps]))
            visualize_results(
                (image_number, 'i', step_num, 'comp'),
                out.x * (1 - keep_mask) + uned_img * keep_mask,
                summarize=(step_num in [0, args.num_steps]))
            # lpb_blend = laplacian_pyramid_blend(out.x, uned_img, keep_mask)
            # visualize_results(
            #     (image_number, 'i', step_num, 'lpb'),
            #     lpb_blend,
            #    summarize=(step_num in [0, args.num_steps]))
            visualize_results(
                    (image_number, 'i', step_num, 'res'),
                    10 * (out.x - uned_img),
                    summarize=(step_num in [0, args.num_steps]))
            # For detailed snapshots, record these tensors
            with torch.no_grad():
                report['init_z'] = generator.init_z
                for n in [8, 10, 12, 14]:
                    initval = getattr(generator, 'init_%d' % n)
                    report['init_%d' % n] = initval
                    report['current_%d' % n] = (
                        initval * (1 + out['s%d' % n])) # n in [6-14]
                    if n in [8, 10, 12, 14]:
                        report['s%d' % n] = out['s%d' % n]
                report.update(out)
                report.update(dict(
                    paint_r4=paint_r4,
                    mask_r4=mask_r4,
                    keep_mask=keep_mask,
                    v=v,
                    goal_v=goal_v,
                    vgg_keep_mask=vgg_keep_mask
                ))
            save_checkpoint(
                phase='i',
                step=step_num,
                lr=learning_rate,
                optimizer=optimizer.state_dict(),
                **report)
        # Take a step
        optimizer.zero_grad()
        loss.backward()
        if step_num < args.num_steps:
            optimizer.step()
            scheduler.step()

        # Once in a while, update goal_v by pasting in current output
        if step_num > 0 and step_num % args.vgg_update_freq == 0:
            with torch.no_grad():
                goal_v_img = masked_uned_img + out.x * (1 - keep_mask)
                goal_v = VF(goal_v_img)
            visualize_results((image_number, 'i', step_num, 'gv'), goal_v_img)

def delete_log():
    try:
        os.remove(os.path.join(expdir, 'log.txt'))
    except:
        pass

def log_progress(s):
    with open(os.path.join(expdir, 'log.txt'), 'a') as f:
        f.write(s + '\n')
    print_progress(s)

def save_checkpoint(**kwargs):
    dirname = os.path.join(expdir, 'snapshots')
    os.makedirs(dirname, exist_ok=True)
    filename = 'step_%s_%d.pth.tar' % (kwargs['phase'], kwargs['step'])
    torch.save(kwargs, os.path.join(dirname, filename))
    # Also save as .mat file for analysis.
    numeric_data = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if isinstance(v, (numbers.Number, numpy.ndarray, torch.Tensor))}
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
            for param in model.parameters():
                param.requires_grad = requires_grad

if __name__ == '__main__':
    exit_if_job_done(expdir, redo=args.redo)
    main()
    mark_job_done(expdir)
