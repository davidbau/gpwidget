import torch, torchvision, os
from . import parallelfolder

def load_proggan(domain):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)
    from . import proggan
    weights_filename = dict(
        bedroom='proggan_bedroom-d8a89ff1.pth',
        church='proggan_churchoutdoor-7e701dd5.pth',
        conferenceroom='proggan_conferenceroom-21e85882.pth',
        diningroom='proggan_diningroom-3aa0ab80.pth',
        kitchen='proggan_kitchen-67f1e16c.pth',
        livingroom='proggan_livingroom-5ef336dd.pth',
        restaurant='proggan_restaurant-b8578299.pth')[domain]
    # Posted here.
    url = 'http://gandissect.csail.mit.edu/models/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = proggan.from_state_dict(sd)
    return model

g_datasets = {}

def load_dataset(domain, download=True):
    if domain in g_datasets:
        return g_datasets[domain]
    dirname = os.path.join('datasets', 'minilsun', domain)
    if download and not os.path.exists(dirname):
        torchvision.datasets.utils.download_and_extract_archive(
                'http://gandissect.csail.mit.edu/datasets/minilsun.zip',
                'datasets',
                md5='a67a898673a559db95601314b9b51cd5')
    return parallelfolder.ParallelImageFolders([dirname])

g_transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_image(domain, imgnum):
    return g_transform(load_dataset(domain)[imgnum][0])

if __name__ == '__main__':
    main()

