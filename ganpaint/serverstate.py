import os, torch, numpy, base64, json, re, threading, random, collections
from PIL import Image
from io import BytesIO
from encoder import imagedata, subsequence, edit_algs, segviz
import time
from encoder.fit_image_model import create_generator, load_cached_generator


def img_to_base64(imgarray, for_html=True, image_format='jpeg'):
    """
    Converts a numpy array to a jpeg base64 url
    """
    input_image_buff = BytesIO()
    Image.fromarray(imgarray).save(input_image_buff, image_format,
                                   quality=99, optimize=True, progressive=True)
    res = base64.b64encode(input_image_buff.getvalue()).decode('ascii')
    if for_html:
        return 'data:image/' + image_format + ';base64,' + res
    else:
        return res

def base64_to_pil(string_data):
    string_data = re.sub('^(?:data:)?image/\\w+;base64,', '', string_data)
    return Image.open(BytesIO(base64.b64decode(string_data)))

def pil_to_base64(img):
    input_image_buff = BytesIO()
    img.save(input_image_buff, 'png',
             quality=99, optimize=True, progressive=True)
    return 'data:image/png;base64,' + base64. \
        b64encode(input_image_buff.getvalue()).decode('ascii')


# Flip to tru to use ImageTunedEditAlgorithm instead
edit_with_adapted_GAN = False

class GANPaintProject:
    """
    Project description for an encoder model being used in the tool
    """

    def __init__(self, config, project_dir, path_url, public_host, cachedir):
        print('config done', project_dir)
        self.use_cuda = torch.cuda.is_available()
        self.config = config
        self.project_dir = project_dir
        self.cachedir = cachedir
        self.path_url = path_url
        self.public_host = public_host
        self.features = ['-'] + config['features']
        self.image_numbers = config['image_numbers']
        self.image_reps = [
                imagedata.load_representation(i, model=config['model'],
                    image_type='n') for i in self.image_numbers]
        self.original_images = [
                imagedata.load_images([i],
                    model=config['model'])[0]
                for i in self.image_numbers]
        self.editor = edit_algs.BaseEditAlgorithm(config['model'],
            squash_units=imagedata.load_squash_units(config['model']))
        self.memorycache = LRUCache(3) # Save the last 3 image models.

    def upload_image(self, image_str):
        image = base64_to_pil(image_str)
        image_id, generator = create_generator(image, self.config['model'],
                self.cachedir)
        self.memorycache[image_id] = generator
        return image_id

    def cached_generator(self, image_id):
        try:
            return self.memorycache[image_id]
        except KeyError:
            pass
        result = load_cached_generator(
                image_id, self.config['model'], self.cachedir)
        if result is None:
            raise FileNotFoundError()
        self.memorycache[image_id] = result
        return result

    def generate_images(self, image_ids, interventions, interpolations,
                        save=False):
        result = []
        for image_id in image_ids:
            generator = self.cached_generator(image_id)
            rep = dict(r4=generator.init_z)
            orig = None
            seg = self.editor.merge_edit_layers(interventions)
            sharp_edit_mask = (torch.from_numpy(seg) > 0).max(1, keepdim=True
                )[0].float().cuda()
            edit_mask = torch.nn.functional.interpolate(sharp_edit_mask,
                    scale_factor=0.5, mode='bilinear', align_corners=False)
            with torch.no_grad():
                md12 = generator.d12 * (1 - edit_mask)
            masked_generator = lambda z: generator(z, d12=md12)

            out = self.editor.edit(interventions,
                                         rep,
                                         orig,
                                         generator=masked_generator)
            new_image = out['x']
            result.append(dict(
                id=image_id,
                d=pil_to_base64(imagedata.pil_from_tensor(new_image))
            ))
            # For debugging purposes, we can record the last edit.
            if len(interventions) and save:
                self.record_edit(image_id, interventions, new_image)
        return result

    def record_edit(self, imageid, interventions, new_image):
        savedir = os.path.join(self.cachedir, self.config['model'],
                'img_%s' % imageid, 'edit')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        temp = int(round(time.time() * 1000))
        image_num = str(imageid) + '_time' + str(temp)
        json_file_name = os.path.join(
                savedir, '%s_int.json' % image_num)
        with open(json_file_name, 'w') as f:
            json.dump(interventions, f, sort_keys=True, indent=2)
        seg = self.editor.merge_edit_layers(interventions)
        Image.fromarray(segviz.segment_visualization(seg[0],
                                                     seg.shape[
                                                     2:])).save(
            os.path.join(savedir, '%s_seg.png' % image_num))
        imagedata.save_tensor_image(new_image, os.path.join(
            savedir, '%s_base.png' % image_num))

class LRUCache:
    '''
    A simple implementation of an LRU cache that exploits OrderedDict.
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key):
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def __setitem__(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __len__(self):
        return len(self.cache)
