#!/usr/bin/env python

import argparse, connexion, os, sys, yaml, json, socket
from flask import send_from_directory, redirect
from flask_cors import CORS

from encoder.easydict import EasyDict
from encoder.serverstate import GANPaintProject

__author__ = 'Hendrik Strobelt, David Bau'

# TODO: decide on a project config
CONFIG_FILE_NAME = 'gan_encode.yaml'
projects = {}

app = connexion.App(__name__, debug=False)


def get_all_projects():
    res = []
    for key, project in projects.items():  # type: str,GANPaintProject
        # print key
        res.append({
            'project': key,
            'info': {
                'features': project.features,
                'image_ids': project.image_numbers
            }
        })
    return sorted(res, key=lambda x: x['project'])


def post_generate(gen_req):
    project = gen_req['project']
    ids = gen_req.get('ids')
    interpolations = gen_req.get('interpolations', [1])
    interventions = gen_req.get('interventions', None)
    save = gen_req.get('save', False)
    generated = projects[project].generate_images(ids, interventions,
                                                  interpolations, save=save)
    return {
        'request': gen_req,
        'res': generated
    }

def post_upload(upload_req):
    project = upload_req['project']
    image_data = upload_req['image']
    imageid = projects[project].upload_image(image_data)
    return {
        'res': {
            'id': imageid
        }
    }

@app.route('/client/<path:path>')
def send_static(path):
    """ serves all files from ./client/ to ``/client/<path:path>``

    :param path: path from api call
    """
    return send_from_directory(args.client, path)


@app.route('/data/<path:path>')
def send_data(path):
    """ serves all files from the data dir to ``/dissect/<path:path>``

    :param path: path from api call
    """
    print('Got the data route for', path)
    return send_from_directory(args.data, path)


@app.route('/')
def redirect_home():
    return redirect('/client/ganpaint.html', code=302)


def load_projects(directory, cachedir):
    """
    searches for CONFIG_FILE_NAME in all subdirectories of directory
    and creates data handlers for all of them

    :param directory: scan directory
    :return: null
    """
    project_dirs = []
    # Don't search more than 2 dirs deep.
    search_depth = 2 + directory.count(os.path.sep)
    for root, dirs, files in os.walk(directory):
        if CONFIG_FILE_NAME in files:
            project_dirs.append(root)
            # Don't get subprojects under a project dir.
            del dirs[:]
        elif root.count(os.path.sep) >= search_depth:
            del dirs[:]
    for p_dir in project_dirs:
        print('Loading %s' % os.path.join(p_dir, CONFIG_FILE_NAME))
        with open(os.path.join(p_dir, CONFIG_FILE_NAME), 'r') as jf:
            # config = EasyDict(json.load(jf))
            config = yaml.load(jf)
            dh_id = os.path.split(p_dir)[1]
            projects[dh_id] = GANPaintProject(
                config=config,
                project_dir=p_dir,
                path_url='data/' + os.path.relpath(p_dir, directory),
                public_host=args.public_host,
                cachedir=cachedir)


app.add_api('server.yaml')

# add CORS support
CORS(app.app, headers='Content-Type')

parser = argparse.ArgumentParser()
parser.add_argument("--nodebug", default=False)
parser.add_argument("--address",
                    default="127.0.0.1")  # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--public_host", default=None)
parser.add_argument("--nocache", default=False)
parser.add_argument("--data", type=str, default='serverdata')
parser.add_argument("--cachedir", type=str, default='cache')
parser.add_argument("--client", type=str, default='paint/dist')

if __name__ == '__main__':
    args = parser.parse_args()
    # for d in [args.data, args.client]:
    #     if not os.path.isdir(d):
    #         print('No directory %s' % os.path.abspath(d))
    #         sys.exit(1)
    args.data = os.path.abspath(args.data)
    args.cachedir = os.path.abspath(args.cachedir)
    args.client = os.path.abspath(args.client)
    if args.public_host is None:
        args.public_host = '%s:%d' % (socket.getfqdn(), int(args.port))
    app.run(port=int(args.port), debug=not args.nodebug, host=args.address,
            use_reloader=False)
else:
    args, _ = parser.parse_known_args()
    if args.public_host is None:
        args.public_host = '%s:%d' % (socket.getfqdn(), int(args.port))
    load_projects(args.data, args.cachedir)
