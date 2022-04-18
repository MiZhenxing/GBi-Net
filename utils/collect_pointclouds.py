import os, sys
import argparse
import glob
import errno
import os.path as osp
import shutil

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def collect_dtu(args):
	mkdir_p(args.target_dir)
	all_scenes = sorted(glob.glob(args.root_dir+'/*'))
	all_scenes = list(filter(os.path.isdir, all_scenes))
	print("Found ", len(all_scenes), " scenes")
	for scene in all_scenes:
		# scene_id = int(scene.strip().split('/')[-1][len('scan'):])
		scene_id = int(''.join(i for i in scene.strip().split('/')[-1] if i.isdigit()))
		all_plys = sorted(glob.glob('{}/{}/consistencyCheck*'.format(scene, args.point_dir)))
		print('Found points: ', all_plys)

		shutil.copyfile(all_plys[-1]+'/final3d_model.ply', '{}/binary_{:03d}_l3.ply'.format(args.target_dir, scene_id))

def collect_tanks(args):
	mkdir_p(args.target_dir)
	all_scenes = sorted(glob.glob(args.root_dir + '/*'))
	all_scenes = list(filter(os.path.isdir, all_scenes))
	for scene in all_scenes:
		all_plys = sorted(glob.glob('{}/{}/consistencyCheck*'.format(scene, args.point_dir)))
		print('Found points: ', all_plys)
		scene_name = scene.strip().split('/')[-1]
		try:
			shutil.copyfile(all_plys[-1]+'/final3d_model.ply', osp.join(args.target_dir, '{}.ply'.format(scene_name)))
			shutil.copy(osp.join(args.tanks_log_dir, scene_name, '{}.log'.format(scene_name)),
							osp.join(args.target_dir))
		except:
			print("no {}".format(osp.join(args.tanks_log_dir, scene_name, '{}.log'.format(scene_name))))


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--root_dir', help='path to prediction', type=str,)
	parser.add_argument('--point_dir', type=str)
	parser.add_argument('--target_dir', type=str)
	parser.add_argument('--tanks_log_dir', type=str, default=None)
	parser.add_argument('--dataset', type=str, )

	args = parser.parse_args()
	if args.dataset == 'dtu':
		collect_dtu(args)
	elif args.dataset == 'tanks':
		collect_tanks(args)
	else:
		print('Unknown dataset.')
