"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

#total_seen 8192/131072/163840/172032/180224不行196608不行524288不行

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

'''seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}'''
#seg_classes = {'SpaceV': [0, 1, 2], 'PlaneV': [3, 4], 'SpaceLap': [5, 6], 'PlaneLap': [7, 8], 'Fillet': [9, 10]}
seg_classes = {'SpaceV': [0, 1], 'SpaceLap': [2, 3], 'Fillet': [4, 5]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root = root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    '''num_classes = 16
    num_part = 50'''
    #num_classes = 5
    #num_part = 11
    num_classes = 3
    num_part = 6

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
        #seg_label_to_cat {0: 'SpaceV', 1: 'SpaceV', 2: 'SpaceV', 3: 'PlaneV', 4: 'PlaneV', 5: 'SpaceLap', 6: 'SpaceLap', 7: 'PlaneLap', 8: 'PlaneLap', 9: 'Fillet', 10: 'Fillet'}
        '''print('seg_label_to_cat',seg_label_to_cat)
        print('seg_label_to_cat[0]', seg_label_to_cat[0])
        print('seg_label_to_cat[1]', seg_label_to_cat[1])
        print('seg_label_to_cat[2]', seg_label_to_cat[2])
        print('seg_label_to_cat[3]', seg_label_to_cat[3])
        '''
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            batchsize, num_point, _ = points.size()#num_point 8192,batchsize 1
            #print('batchsize',batchsize)
            #print('num_point', num_point)
            cur_batch_size, NUM_POINT, _ = points.size()#NUM_POINT 8192,cur_batch_size 1
            #print('cur_batch_size',cur_batch_size)
            #print('NUM_POINT', NUM_POINT)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            #points:点，label：分类，target：部件分类
            #print('points', points) #points tensor([[[-0.1192,  0.1214,  0.0355],
            #print('label', label)#label tensor([[3]], device='cuda:0')
            #print('target', target)#target tensor([[8, 8, 8,  ..., 8, 8, 8]], device='cuda:0')
            points = points.transpose(2, 1)
            #print('points', points)
            '''points tensor([[[-0.0265,  0.3001, -0.2050,  ..., -0.3275,  0.1698, -0.4013],
            [-0.1913, -0.0278, -0.0694,  ...,  0.2017, -0.2049, -0.0537],
            [ 0.3327,  0.3870,  0.3680,  ...,  0.4391,  0.3443,  0.4406]]],'''
            classifier = classifier.eval()
            #print('classifier',classifier)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            #print('vote_pool',vote_pool)
            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred
            '''print('seg_pred', seg_pred)
            print(seg_pred.shape)#torch.Size([1, 8192, 11])
            print('vote_pool', vote_pool)
            print(vote_pool.shape)#torch.Size([1, 8192, 11])
            print('args.num_votes', args.num_votes)
            '''
            seg_pred = vote_pool / args.num_votes#3
            #print('seg_pred1', seg_pred)
            cur_pred_val = seg_pred.cpu().data.numpy()
            #print('cur_pred_val', cur_pred_val)
            #print(cur_pred_val.shape)
            cur_pred_val_logits = cur_pred_val
            #print('cur_pred_val_logits', cur_pred_val_logits)
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            #print('cur_pred_val', cur_pred_val)#cur_pred_val [[0 0 0 ... 0 0 0]]
            #print(cur_pred_val.shape)#(1, 8192)
            target = target.cpu().data.numpy()
            #print('target', target)#target [[1 1 2 ... 1 1 1]]
            #print(target.shape)#(1, 8192)
            #print('range(cur_batch_size)',range(cur_batch_size))#range(0, 1)
            print('target[0, 0]',target[0, 0])#target[0, 0] 1
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                print('cat', cat)#cat SpaceV
                logits = cur_pred_val_logits[i, :, :]
                print('logits', logits)
                print('logits.shape', logits.shape)
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                print('cur_pred_val', cur_pred_val)#cur_pred_val [[1 1 2 ... 2 1 2]]#预测标志
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            print('total_correct', total_correct)# total_correct 4951/76488/
            total_seen += (cur_batch_size * NUM_POINT)
            print('total_seen', total_seen)#total_seen 8192/131072/

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


    log_string('Accuracy is: %.5f'%test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f'%test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f'%test_metrics['inctance_avg_iou'])

if __name__ == '__main__':
    args = parse_args()
    print(__file__)
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    main(args)

