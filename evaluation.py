from data.volume import Patient
import SimpleITK as sitk
import os
from PIL import Image
import numpy as np
import copy
import math
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.morphology import binary_erosion, binary_dilation
from skimage.morphology.selem import square
import medpy.metric.binary as tool
from process.utils import CRFs, connected_components


def score(p_dir, data_file, selected_modal, selected_epoch, seg_type):

    pre_dir = []
    for sepoch in selected_epoch:
        pre_dir.append(p_dir + '/' + sepoch)
        print(p_dir + '/' + sepoch, data_file)

    patients = Patient(data_file, selected_modal)
    metrics = ['Dice', 'Jaccard', 'RAVD', 'ASSD', 'HD95', 'HD', 'Sens', 'Spec', 'Prec', 'Recall', 'F1']
    info = {x: {k: 0 for k in metrics}
            for x in selected_modal}
    n_sample = {x: 0 for x in selected_modal}
    modal_pid = {x: [] for x in selected_modal}

    # deleted_id = ['115', '153', '311', '044', '053', '219', '303', '315', '221', '163',
    #                 '328', '348', '358', '369', '117', '155', '182', '317', '097', '110']
    a = [[], [], [], [], [], [], [], [], [], [], []]
    aver = {}
    get_pid = []
    for modal in selected_modal:
        aver[modal] = copy.deepcopy(a)
    mid = {'Dice': 0, 'Jaccard': 1, 'RAVD': 2, 'ASSD': 3, 'HD95': 4, 'HD': 5, 'Sens': 6, 'Spec': 7, 'Prec': 8,
           'Recall': 9, 'F1': 10}
    print_str = 'pre zero -->'
    print_str2 = 'assd zero -->'
    for p in patients:
        # For Each Patient.

        modal = None
        pred_volume = []
        gt_volumn = []
        p_id = None

        for slic, label in p:
            pred = None
            basename = os.path.basename(label)
            p_id = basename.split('_')[0]
            if not modal:
                iamge_dir = os.path.split(slic)[0]
                modal = iamge_dir.split('/')[-1]

            image = Image.open(slic)
            for p_dir in pre_dir:
                if pred is None:
                    pred = np.array(Image.open(os.path.join(p_dir + '/{}'.format(modal), basename)))
                else:
                    pred += np.array(Image.open(os.path.join(p_dir + '/{}'.format(modal), basename)))
            gt = Image.open(label)
            image, gt = np.array(image), np.array(gt)


            pred[pred != 0] = 1
            pred_volume.append(pred[np.newaxis, :, :])
            gt_volumn.append(gt[np.newaxis, :, :])
        # if p_id in deleted_id:
        #    continue
        # Reconstruct to 3D Volume.
        pred_volume = np.concatenate(pred_volume)
        gt_volumn = np.concatenate(gt_volumn)
        new_pred_volumn = connected_components(pred_volume.copy(), seg_type)
        #new_pred_volumn = pred_volume.copy()
        new_pred_volumn[new_pred_volumn != 0] = 1

        new_gt_volumn = gt_volumn.copy()
        if seg_type == 'ET':
            new_gt_volumn[new_gt_volumn == 120] = 0  # 0 60 120 180 240
            new_gt_volumn[new_gt_volumn == 60] = 0
            new_gt_volumn[new_gt_volumn == 180] = 0
        elif seg_type == 'TC':  #
            new_gt_volumn[new_gt_volumn == 120] = 0
        elif seg_type != 'WT':
            print('******************************************error type!!')
        new_gt_volumn[new_gt_volumn != 0] = 1

        if new_gt_volumn.sum() == 0:
            continue
        if new_pred_volumn.sum() == 0:
            print_str += '{}:{}  '.format(modal, p_id)
            continue
        pred = sitk.GetImageFromArray(new_pred_volumn, isVector=False)
        gt = sitk.GetImageFromArray(new_gt_volumn, isVector=False)

        # Calc Overlap Score.
        over_filter = sitk.LabelOverlapMeasuresImageFilter()
        over_filter.Execute(gt, pred)
        dice = over_filter.GetDiceCoefficient()
        jaccard = over_filter.GetJaccardCoefficient()
        #print('{}_{}:\t\t{}'.format(modal,p_id, dice))

        n_sample[modal] += 1
        modal_pid[modal].append(p_id)

        info[modal]['Dice'] += dice
        info[modal]['Jaccard'] += jaccard
        aver[modal][mid['Dice']].append(dice)  # ***************
        aver[modal][mid['Jaccard']].append(jaccard)  # ***************

        ravd = (new_pred_volumn.sum() / new_gt_volumn.sum()) - 1
        ravd = abs(ravd)
        info[modal]['RAVD'] += ravd
        aver[modal][mid['RAVD']].append(ravd)  # ***************
        assd = tool.assd(new_pred_volumn, new_gt_volumn, voxelspacing=(1, 240 / 128, 240 / 128))
        info[modal]['ASSD'] += assd
        aver[modal][mid['ASSD']].append(assd)  # ***************


        flat_pred = np.reshape(new_pred_volumn, (-1))
        flat_gt = np.reshape(new_gt_volumn, (-1))
        prec = precision_score(flat_gt, flat_pred)
        recall = recall_score(flat_gt, flat_pred)
        f1score = f1_score(flat_gt, flat_pred)

        info[modal]['Prec'] += prec
        info[modal]['Recall'] += recall
        info[modal]['F1'] += f1score
        aver[modal][mid['Prec']].append(prec)  # ***************
        aver[modal][mid['Recall']].append(recall)  # ***************
        aver[modal][mid['F1']].append(f1score)  # ***************
        # Calc Distance Score.

        h95 = tool.hd95(new_pred_volumn, new_gt_volumn, voxelspacing=(1, 240 / 128, 240 / 128))
        h = tool.hd(new_pred_volumn, new_gt_volumn, voxelspacing=(1, 240 / 128, 240 / 128))
        info[modal]['HD'] += h
        info[modal]['HD95'] += h95
        aver[modal][mid['HD']].append(h)  # ***************
        aver[modal][mid['HD95']].append(h95)  # ***************
        sens = tool.sensitivity(new_pred_volumn, new_gt_volumn)
        spec = tool.specificity(new_pred_volumn, new_gt_volumn)
        info[modal]['Sens'] += sens
        info[modal]['Spec'] += spec
        aver[modal][mid['Sens']].append(sens)  # ***************
        aver[modal][mid['Spec']].append(spec)  # ***************

        # print(pid, distance, dice, vs, sens, spec, prec, recall, f1score)
        get_pid.append(p_id)
    # print(modal_pid)
    print(print_str)
    print(print_str2)
    # avg_mod = dict(Dice=0, Hausdorff_distance=0, Volume_similarity=0, Sensitivity=0, Specificity=0)
    # avg_sample = dict(Dice=0, Hausdorff_distance=0, Volume_similarity=0, Sensitivity=0, Specificity=0)
    # ak = 0
    allinfo = {}
    selected_metric = ['Dice', 'Jaccard', 'RAVD', 'ASSD', 'HD95', 'HD', 'Sens', 'Spec', 'Prec', 'Recall', 'F1']
    for modal in selected_modal:
        s = info[modal]
        t = aver[modal]
        # print_str = '{}({}):\t'.format(modal, n_sample[modal])
        dic = {}
        for k, v in s.items():
            av = v / n_sample[modal]
            sum = 0

            for p in t[mid[k]]:
                sum += math.pow(p - av, 2)
            # sd = sum/len(t[mid[k]])
            if n_sample[modal] != len(t[mid[k]]):
                print('error')
                break
            variance = sum / n_sample[modal]
            sd = math.sqrt(variance)
            # print_str += '{}: {:.2f}~{:.2f},\t'.format(k,
            #     (av if k in ['HD'] else av*100),
            #     (sd if k in ['HD'] else sd*100))
            # avg_mod[k] += v / n_sample[mod]
            # avg_sample[k] += v
            if k in selected_metric:
                dic[k] = [av, sd]
        allinfo[modal] = dic
        # ak += n_sample[mod]
        # print(print_str)
    sum_sample = 0
    print_str = '\t'
    for modal in selected_modal:
        sum_sample += n_sample[modal]
        print_str += '{}({}):\t'.format(modal, n_sample[modal])
    print_str += 'overall\n'
    for metric in selected_metric:
        aver = 0
        variance = 0
        if metric == 'Jaccard':
            print_str += '{}:'.format(metric)
        else:
            print_str += '{}:\t'.format(metric)
        for modal in selected_modal:
            a = allinfo[modal][metric][0]
            # v = allinfo[modal][metric][1]
            aver += a * n_sample[modal]
            # variance += math.pow(v, 2) * n_sample[modal]
            print_str += '{:.2f}\t'.format(
                (a if metric in ['ASSD', 'HD95', 'HD', 'RAVD'] else a * 100))
        aver = aver / sum_sample
        # variance = variance / sum_sample
        # sd = math.sqrt(variance)
        print_str += '{:.2f}\n'.format(
            (aver if metric in ['ASSD', 'HD95', 'HD', 'RAVD'] else aver * 100))
    print(print_str)
    print('*' * 120)
    # print_str = 'Avg per Modality, '
    # for k, v in avg_mod.items():
    #     print_str += '{:.4f}, '.format(v / len(selected_attr))
    # print(print_str)
    #
    # print_str = 'Avg per Samples, '
    # for k, v in avg_sample.items():
    #     print_str += '{:.4f}, '.format(v / ak)
    # print(print_str)






if __name__ == '__main__':
    args = argparse.ArgumentParser('Compute the static between predictions and ground truth.')
    args.add_argument('--selected_epoch',nargs='+',default=[])
    args.add_argument('--result_dir', type=str, default='')
    args.add_argument('--test_list', type=str, default='')
    args.add_argument('--seg_type', type=str, default='')
    args.add_argument('--selected_modal', nargs='+', default=[''])
    opt = args.parse_args()

    score(opt.result_dir, opt.test_list, opt.selected_modal, opt.selected_epoch, opt.seg_type)
