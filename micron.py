from collections import defaultdict
from datetime import datetime
from models.models import MICRON
from sklearn.metrics import roc_curve
from torch.optim import RMSprop
from utils.util import llprint, multi_label_metric, ddi_rate_score, get_n_params
import argparse
import configparser
import dill
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

'''
Runs training and evaluation on Micron model presented in the MICRON paper:
https://arxiv.org/abs/2105.01876
Code in this file is referenced from:
https://github.com/ycq091044/MICRON/blob/main/src/MICRON.py
'''

# Settings
config = configparser.ConfigParser()
config.read('config/config.ini')
data_path = config['micron']['data_path']
voc_path = config['micron']['voc_path']
ddi_adj_path = config['micron']['ddi_adj_path']
model_name = config['micron']['model_name']
resume_path = config['micron']['resume_path']
EPOCH = int(config['micron']['epoch'])
seed = int(config['micron']['seed'])
start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
results_file_name = f'results/{model_name}/{start_time}.txt'

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

if not os.path.exists(os.path.join("results", model_name)):
    os.makedirs(os.path.join("results", model_name))

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='learning rate')
parser.add_argument('--dim', type=int, default=64, help='dimension')

args = parser.parse_args()


# Evaluate
def eval(model, data_eval, voc_size, epoch, val=0, threshold1=0.8, threshold2=0.2):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0
    label_list, prob_list = [], []
    add_list, delete_list = [], []

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        add_temp_list, delete_temp_list = [], []
        if len(input) < 2: continue
        for adm_idx, adm in enumerate(input):
            if adm_idx == 0:
                representation_base, _, _, _, _ = model(input[:adm_idx + 1])

                y_old = np.zeros(voc_size[2])
                y_old[adm[2]] = 1
                continue

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            label_list.append(y_gt_tmp)

            _, _, residual, _, _ = model(input[:adm_idx + 1])
            # Prediction prod
            representation_base += residual
            y_pred_tmp = F.sigmoid(representation_base).detach().cpu().numpy()[0]
            y_pred_prob.append(y_pred_tmp)
            prob_list.append(y_pred_tmp)

            previous_set = np.where(y_old == 1)[0]

            # Prediction med set
            y_old[y_pred_tmp >= threshold1] = 1
            y_old[y_pred_tmp < threshold2] = 0
            y_pred.append(y_old)

            # Prediction label
            y_pred_label_tmp = np.where(y_old == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

            # Add or delete
            add_gt = set(np.where(y_gt_tmp == 1)[0]) - set(previous_set)
            delete_gt = set(previous_set) - set(np.where(y_gt_tmp == 1)[0])

            add_pre = set(np.where(y_old == 1)[0]) - set(previous_set)
            delete_pre = set(previous_set) - set(np.where(y_old == 1)[0])

            add_distance = len(set(add_pre) - set(add_gt)) + len(set(add_gt) - set(add_pre))
            delete_distance = len(set(delete_pre) - set(delete_gt)) + len(set(delete_gt) - set(delete_pre))

            add_temp_list.append(add_distance)
            delete_temp_list.append(delete_distance)

        if len(add_temp_list) > 1:
            add_list.append(np.mean(add_temp_list))
            delete_list.append(np.mean(delete_temp_list))
        elif len(add_temp_list) == 1:
            add_list.append(add_temp_list[0])
            delete_list.append(delete_temp_list[0])

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred),
                                                                                 np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # DDI rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)

    eval_results = '\nDDI Rate: {:.4f}, Jaccard: {:.4f},  F1: {:.4f}, Add: {:.4f}, Delete: {:.4f}, Med: {:.4f}\n'.format(
        np.float(ddi_rate), np.mean(ja), np.mean(avg_f1), np.mean(add_list), np.mean(delete_list), med_cnt / visit_cnt
    )

    llprint(eval_results)

    with open(results_file_name, "a") as f:
        f.write(eval_results)
        f.close()

    if val == 0:
        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(
            add_list), np.mean(delete_list), med_cnt / visit_cnt
    else:
        return np.array(label_list), np.array(prob_list)


def main():
    device = torch.device('cuda')

    # Load data
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    torch.manual_seed(seed)
    np.random.seed(seed)
    np.random.shuffle(data)

    split_point = int(len(data) * 3 / 5)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = MICRON(voc_size, ddi_adj, emb_dim=args.dim, device=device)

    if args.test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        label_list, prob_list = eval(model, data_eval, voc_size, 0, 1)

        threshold1, threshold2 = [], []
        for i in range(label_list.shape[1]):
            _, _, boundary = roc_curve(label_list[:, i], prob_list[:, i], pos_label=1)
            # Boundary1 should be in [0.5, 0.9], boundary2 should be in [0.1, 0.5]
            threshold1.append(min(0.9, max(0.5, boundary[max(0, round(len(boundary) * 0.05) - 1)])))
            threshold2.append(max(0.1, min(0.5, boundary[min(round(len(boundary) * 0.95), len(boundary) - 1)])))
        print(np.mean(threshold1), np.mean(threshold2))
        threshold1 = np.ones(voc_size[2]) * np.mean(threshold1)
        threshold2 = np.ones(voc_size[2]) * np.mean(threshold2)
        eval(model, data_test, voc_size, 0, 0, threshold1, threshold2)
        print('test time: {}'.format(time.time() - tic))

        return

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = RMSprop(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    weight_list = [[0.25, 0.25, 0.25, 0.25]]

    start_run_str = '===Starting training and evaluations for MICRON model==='
    print(start_run_str)
    with open(results_file_name, "a") as f:
        f.write(start_run_str)
        f.close()

    for epoch in range(EPOCH):
        t = 0
        tic = time.time()
        epoch_start_str = '\nepoch {} --------------------------'.format(epoch + 1)
        print(epoch_start_str)
        with open(results_file_name, "a") as f:
            f.write(epoch_start_str)
            f.close()

        sample_counter = 0
        mean_loss = np.array([0, 0, 0, 0])

        model.train()
        for step, input in enumerate(data_train):
            loss = 0
            if len(input) < 2: continue
            for adm_idx, adm in enumerate(input):
                if adm_idx == 0: continue
                seq_input = input[:adm_idx + 1]

                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_bce_target_last = np.zeros((1, voc_size[2]))
                loss_bce_target_last[:, input[adm_idx - 1][2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                loss_multi_target_last = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(input[adm_idx - 1][2]):
                    loss_multi_target_last[0][idx] = item

                result, result_last, _, loss_ddi, loss_rec = model(seq_input)

                loss_bce = 0.75 * F.binary_cross_entropy_with_logits(result,
                                                                     torch.FloatTensor(loss_bce_target).to(device)) + \
                           (1 - 0.75) * F.binary_cross_entropy_with_logits(result_last,
                                                                           torch.FloatTensor(loss_bce_target_last).to(
                                                                               device))
                loss_multi = 5e-2 * (0.75 * F.multilabel_margin_loss(F.sigmoid(result),
                                                                     torch.LongTensor(loss_multi_target).to(device)) +
                                     (1 - 0.75) * F.multilabel_margin_loss(F.sigmoid(result_last),
                                                                           torch.LongTensor(loss_multi_target_last).to(
                                                                               device)))

                y_pred_tmp = F.sigmoid(result).detach().cpu().numpy()[0]
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_label = np.where(y_pred_tmp == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)

                if sample_counter == 0:
                    lambda1, lambda2, lambda3, lambda4 = weight_list[-1]
                else:
                    current_loss = np.array([loss_bce.detach().cpu().numpy(), loss_multi.detach().cpu().numpy(),
                                             loss_ddi.detach().cpu().numpy(), loss_rec.detach().cpu().numpy()])
                    current_ratio = (current_loss - np.array(mean_loss)) / np.array(mean_loss)
                    instant_weight = np.exp(current_ratio) / sum(np.exp(current_ratio))
                    lambda1, lambda2, lambda3, lambda4 = instant_weight * 0.75 + np.array(weight_list[-1]) * 0.25
                    # Update weight_list
                    weight_list.append([lambda1, lambda2, lambda3, lambda4])
                # Update mean_loss
                mean_loss = (mean_loss * (sample_counter - 1) + np.array([loss_bce.detach().cpu().numpy(),
                                                                          loss_multi.detach().cpu().numpy(),
                                                                          loss_ddi.detach().cpu().numpy(),
                                                                          loss_rec.detach().cpu().numpy()])) / sample_counter

                if current_ddi_rate > 0.08:
                    loss += lambda1 * loss_bce + lambda2 * loss_multi + \
                            lambda3 * loss_ddi + lambda4 * loss_rec
                else:
                    loss += lambda1 * loss_bce + lambda2 * loss_multi + \
                            lambda4 * loss_rec

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, add, delete, avg_med = eval(model, data_eval, voc_size, epoch)
        computation_results_str = 'training time: {:.4f}, test time: {:.4f}'.format(time.time() - tic,
                                                                                    time.time() - tic2)
        print(computation_results_str)

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['add'].append(add)
        history['delete'].append(delete)
        history['med'].append(avg_med)

        if epoch >= 5:
            mean_eval_results = 'Mean DDI: {:.4f}, Mean Jaccard: {:.4f}, Mean F1: {:.4f}, Mean Add: {:.4f},' \
                                ' Mean Delete: {:.4f}, Mean Med: {:.4f}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['add'][-5:]),
                np.mean(history['delete'][-5:]),
                np.mean(history['med'][-5:])
            )
            print(mean_eval_results)
            with open(results_file_name, "a") as f:
                f.write(f'{mean_eval_results}\n')
                f.close()

        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name,
                                                         'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja,
                                                                                                    ddi_rate)), 'wb'))

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        best_epoch_str = 'best_epoch: {}'.format(best_epoch)
        print(best_epoch_str)
        with open(results_file_name, "a") as f:
            f.write(f'{computation_results_str}\n')
            f.write(f'{best_epoch_str}\n')
            f.close()

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))


if __name__ == '__main__':
    main()
