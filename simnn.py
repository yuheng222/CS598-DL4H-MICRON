from collections import defaultdict
from datetime import datetime
from models.models import SimNN
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
Runs training and evaluation on SimNN model presented in the MICRON paper:
https://arxiv.org/abs/2105.01876
Code in this file is referenced from:
https://github.com/ycq091044/MICRON/blob/main/src/SimNN.py
'''

# Settings
config = configparser.ConfigParser()
config.read('config/config.ini')
data_path = config['simnn']['data_path']
voc_path = config['simnn']['voc_path']
ddi_adj_path = config['simnn']['ddi_adj_path']
model_name = config['simnn']['model_name']
resume_path = config['simnn']['resume_path']
EPOCH = int(config['simnn']['epoch'])
seed = int(config['simnn']['seed'])
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
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='learning rate')
parser.add_argument('--dim', type=int, default=64, help='dimension')

args = parser.parse_args()


# Evaluate
def eval(model, data_eval, voc_size, epoch, val=0, threshold1=0.8, threshold2=0.2):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0
    add_list, delete_list = [], []

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        add_temp_list, delete_temp_list = [], []
        if len(input) < 2: continue
        for adm_idx, adm in enumerate(input):
            if adm_idx == 0:
                y_old = np.zeros(voc_size[2])
                y_old[adm[2]] = 1
                continue

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            result_out = model(input[:adm_idx + 1])
            # Prediction prod
            y_pred_tmp = F.sigmoid(result_out[:, 0]).detach().cpu().numpy().tolist()
            y_pred_prob.append(y_pred_tmp)

            previous_set = np.where(y_old == 1)[0]

            # Prediction med set
            assignment = torch.max(result_out, axis=1)[1].cpu().numpy()
            y_old[assignment == 1] = 1
            y_old[assignment == 2] = 0
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

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(
        add_list), np.mean(delete_list), med_cnt / visit_cnt


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
    print(voc_size)
    model = SimNN(voc_size, ddi_adj, emb_dim=args.dim, device=device)

    if args.test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        eval(model, data_test, voc_size, 0)
        print('test time: {}'.format(time.time() - tic))
        return

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = RMSprop(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    criterion = torch.nn.CrossEntropyLoss()

    start_run_str = '===Starting training and evaluations for SimNN model==='
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

                target_list = loss_bce_target - loss_bce_target_last
                target_list[target_list == -1] = 2
                result = model(seq_input)

                loss += criterion(result, torch.LongTensor(target_list[0]).to(device))

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
