# evaluate voiceMOS challenge answers - true scores vs predictions
# adapted from https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/predict.py  (voiceMOS baseline "SSL-MOS")
#   but with added per-system evaluation using averaged validation scores - the original used values from mydata_system.csv, which also includes the training data
# MK, 2022-01-13 
#  v. 1.0.1 - slightly modified output, otherwise unchanged
#  v. 1.1.0 (Feb 2024) - skip files missing from true answers
#  
# original copyright notice: 
# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import argparse
import numpy as np
import scipy.stats

def systemID(uttID):
    return uttID.split('-')[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_scores', type=str, required=True, help='Path to the true answers (val_mos_list.txt).')
    parser.add_argument('--predict_scores', type=str, required=True, help='Path to the predicted scores (answers.txt)')
    parser.add_argument('--system_csv', type=str, required=False, default=None, help='Path to the average scores per system (mydata_system.csv)')
    args = parser.parse_args()

    validlist = args.true_scores
    answerlist = args.predict_scores
    system_csv_path = args.system_csv

    true_MOS = { }
    validf = open(validlist, 'r')
    for line in validf:
        parts = line.strip().split(',')
        uttID = parts[0]
        MOS = float(parts[1])
        true_MOS[uttID] = MOS

    num_missing = 0
    num_predictions_total = 0

    predictions = { }
    predictf = open(answerlist, 'r')
    for line in predictf:
        parts = line.strip().split(',')
        uttID = parts[0]
        num_predictions_total += 1
        if uttID in true_MOS:
            MOS = float(parts[1])
            predictions[uttID] = MOS
        else:
            num_missing += 1
            #print("True score not available for %s. Skipping."%uttID)
            continue

    if num_missing > 0:
        print("Skipped %d files with missing true scores (out of %d total predictions)."%(num_missing,num_predictions_total))

    ## compute correls.
    sorted_uttIDs = sorted(predictions.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
        ts.append(t)
        ps.append(p)

    truths = np.array(ts)
    preds = np.array(ps)

    ### UTTERANCE
    MSE=np.mean((truths-preds)**2)
    print('[UTTERANCE] Test error (MSE) = %f' % MSE)
    LCC=np.corrcoef(truths, preds)
    print('[UTTERANCE] Linear correlation coefficient = %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(truths.T, preds.T)
    print('[UTTERANCE] Spearman rank correlation coefficient = %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(truths, preds)
    print('[UTTERANCE] Kendall Tau rank correlation coefficient = %f' % KTAU[0])

    ### SYSTEM
    pred_sys_MOSes = { }
    for uttID in sorted_uttIDs:
        sysID = systemID(uttID)
        noop = pred_sys_MOSes.setdefault(sysID, [ ])
        pred_sys_MOSes[sysID].append(predictions[uttID])

    pred_sys_MOS_avg = { }
    for k, v in pred_sys_MOSes.items():
        avg_MOS = sum(v) / (len(v) * 1.0)
        pred_sys_MOS_avg[k] = avg_MOS


    true_sys_MOSes = { }
    for uttID in sorted_uttIDs:
        sysID = systemID(uttID)
        noop = true_sys_MOSes.setdefault(sysID, [ ])
        true_sys_MOSes[sysID].append(true_MOS[uttID])    

    true_sys_MOS_avg_calculated = { }
    for k, v in true_sys_MOSes.items():
        avg_MOS = sum(v) / (len(v) * 1.0)
        true_sys_MOS_avg_calculated[k] = avg_MOS

    ## avg. system scores from mydata_system.csv
    if system_csv_path is not None:
        true_sys_MOS_avg = { }
        csv_file = open(system_csv_path, 'r')
        csv_file.readline()  ## skip header
        for line in csv_file:
            parts = line.strip().split(',')
            sysID = parts[0]
            MOS = float(parts[1])
            true_sys_MOS_avg[sysID] = MOS

    ## make lists sorted by system
    pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
    sys_p = [ ]
    sys_t = [ ]
    sys_t_a = [ ]
    for sysID in pred_sysIDs:
        sys_p.append(pred_sys_MOS_avg[sysID])
        sys_t_a.append(true_sys_MOS_avg_calculated[sysID])
        if system_csv_path is not None:
            sys_t.append(true_sys_MOS_avg[sysID])

    
    sys_predicted = np.array(sys_p)

    sys_true_a = np.array(sys_t_a)


    if system_csv_path is not None:
        sys_true = np.array(sys_t)

        print('-----')
        print('System scores using mydata_system.csv as truth, as per the original script (averaged over training *and* validation data):')

        MSE=np.mean((sys_true-sys_predicted)**2)
        print('[SYSTEM] Test error (MSE) = %f' % MSE)
        LCC=np.corrcoef(sys_true, sys_predicted)
        print('[SYSTEM] Linear correlation coefficient = %f' % LCC[0][1])
        SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
        print('[SYSTEM] Spearman rank correlation coefficient = %f' % SRCC[0])
        KTAU=scipy.stats.kendalltau(sys_true, sys_predicted)
        print('[SYSTEM] Kendall Tau rank correlation coefficient = %f' % KTAU[0])

        print('-----')
        print('System scores using average values from the evaluated utterances (this is the version used in the official evaluation):')

    MSE=np.mean((sys_true_a-sys_predicted)**2)
    print('[SYSTEM] Test error (MSE) = %f' % MSE)
    LCC=np.corrcoef(sys_true_a, sys_predicted)
    print('[SYSTEM] Linear correlation coefficient = %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(sys_true_a.T, sys_predicted.T)
    print('[SYSTEM] Spearman rank correlation coefficient = %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(sys_true_a, sys_predicted)
    print('[SYSTEM] Kendall Tau rank correlation coefficient = %f' % KTAU[0])

if __name__ == '__main__':
    main()