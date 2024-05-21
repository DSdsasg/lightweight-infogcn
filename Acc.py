# import argparse
# import pickle
# import numpy as np
# from tqdm import tqdm
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
#
# def ensemble(ds, items):
#     if 'ntu120' in ds:
#         num_class=120
#         if 'xsub' in ds:
#             npz_data = np.load('./data/ntu120/NTU120_CSub.npz')
#             label = np.where(npz_data['y_test'] > 0)[1]
#         elif 'xset' in ds:
#             npz_data = np.load('./data/ntu120/NTU120_CSet.npz')
#             label = np.where(npz_data['y_test'] > 0)[1]
#     elif 'ntu' in ds:
#         num_class=60
#         if 'xsub' in ds:
#             npz_data = np.load('./data/ntu/NTU60_CS.npz')
#             label = np.where(npz_data['y_test'] > 0)[1]
#         elif 'xview' in ds:
#             npz_data = np.load('./data/ntu/NTU60_CV.npz')
#             label = np.where(npz_data['y_test'] > 0)[1]
#     elif 'NW_UCLA' in ds:
#         num_class=10
#         npz_data = np.load('data/NW_UCLA/val_label.pkl')
#         label = np.where(npz_data['y_test'] > 0)[1]
#     else:
#         raise NotImplementedError
#
#     ckpt_dirs, alphas = list(zip(*items))
#
#     ckpts = []
#     for ckpt_dir in ckpt_dirs:
#         with open(ckpt_dir, 'rb') as f:
#             ckpts.append(list(pickle.load(f).items()))
#
#     right_num = total_num = right_num_5 = 0
#
#     predictions = []
#
#     for i in tqdm(range(len(label))):
#         l = label[i]
#         r = np.zeros(num_class)
#         for alpha, ckpt in zip(alphas, ckpts):
#             _, r11 = ckpt[i]
#             r += r11 * alpha
#
#
#         predictions.append(np.argmax(r))
#
#         rank_5 = r.argsort()[-5:]
#         right_num_5 += int(int(l) in rank_5)
#         r = np.argmax(r)
#         right_num += int(r == int(l))
#         total_num += 1
#
#
#
#     conf_matrix = confusion_matrix(label, predictions)
#
#     class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
#
#     print("Confusion Matrix:")
#     print(conf_matrix)
#
#     for i, acc in enumerate(class_accuracy):
#         print(f"Class {i + 1} Accuracy: {acc * 100:.2f}%")
#
#
#
#     acc = right_num / total_num
#     acc5 = right_num_5 / total_num
#
#     print('Top1 Acc: {:.4f}%'.format(acc * 100))
#     print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
#
#     return conf_matrix, class_accuracy
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset',
#                         required=True,
#                         choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW_UCLA'},
#                         help='the work folder for storing results')
#
#     parser.add_argument('--position_ckpts', nargs='+',
#                         help='Directory containing "epoch1_test_score.pkl" for position eval results')
#
#     arg = parser.parse_args()
#
#     item = []
#     for ckpt in arg.position_ckpts:
#         item.append((ckpt, 1.5))
#
#     conf_matrix, class_accuracy = ensemble(arg.dataset, item)
#
#     # Visualization of Confusion Matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()


import argparse
import pickle
import numpy as np
from tqdm import tqdm


def ensemble(ds, items, class_names=None):
    if 'ntu120' in ds:
        num_class = 120
        if 'xsub' in ds:
            npz_data = np.load('./data/ntu120/NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in ds:
            npz_data = np.load('./data/ntu120/NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in ds:
        num_class = 60
        if 'xsub' in ds:
            npz_data = np.load('./data/ntu/NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in ds:
            npz_data = np.load('./data/ntu/NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'NW_UCLA' in ds:
        num_class = 10
        npz_data = np.load('data/NW_UCLA/val_label.pkl')
        label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    ckpt_dirs, alphas = list(zip(*items))

    ckpts = []
    for ckpt_dir in ckpt_dirs:
        with open(ckpt_dir, 'rb') as f:
            ckpts.append(list(pickle.load(f).items()))

    class_correct = np.zeros(num_class)
    class_total = np.zeros(num_class)

    for i in tqdm(range(len(label))):
        l = label[i]
        r = np.zeros(num_class)
        for alpha, ckpt in zip(alphas, ckpts):
            _, r11 = ckpt[i]
            r += r11 * alpha

        predicted_label = np.argmax(r)
        class_correct[l] += int(predicted_label == l)
        class_total[l] += 1

    class_acc = class_correct / class_total

    if class_names:
        for name, acc in zip(class_names, class_acc):
            print('{}: {:.4f}%'.format(name, acc * 100))
    else:
        for i, acc in enumerate(class_acc):
            print('{}: {:.4f}%'.format(i, acc * 100))

    overall_acc = np.sum(class_correct) / np.sum(class_total)
    print('Overall Top-1 Accuracy: {:.4f}%'.format(overall_acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW_UCLA'},
                        help='the work folder for storing results')

    parser.add_argument('--position_ckpts', nargs='+',
                        help='Directory containing "epoch1_test_score.pkl" for position eval results')
    parser.add_argument('--motion_ckpts', nargs='+',
                        help='Directory containing "epoch1_test_score.pkl" for motion eval results')

    parser.add_argument('--class_names_file', help='File containing names of the classes, one per line')

    arg = parser.parse_args()

    class_names = None
    if arg.class_names_file:
        with open(arg.class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    item = []
    for ckpt in arg.position_ckpts:
        item.append((ckpt, 1.5))
    for ckpt in arg.motion_ckpts:
        item.append((ckpt, 1.3))

    ensemble(arg.dataset, item, class_names)

