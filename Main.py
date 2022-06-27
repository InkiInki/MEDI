# coding: utf-8
"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2022 0506
Last modify: 2022 0507
"""

import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import torch.utils.data as data_utils

from MIL import MIL
from utils import get_k_cv_idx, dis_euclidean
from NN import NN, l1_regularization, l2_regularization


class MEDI(MIL):
    """
    The MEDI algorithm
    """

    def __init__(self,
                 file_name: str,
                 epoch: int = 10,
                 lr: float = 0.001,
                 max_dim: int = 1000,
                 norm_type: str = "l1",  # l1, l2
                 distill_type: str = "agg",  # agg, max, max_min
                 bag_space: None = None):
        """
        :param file_name:               The file name
        :param epoch:                   The number of epoch for MEDI
        :param
        """
        super(MEDI, self).__init__(file_name, bag_space=bag_space)
        self.epoch = epoch
        self.lr = lr
        self.max_dim = max_dim
        self.norm_type = norm_type
        self.distill_type = distill_type

        self.loss = nn.CrossEntropyLoss()

    def __mapping_data(self, net):
        """"""
        ret_vec = []
        for i in range(self.N):

            temp_vec = self.__mapping_bag(i, net)
            temp_vec = np.sign(temp_vec) * np.sqrt(np.abs(temp_vec))
            temp_vec = temp_vec / dis_euclidean(temp_vec, np.zeros_like(temp_vec))
            ret_vec.append(temp_vec.tolist())

        return np.array(ret_vec)

    def __mapping_bag(self, idx, net):
        bag = self.bag_space[idx, 0][:, :-1]
        bag = torch.from_numpy(bag)

        bag = net.get_mapping(bag.float())
        prob = net.classify(bag).detach().numpy()
        bag = bag.detach().numpy()
        max_class = [prob[np.argsort(prob[:, c])[-1], c] for c in range(self.C)]
        max_class = np.argmax(max_class)
        distill_order = np.argsort(prob[:, max_class])[::-1]

        if self.distill_type == "agg":
            return np.average(bag, 0)
        elif self.distill_type == "max":
            return bag[distill_order[0]]
        else:
            return np.average([bag[distill_order[-1]], bag[distill_order[0]]], 0)

    def get_mapping(self):
        """"""
        tr_idxes, te_idxes = get_k_cv_idx(self.N, k=5)
        for i, (tr_idx, te_idx) in enumerate(zip(tr_idxes, te_idxes)):
            # print("%d-th loop of 10-cv" % (i + 1))
            tr_ins, tr_ins_lab, _ = self.get_sub_ins_space(tr_idx)
            tr_ins, tr_ins_lab = torch.from_numpy(tr_ins), torch.from_numpy(tr_ins_lab)
            tr_data = data_utils.TensorDataset(tr_ins, tr_ins_lab)
            tr_loader = data_utils.DataLoader(tr_data, batch_size=10, shuffle=True)
            del tr_ins, tr_ins_lab, tr_data
            net = NN(self.d, self.C)
            optim = opt.Adam(net.parameters(), self.lr)
            batch_count = 0
            for epoch in range(self.epoch):
                tr_loss, tr_acc, num_test_ins = 0, 0, 0
                for batch, (data, label) in enumerate(tr_loader):
                    if label.min() == -1:
                        label[label == -1] = 0
                    pre_lab = net(data.float())
                    loss = self.loss(pre_lab, label.long())
                    optim.zero_grad()
                    loss.backward()
                    if self.norm_type == "l1":
                        l1_regularization(net)
                    elif self.norm_type == "l2":
                        l2_regularization(net)
                    else:
                        l1_regularization(net)
                        l2_regularization(net)
                    optim.step()
                    tr_loss += loss.cpu().item()
                    tr_acc += (pre_lab.argmax(dim=1) == label).sum().cpu().item()
                    batch_count += 1
                    num_test_ins += len(label)
                # print("Epoch %d, loss %.4f, tr acc %.4f" % (epoch + 1, tr_loss / batch_count, tr_acc / num_test_ins))

            mapping_mat = self.__mapping_data(net)

            yield mapping_mat[tr_idx], self.bag_lab[tr_idx], mapping_mat[te_idx], self.bag_lab[te_idx], None


def test_10cv():
    """
    """
    po_label = 0
    file_name = "D:/OneDrive/Files/Code/Data/MIL/Text/alt_atheism.mat"
    epoch = 5

    """======================================================="""
    epoch = 1 if epoch == 0 else epoch
    loops = 5
    te_f1_k, te_acc_k, te_roc_k = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_s, te_acc_s, te_roc_s = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_j, te_acc_j, te_roc_j = np.zeros(loops), np.zeros(loops), np.zeros(loops)

    print("=================================================")
    print("File name: %s; Epoch: %d" % (file_name.split(".")[-2].split("/")[-1], epoch))
    # from MnistLoad import MnistLoader
    # bag_space = MnistLoader(seed=1, po_label=po_label, data_type="mnist", data_path=file_name).bag_space
    from Classifier import Classifier
    DISTILL = ["agg", "max", "max_min"]
    for distill in DISTILL:
        dsk = MEDI(file_name, epoch=epoch, norm_type="l2", distill_type=distill)
        print(distill)
        for i in range(loops):
            classifier = Classifier(["knn", "svm", "j48"], ["f1_score", "acc"])
            data_iter = dsk.get_mapping()
            te_per = classifier.test(data_iter)
            te_f1_k[i], te_acc_k[i] = te_per["knn"][0], te_per["knn"][1]
            te_f1_s[i], te_acc_s[i] = te_per["svm"][0], te_per["svm"][1]
            te_f1_j[i], te_acc_j[i] = te_per["j48"][0], te_per["j48"][1]
            print("%.4lf, %.4lf, %.4lf; %.4lf, %.4lf, %.4lf; %.4lf, %.4lf, %.4lf; \n"
                  % (te_f1_k[i], te_acc_k[i], te_roc_k[i],
                     te_f1_s[i], te_acc_s[i], te_roc_s[i],
                     te_f1_j[i], te_acc_j[i], te_roc_j[i]
                     ), end=" ")

        print("knn-f1 std   knn-acc std   knn-roc std   svm-f1 std    svm-acc std   svm-roc std   "
              "j48-f1 std    j48-acc std   j48-roc std")
        print("%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
              "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf " % (np.sum(te_f1_k) / loops, np.std(te_f1_k),
                                                        np.sum(te_acc_k) / loops, np.std(te_acc_k),
                                                        np.sum(te_f1_s) / loops, np.std(te_f1_s),
                                                        np.sum(te_acc_s) / loops, np.std(te_acc_s),
                                                        np.sum(te_f1_j) / loops, np.std(te_f1_j),
                                                        np.sum(te_acc_j) / loops, np.std(te_acc_j)))


if __name__ == '__main__':
    import time
    s_t = time.time()
    test_10cv()
    print("%.4f" % (time.time() - s_t))
