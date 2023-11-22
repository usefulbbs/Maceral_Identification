"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""

import  numpy       as     np
import  pandas      as     pd
import  torch
import  ttach       as     tta
from    collections import OrderedDict

class SegmentationMetric(object): #这个类是用来计算语义分割的评价指标的
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
 
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU) # 求各类别IoU的平均
        return IoU, mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
 

def model_score(model_name, num_epoch, test_loader, bb='resnet18', cc=256): #用来计算验证集上的最优模型在测试集上的评价指标

    if torch.cuda.is_available():
        mydevice = torch.device('cuda:3')

    PA = []
    CPA = []
    MPA = []
    IOU = []
    MIOU = []
    LabelList = []
    result = {}
    mask = {}
    label_all = OrderedDict()
    pre_lab_all = OrderedDict()

    for index in range(1,6):

        cross_num = str(index)

        model = torch.load('log/' + model_name + cross_num + bb+str(cc) + '_epoch_' + str(num_epoch) + '.pkl')#, map_location='cpu')
        model.to(device=mydevice)
        model.eval()

        metric = SegmentationMetric(4) # 4表示有4个分类，有几个分类就填几

        with torch.no_grad():

            IMG_PRE = OrderedDict()
            IMG_LAB = OrderedDict()

            for step, (imgs, targets) in enumerate(test_loader):
                imgs = imgs.float().to(device=mydevice)
                targets = targets.long().to(device=mydevice)
                if model_name == 'UNetFormerEDGE':
                    # transforms = tta.aliases.flip_transform()
                    # tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='max')
                    # pre_lab = torch.argmax(tta_model(imgs).cpu(), dim=1).squeeze(0).numpy()
                    out = model(imgs)
                    pre_lab = torch.argmax(out.cpu(), 1).squeeze(0).numpy()
                else:
                    out = model(imgs)
                    pre_lab = torch.argmax(out.cpu(), 1).squeeze(0).numpy()
                label0 = targets.cpu().numpy()
                IMG_PRE[step] = pre_lab
                IMG_LAB[step] = label0
            label_all[index] = np.concatenate([value for value in IMG_LAB.values()], axis=0)
            pre_lab_all[index] = np.concatenate([value for value in IMG_PRE.values()], axis=0)

    out = np.concatenate([value for value in pre_lab_all.values()], axis=0)
    lab = np.concatenate([value for value in label_all.values()], axis=0)

    metric.addBatch(out, lab)
    pa = metric.pixelAccuracy()
    PA.append(pa)
    cpa = metric.classPixelAccuracy()
    CPA.append(cpa)
    mpa = metric.meanPixelAccuracy()
    MPA.append(mpa)
    IoU, mIoU = metric.meanIntersectionOverUnion()
    IOU.append(IoU)
    MIOU.append(mIoU)
    metric.reset()

    print('pa is : %f' % pa)
    print('cpa is :') # 列表
    print(cpa)
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU)

    log = pd.DataFrame(
          data = {
            "PA"        : PA,
            'CPA'       : CPA,
            "MPA"       : MPA,
            "IOU"       : IOU,
            "MIOU"      : MIOU
                 }    )
    log.to_json('log/' + model_name + bb+str(cc) + '.json') #保存对应的结果


if __name__ == '__main__':
    for model_name in ['UNetFormerEDGE']:
    # for model_name in ['PSP', 'ATTUNet', 'deeplabV3+', 'Unet', 'UNetFormerEDGE', 'UNetFormer']:
        print(model_name)
        model_score(model_name)