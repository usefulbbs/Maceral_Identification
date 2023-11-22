import  matplotlib.pylab as plt
import  random
import  ssl
from    _utils  import *
from    dataset import MyDataset
from    Loss    import DiceLoss
from    score   import model_score
from    torch   import utils, cuda, device, nn

ssl._create_default_https_context = ssl._create_unverified_context

def model_train(model_name, bb='resnet18', cc=256): #定义训练的main函数

    for i in range(5):#五折

        lr = 2e-4
        num_epoch = 2

        if cuda.is_available():
            mydevice = device('cuda:3')
        crit = nn.CrossEntropyLoss()
        crit_d = DiceLoss()
        mynet = model_select(model_name, bb, cc, mydevice) #选择对应的模型

        cross_fold = {  '1' : [102, 105,                115, 116, 117],
                        '2' : [101,      107, 109, 111, 114],
                        '3' : [103,           110, 112,           119],
                        '4' : [               108, 113,           120, 122],
                        '5' : [100, 104, 106,                     118, 121]}
        cross_num = str(i+1)
        test_img_name_list = cross_fold[cross_num]
        
        train_dataset = MyDataset('cut_image1', 'cut_image2', 'cut_label', test_img_name_list, has_edge=[True if model_name == 'UNetFormerEDGE' else False][0], train_val_test='train')
        train_loader = utils.data.DataLoader(train_dataset, 32, True, drop_last=True) #加载训练数据集

        val_dataset = MyDataset('cut_image1', 'cut_image2', 'cut_label', test_img_name_list, has_edge=False, train_val_test='val')
        val_loader = utils.data.DataLoader(val_dataset, 32, False, drop_last=False) #加载验证数据集 
    
        test_dataset = MyDataset('cut_image1', 'cut_image2', 'cut_label', test_img_name_list, has_edge=False, train_val_test='test')
        test_loader = utils.data.DataLoader(test_dataset, 32, False, drop_last=False) #加载验证数据集 

        best_model, log = train_model(mynet, lr, crit, crit_d, mydevice, train_loader, val_loader, test_loader, num_epoch = num_epoch, 
                                      edge_flag=[True if model_name == 'UNetFormerEDGE' else False][0], ah_flag=[True if model_name == 'UNetFormer' else False][0]) #训练模型

        torch.save(best_model, 'log/' + model_name + cross_num + bb+str(cc) + '_epoch_' + str(num_epoch) + '.pkl')
        log.to_json(           'log/' + model_name + cross_num + bb+str(cc) + '_epoch_' + str(num_epoch) + '.json', orient='index') #保存验证集上最好的模型与log

        train_loss_all = list(log['train_loss_all'])
        val_loss_all = list(log['val_loss_all'])
        miou = list(log['mIOU'])

        x = range(1, num_epoch+1)
        plt.figure('train_loss_all')
        plt.plot(x, train_loss_all, 'r')
        plt.figure('val_loss_all')
        plt.plot(x, val_loss_all, 'b')
        plt.figure('miou')
        plt.plot(x, miou, 'g')
        plt.show() #可视化

    model_score(model_name, num_epoch, test_loader, bb, cc)

model_train(model_name = 'PSP')
model_train(model_name = 'deeplabV3+')
model_train(model_name = 'Unet')
model_train(model_name = 'ATTUNet')
model_train(model_name = 'UNetFormer')
model_train(model_name = 'UNetFormerEDGE', bb='resnet18', cc=64) 
model_train(model_name = 'UNetFormerEDGE', bb='resnet18', cc=128)
model_train(model_name = 'UNetFormerEDGE', bb='resnet18', cc=256) #best model
model_train(model_name = 'UNetFormerEDGE', bb='resnet18', cc=512)
model_train(model_name = 'UNetFormerEDGE', bb='resnet34', cc=64) 
model_train(model_name = 'UNetFormerEDGE', bb='resnet34', cc=128)
model_train(model_name = 'UNetFormerEDGE', bb='resnet34', cc=256)
model_train(model_name = 'UNetFormerEDGE', bb='resnet34', cc=512)
model_train(model_name = 'UNetFormerEDGE', bb='resnet50', cc=64) 
model_train(model_name = 'UNetFormerEDGE', bb='resnet50', cc=128)
model_train(model_name = 'UNetFormerEDGE', bb='resnet50', cc=256) 
model_train(model_name = 'UNetFormerEDGE', bb='resnet50', cc=512)
model_train(model_name = 'UNetFormerEDGE', bb='resnet101', cc=64) 
model_train(model_name = 'UNetFormerEDGE', bb='resnet101', cc=128)
model_train(model_name = 'UNetFormerEDGE', bb='resnet101', cc=256) 
model_train(model_name = 'UNetFormerEDGE', bb='resnet101', cc=512)
