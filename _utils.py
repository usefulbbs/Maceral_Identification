import  copy
import  gc
import  matplotlib.pylab    as plt
import  numpy               as np
import  pandas              as pd
import  segmentation_models_pytorch as smp
import  torch
from    collections        import OrderedDict
from    score              import SegmentationMetric
from    torchvision        import transforms
from    tqdm               import tqdm
from    UNetFormer         import UNetFormer_, UNetFormer_edge

def train_model(model, l_r, crit, crit_d, device, traindataloader, valdataloader, testdataloader, num_epoch=40, edge_flag=None, ah_flag=None): #训练以及验证模型的函数
    # since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_miou = 0.0
    train_loss_all = []
    val_loss_all = []
    MIOU_all = []

    optimizer_CosineLR = torch.optim.AdamW(model.parameters(), lr=l_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False, foreach=None, capturable=False)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_CosineLR, T_max=8, eta_min=0)
    test_crit = torch.nn.CrossEntropyLoss()
    bce_crit = torch.nn.BCELoss() #设置优化器和损失函数
    # print(edge_flag,edge_flag==False, ah_flag)
    for epoch in range(num_epoch):

        gc.collect()
        torch.cuda.empty_cache()
        metric = SegmentationMetric(4) # 4表示有4个分类，有几个分类就填几 此处实例化计算mIoU等评价指标的类

        print("epoch : {} / {}".format(epoch + 1, num_epoch))
        print("-" * 20)
        train_loss = 0
        train_num = 0

        model.train()
        loop = tqdm(enumerate(traindataloader), total=len(traindataloader))
        if edge_flag == False: #此处是因为模型在训练时有额外的辅助损失 而其他模型没有
            for step, (imgs, targets) in loop:

                imgs = imgs.float().to(device)
                targets = targets.long().to(device)
                if ah_flag:
                    out, ah = model(imgs)
                    loss = crit(out, targets) + crit_d(out, targets) + 0.4*crit(ah, targets) + 0.4*crit_d(ah, targets)
                else:
                    out = model(imgs)
                    loss = crit(out, targets) + crit_d(out, targets)

                optimizer_CosineLR.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
                optimizer_CosineLR.step()

                train_loss += loss.item() * len(targets)
                train_num += len(targets)

                loop.set_description(f'Epoch [{epoch+1}/{num_epoch}]')
                loop.set_postfix(loss=train_loss/(step+1))
        else:
            print("edge is True!!")
            for step, (imgs, targets, edge) in loop:

                imgs = imgs.float().to(device)
                targets = targets.long().to(device)
                edge = edge.to(device, dtype=torch.float32)
                out, oo, ah, ed = model(imgs)
                loss_main = crit(out, targets) + crit_d(out, targets)
                loss_aux1 = crit(oo, targets)
                loss_aux2 = crit(ah, targets)
                loss_aux3 = bce_crit(ed, edge)
                '''------------------loss---------------------'''
                loss = loss_main + 0.4*loss_aux1 + 0.4*loss_aux2 + 0.4*loss_aux3 #计算损失
                # loss = loss_main + 0.4*loss_aux2
                # loss = loss_main + 0.4*loss_aux1
                # loss = loss_main
                # loss = loss_main + 0.4*loss_aux1 + 0.4*loss_aux2

                optimizer_CosineLR.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
                optimizer_CosineLR.step()
                train_loss += loss.item() * len(targets)
                train_num += len(targets)
                loop.set_description(f'Epoch [{epoch+1}/{num_epoch}]')
                loop.set_postfix(loss=train_loss/(step+1))

        train_loss_all.append(train_loss / train_num)
        # print("{}, train loss : {:.4f}".format(epoch + 1, train_loss_all[-1]))

        model.eval() #验证模型
        with torch.no_grad():
            val_loss = 0
            val_num = 0
            IMG_PRE = OrderedDict()
            IMG_LAB = OrderedDict()
            for step, (imgs, targets) in enumerate(valdataloader):
                imgs = imgs.float().to(device)
                targets = targets.long().to(device)
                out = model(imgs)
                pre_lab = torch.argmax(out.cpu(), 1).squeeze(0).numpy()
                label0 = targets.cpu().numpy()
                IMG_PRE[step] = pre_lab
                IMG_LAB[step] = label0
                loss = test_crit(out, targets)
                val_loss += loss.item() * len(targets)
                val_num += len(targets)
            label_all = np.concatenate([value for value in IMG_LAB.values()], axis=0)
            pre_lab_all = np.concatenate([value for value in IMG_PRE.values()], axis=0)
            metric.addBatch(label_all, pre_lab_all)
            IoU, mIoU = metric.meanIntersectionOverUnion()
            print(IoU, mIoU)
            MIOU_all.append(mIoU)

            val_loss_all.append(val_loss / val_num)

            if mIoU > best_miou:
                best_miou = mIoU
                best_model = copy.deepcopy(model.state_dict())

        # print('mIoU: ' + str(round(np.mean(MIOU), 4)))
        # print("{}, val loss : {:.4f}".format(epoch + 1, val_loss_all[-1]))
        CosineLR.step()
        # time_use = time.time() - since
        # print("train and val time : {}".format(time_use))

    train_process = pd.DataFrame(
        index = range(num_epoch),
        data = {
            'mIOU':MIOU_all,
            "train_loss_all":train_loss_all,
            "val_loss_all":val_loss_all})

    model.load_state_dict(best_model) #保存最优模型以及训练Log
    model = copy.deepcopy(model)

    return model, train_process

def label2img(prelabel, colormap): #可视化标签函数
    h, w = prelabel.shape
    prelabel = prelabel.reshape(h * w, -1)
    image = np.zeros((h * w, 3), dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)
        image[index, :] = colormap[ii]
    return image.reshape(h, w, 3)

def model_select(model_name, bb, cc, mydevice):
    if model_name == 'deeplabV3+':
        mynet = smp.DeepLabV3Plus(
                                encoder_name='resnet18',
                                encoder_depth=5, 
                                encoder_weights='imagenet', 
                                encoder_output_stride=16, 
                                decoder_channels=256, 
                                decoder_atrous_rates=(12, 24, 36),
                                in_channels=6, classes=4, 
                                activation=None,
                                upsampling=4, 
                                aux_params=None)
    if model_name == 'ATTUNet':
        mynet = smp.Unet(     encoder_name='resnet18', 
                                encoder_depth=5, 
                                encoder_weights='imagenet', 
                                decoder_use_batchnorm=True, 
                                decoder_channels=(256, 128, 64, 32, 16), 
                                decoder_attention_type='scse', 
                                in_channels=6, 
                                classes=4, 
                                activation=None, 
                                aux_params=None)
    if model_name == 'Unet':
        mynet = smp.Unet(     encoder_name='resnet18', 
                                encoder_depth=5, 
                                encoder_weights='imagenet', 
                                decoder_use_batchnorm=True, 
                                decoder_channels=(256, 128, 64, 32, 16), 
                                decoder_attention_type=None, 
                                in_channels=6, 
                                classes=4, 
                                activation=None, 
                                aux_params=None)
    if model_name == 'PSP':
        mynet = smp.PSPNet(   encoder_name='resnet18', 
                                encoder_weights='imagenet', 
                                encoder_depth=3, 
                                psp_out_channels=512, 
                                psp_use_batchnorm=True, 
                                psp_dropout=0.2, 
                                in_channels=6, classes=4, 
                                activation=None, upsampling=8, aux_params=None)
    if model_name == 'UNetFormerEDGE':
        mynet = UNetFormer_edge(
                                    decode_channels=cc,
                                    dropout=0.12,
                                    backbone_name=bb,
                                    pretrained=True,
                                    window_size=8,
                                    num_classes=4,
                                    in_chans=6)
    if model_name == 'UNetFormer':
        mynet = UNetFormer_(
                                    decode_channels=256,
                                    dropout=0,
                                    backbone_name='resnet18',
                                    pretrained=True,
                                    window_size=8,
                                    num_classes=4)
    mynet.to(device=mydevice)
    return mynet