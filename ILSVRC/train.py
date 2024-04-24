import os
import argparse
import torch
import torch.nn as nn
from Model import *
from DataLoader import ImageDataset
from torch.autograd import Variable
from utils.loss import Loss
from utils.accuracy import *
from utils.optimizer import *
from utils.lr import *
import os
import random
import time

seed = 2

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        ##  path
        self.parser.add_argument("--root", type=str, default="IMNET_DATA")
        self.parser.add_argument("--num_classes", type=int, default=1000)

        self.parser.add_argument("--save_path", type=str, default="logs")
        self.parser.add_argument("--load_path", type=str, default="VGG.pth.tar")
        ##  image
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--resize_size", type=int, default=256)
        ##  dataloader
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument("--nest", action="store_true")
        ##  train
        self.parser.add_argument("--batch_size", type=int, default=32 * 4)
        self.parser.add_argument("--epochs", type=int, default=9)
        self.parser.add_argument("--pretrain", type=str, default="True")
        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--lr", type=float, default=0.001)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)
        self.parser.add_argument("--power", type=float, default=0.9)
        self.parser.add_argument("--momentum", type=float, default=0.9)
        ##  model
        self.parser.add_argument(
            "--arch", type=str, default="vgg"
        )  ## choosen  [ vgg, resnet, inception, mobilenet]
        ##  show
        self.parser.add_argument("--show_step", type=int, default=500)
        ##  GPU'
        self.parser.add_argument("--gpu", type=str, default="0,1,2,3")
        ##  parameter
        self.parser.add_argument("--lambda_a", type=float, default=1.0)
        self.parser.add_argument("--lambda_b", type=float, default=1.0)
        self.parser.add_argument("--lambda_c", type=float, default=1.0)
        self.parser.add_argument("--lambda_d", type=float, default=1.0)
        self.parser.add_argument("--lambda_e", type=float, default=1.0)

    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch
        return opt


args = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
lr = args.lr

if __name__ == "__main__":
    """print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)"""
    print(torch.cuda.get_device_name(0))
    if args.phase == "train":
        ##  data.
        MyData = ImageDataset(args)
        MyDataLoader = torch.utils.data.DataLoader(
            dataset=MyData,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        ##  model
        model = eval(args.arch).model(args, pretrained=True)
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.cuda(device=0)
        model = model
        model.train()
        ##  optimizer
        optimizer = get_optimizer(model, args)
        loss_func = nn.CrossEntropyLoss().cuda()

        print(model)
        print("\n\033[32mStart Training ... \033[0m \n")
        for epoch in range(0, args.epochs):
            ##  accuracy
            cls_acc_1 = AverageMeter()
            cls_acc_2 = AverageMeter()
            loss_epoch_1 = AverageMeter()
            loss_epoch_2 = AverageMeter()
            loss_epoch_3 = AverageMeter()
            loss_epoch_4 = AverageMeter()
            loss_epoch_5 = AverageMeter()
            poly_lr_scheduler(optimizer, epoch, decay_epoch=3)
            # poly_lr_scheduler(optimizer, epoch, decay_epoch=1)
            torch.cuda.synchronize()
            start = time.time()

            for step, (path, imgs, label) in enumerate(MyDataLoader):

                imgs, label = Variable(imgs).cuda(device=0), label.cuda(device=0)
                ##  backward
                optimizer.zero_grad()
                # label = label.long()
                output_dict = model(imgs, label, N=1)
                loss_1 = output_dict["losses"][0].mean(0)
                loss_2 = output_dict["losses"][1].mean(0)
                loss_3 = output_dict["losses"][2].mean(0)
                loss_4 = output_dict["losses"][3].mean(0)
                loss_5 = output_dict["losses"][4].mean(0)
                loss = (
                    loss_1
                    + args.lambda_a * loss_2
                    + args.lambda_b * loss_3
                    + args.lambda_c * loss_4
                    + args.lambda_d * loss_5
                )

                loss.backward()
                optimizer.step()

                ##  count_accuracy
                cur_batch = label.size(0)
                cur_cls_acc_1 = 100.0 * compute_cls_acc(output_dict["score_1"], label)
                cls_acc_1.updata(cur_cls_acc_1, cur_batch)
                cur_cls_acc_2 = 100.0 * compute_cls_acc(output_dict["score_2"], label)
                cls_acc_2.updata(cur_cls_acc_2, cur_batch)
                loss_epoch_1.updata(loss_1.data, 1)
                loss_epoch_2.updata(loss_2.data, 1)
                loss_epoch_3.updata(loss_3.data, 1)
                loss_epoch_4.updata(loss_4.data, 1)
                loss_epoch_5.updata(loss_5.data, 1)

                if (step + 1) % args.show_step == 0:
                    print(
                        "Epoch:[{}/{}] step:[{}/{}] LOSS: {:.3f} loss_cls1:{:.3f} loss_cls2:{:.3f} loss_acv2:{:.3f} loss_ac:{:.3f} loss_bf:{:.3f} cls_acc1:{:.2f}% cls_acc2:{:.2f}%".format(
                            epoch + 1,
                            args.epochs,
                            step + 1,
                            len(MyDataLoader),
                            loss,
                            loss_epoch_1.avg,
                            loss_epoch_2.avg,
                            loss_epoch_3.avg,
                            loss_epoch_4.avg,
                            loss_epoch_5.avg,
                            cls_acc_1.avg,
                            cls_acc_2.avg,
                        )
                    )
            if (epoch + 1) >= 5:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_path, "model_epoch" + str(epoch) + ".pth.tar"
                    ),
                )
