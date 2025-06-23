from dataset import ImageDataset, EpisodicBatchSampler
from protonet import PrototypicalNetwork
from models.HBP import PrototypicalNetwork3
from models.FPN import FPN_Network
# from matchingnet import matchingnet
from models.relationnet import RelationNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard.writer import SummaryWriter
import os
import json
import time
import numpy as np
from tqdm import tqdm
from utils import set_seed, metrics, AverageMeter, increment_path
from thop import profile
from thop import clever_format
# from thop import profile
# from thop import clever_format
torch.backends.cudnn.enabled = False
from torch.autograd import Variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Train_protonet(object):
    def __init__(self, args):
        self.args = args
        self.args.seed = set_seed(self.args.seed)
        self.args.result_path = increment_path(self.args.result_path)
        print(f'This training log will be saved in {str(self.args.result_path)}')

        self.args.checkpoints_dir = self.args.result_path / self.args.checkpoints_dir
        self.args.tensorboard_dir = self.args.result_path / self.args.tensorboard_dir
        if not os.path.exists(self.args.checkpoints_dir):
            os.makedirs(self.args.checkpoints_dir)
        if not os.path.exists(self.args.tensorboard_dir):
            os.makedirs(self.args.tensorboard_dir)

        # save training args
        train_args = {k: str(v) for k, v in self.args.__dict__.copy().items()}
        with open(self.args.result_path / 'train_args.json', 'w', encoding='utf-8') as f:
            json.dump(train_args, f, ensure_ascii=False, indent=4)

        self.model = self._build_model()
        # self.model = self._build_relation_model()
        # self.model = self._build_fpn_model()

    def _build_model(self):
        model = PrototypicalNetwork(self.args.img_channels, self.args.hidden_channels)
        # model = PrototypicalNetwork3(self.args.img_channels, self.args.hidden_channels)
        # model = RelationNet(64, 8)
        # model = matchingnet()
        flops, params = profile(model, inputs=(torch.randn(1,1, 3, 84, 84),torch.randn(1, 1,3, 84, 84)))
        print('模型的flops:', flops / (1000 ** 3))
        print('模型的params:', params)
        flops, params = clever_format([flops, params], "%.3f")
        print('模型的flops:', flops)
        print('模型的params:', params)
        model.cuda()

        # flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
        #
        # flops, params = clever_format([flops, params], "%.3f")
        # print('模型的flops:', flops)
        # print('模型的params:', params)
        return model

    def _build_relation_model(self):
        model = RelationNet(64, 8)
        model.cuda()
        return model

    def _build_fpn_model(self):
        model = FPN_Network(3, 64)
        model.cuda()
        return model


    def _get_data(self, mode='train'):
        if mode == 'train':
            data_path = self.args.train_csv_path
            way = self.args.way
            shot = self.args.shot
            query = self.args.query
            episodes = self.args.episodes
        else:
            data_path = self.args.val_csv_path
            way = self.args.val_way
            shot = self.args.val_shot
            query = self.args.val_query
            episodes = self.args.val_episodes

        data_set = ImageDataset(
            data_path=data_path,
            shot=shot,
            query=query,
            img_channels=self.args.img_channels,
            img_size=self.args.img_size,
            mode=mode
        )
        sampler = EpisodicBatchSampler(n_classes=len(data_set), n_way=way, n_episodes=episodes)
        data_loader = DataLoader(data_set, shuffle=False, batch_sampler=sampler, num_workers=0)
        return data_loader

    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay_step)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_decay_step, gamma=0.1)
        return optimizer, scheduler

    def _select_criterion(self):
        criterion = nn.NLLLoss()
        return criterion

    def eval_model(self, epoch):
        self.model.eval()
        val_losses, accuracy, f_score = [AverageMeter() for i in range(3)]
        criterion = self._select_criterion()
        with torch.no_grad():
            val_loader = self._get_data(mode='val')
            for step, (x_support, x_query) in enumerate(tqdm(val_loader, desc='Evaluating model')):
                # data (n, x, c, w, h)
                x_support, x_query = x_support.cuda(), x_query.cuda()
                n = x_support.shape[0]
                q = x_query.shape[1]
                y_query = torch.arange(0, n, requires_grad=False).view(n, 1).expand(n, q).reshape(n * q, ).cuda()
                # infer
                output = self.model.test_mode(x_support, x_query)
                # output = self.model(x_support, x_query)
                # loss
                loss = criterion(output, y_query).item()
                val_losses.update(loss)
                # metrics of acc and f_score
                pred_ids = torch.argmax(output, dim=-1).cpu().numpy()
                y_query = y_query.cpu().numpy()
                acc, f_s = metrics(pred_ids, y_query)
                accuracy.update(acc)
                f_score.update(f_s)
        print(f'Epoch: {epoch} evaluation results: Val_loss: {val_losses.avg}, mAP: {accuracy.avg}, F_score: {f_score.avg}')
        return val_losses.avg, accuracy.avg, f_score.avg

    def eval_model_relation(self, epoch):
        mse = nn.MSELoss().cuda()
        self.model.eval()
        val_losses, accuracy, f_score = [AverageMeter() for i in range(3)]
        criterion = self._select_criterion()
        total_rewards = 0
        with torch.no_grad():
            val_loader = self._get_data(mode='val')
            for step, (x_support, x_query) in enumerate(tqdm(val_loader, desc='Evaluating model')):
                # data (n, x, c, w, h)
                x_support, x_query = x_support.cuda(), x_query.cuda()
                n = x_support.shape[0]
                q = x_query.shape[1]
                y_query = torch.arange(0, n, requires_grad=False).view(n, 1).expand(n, q).reshape(n * q, ).cuda()

                # infer
                output = self.model(x_support, x_query)
                _, predict_labels = torch.max(output.data, 1)
                y_query = y_query.view(-1, n)
                gt_labels, _ = torch.max(y_query.data, 1)
                rewards = [1 if predict_labels[j] == gt_labels[j] else 0 for j in range(n)]
                total_rewards += np.sum(rewards)

                # loss
                one_hot_labels = torch.zeros(n * q, n).scatter_(1, y_query.cpu().view(-1, 1), 1).cuda()

                loss = mse(output, one_hot_labels).item()
                val_losses.update(loss)
                # metrics of acc and f_score
                test_accuracy = total_rewards / 1.0 / n / 10
                f_score=1

                print("test accuracy:", test_accuracy)
        print(f'Epoch: {epoch} evaluation results: Val_loss: {val_losses.avg}, mAP: {test_accuracy}, F_score: {1}')
        return val_losses.avg, test_accuracy, 1

    def train_one_epoch(self, optimizer, scheduler, epoch, total_epochs):
        self.model.train()
        train_losses = AverageMeter()
        mse = nn.MSELoss().cuda()
        criterion = self._select_criterion()
        train_loader = self._get_data(mode='train')
        epoch_iterator = tqdm(train_loader,
                              desc="Training [epoch X/X | episode X/X] (loss=X.X | lr=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        a = 10
        for step, (x_support, x_query) in enumerate(epoch_iterator):
            # data (n, x, c, w, h)

            x_support, x_query = x_support.cuda(), x_query.cuda()
            k = x_support.shape[1]
            n = x_support.shape[0]
            q = x_query.shape[1]
            n_back = x_query.shape[0]
            q_back = x_support.shape[1]
            y_query = torch.arange(0, n, requires_grad=False).view(n, 1).expand(n, q).reshape(n * q, ).cuda()
            y_support = torch.arange(0, n_back, requires_grad=False).view(n_back, 1).expand(n_back, q_back).reshape(n_back * q_back, ).cuda()
            # print('y_query', y_query)
            # training one episode
            optimizer.zero_grad()
            output, output_back = self.model(x_support, x_query)
            # output = self.model(x_support, x_query)
            # output = output.view(output.shape[0])
            # loss_global = nn.CrossEntropyLoss()(output_back, y_query)
            # output = self.model(x_support, x_query)
            # print('out:', output.shape)
            # print('output_back:', output_back.shape)
            # print("y_support:", y_support.shape)

            one_hot_labels = torch.zeros(n*q, n).scatter_(1, y_query.cpu().view(-1,1), 1).cuda()
            # loss_forward = criterion(output, one_hot_labels.long())
            # loss_forward = mse(output, one_hot_labels)
            loss_forward = criterion(output, y_query)
            loss_back = criterion(output_back, y_support)
            # beta = a * (2.5*step / len(epoch_iterator))
            beta = a * (0.1 * step / len(epoch_iterator))
            loss = loss_forward + beta * loss_back
            # loss = loss_forward
            loss.backward()
            optimizer.step()

            # log
            train_losses.update(loss.item())
            epoch_iterator.set_description(
                "Training [epoch %d/%d | episode %d/%d] | (loss=%2.5f | lr=%f)" %
                (epoch, total_epochs, step + 1, len(epoch_iterator), loss.item(), scheduler.get_last_lr()[0])
            )
        scheduler.step()
        return train_losses.avg

    def train(self):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f'[{start_time}] Start Training...')
        # writer = SummaryWriter(self.args.tensorboard_dir)
        best_acc = 0
        best_f_score=0
        early_stop_list = []
        optimizer, scheduler = self._select_optimizer()

        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch(optimizer, scheduler, epoch, self.args.epochs)
            val_loss, accuracy, f_score = self.eval_model(epoch)
            # val_loss, accuracy, f_score = self.eval_model_relation(epoch)


            # logging
            print('Loss', {'TrainLoss': train_loss, 'ValLoss': val_loss}, epoch)
            print('Metrics', {'mAP': accuracy, 'F_score': f_score}, epoch)
            # writer.add_scalars('Loss', {'TrainLoss': train_loss, 'ValLoss': val_loss}, epoch)
            # writer.add_scalars('Metrics', {'mAP': accuracy, 'F_score': f_score}, epoch)

            # save checkpoint
            torch.save(self.model.state_dict(), self.args.checkpoints_dir / 'proto_last.pth')
            if best_acc < accuracy:
                torch.save(self.model.state_dict(), self.args.checkpoints_dir / 'proto_best.pth')
                best_acc = accuracy
                best_f_score = f_score

            # early stop
            early_stop_list.append(val_loss)
            if len(early_stop_list)-np.argmin(early_stop_list) > self.args.patience:
                break
        # writer.close()
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f'[{end_time}] End of all training. The highest accuracy is {best_acc}.The highest fscore is {best_f_score}')
        return
