from train import Train_protonet


class TrainArgs(object):
    def __init__(self):
        # record saving path
        self.result_path = './runs/exp'
        self.checkpoints_dir = 'checkpoint'
        self.tensorboard_dir = 'tensorboard_log'

        # data loader args
        # self.train_csv_path = './train_data.csv'
        # self.val_csv_path = './test_data.csv'
        # self.train_csv_path = './localmaize_train_data.csv'
        # self.val_csv_path = './localmaize_test_data.csv'
        # self.train_csv_path = './localcorn_train_data.csv'
        # self.val_csv_path = './localcorn_test_data.csv'
        # self.train_csv_path = './new_train_data.csv'
        # self.val_csv_path = './new_test_data.csv'
        # self.train_csv_path = './PV3aug_train_data.csv'
        # self.val_csv_path = './PV3aug_test_data.csv'
        # self.train_csv_path = './PV1aug_train_data.csv'
        # self.val_csv_path = './PV1aug_test_data.csv'
        # self.train_csv_path = './PV2aug_train_data.csv'
        # self.val_csv_path = './PV2aug_test_data.csv'


        self.train_csv_path = './data_csv/SPLIT2aug_train_data.csv'
        self.val_csv_path = './data_csv/SPLIT2aug_test_data.csv'
        # self.train_csv_path = './PD2_train_data.csv'
        # self.val_csv_path = './PD2_test_data.csv'
        # self.train_csv_path = './PV1_train_data.csv'
        # self.val_csv_path = './PV1_test_data.csv'
        # self.train_csv_path = './IP102train_data.csv'
        # self.val_csv_path = './IP102test_data.csv'
        # self.train_csv_path = './pdtrain_data.csv'
        # self.val_csv_path = './pdtest_data.csv'
        # self.val_csv_path = './CD_local_corn_test_data.csv'
        # self.val_csv_path = './CD_local_maize_test_data.csv'
        self.img_channels = 3
        self.img_size = 84

        self.way = 5
        self.shot = 20
        self.query = 5
        self.episodes = 10

        self.val_way = 5
        self.val_shot = 20
        self.val_query = 5
        self.val_episodes = 10

        # train args
        self.epochs = 600
        self.patience = 50
        self.learning_rate = 0.0001
        self.lr_decay_step = 1
        self.weight_decay_step = 0
        self.seed = 42

        # model args
        self.hidden_channels = 64
        # self.hidden_channels = 32


if __name__ == '__main__':
    args = TrainArgs()
    exp_p = Train_protonet(args)
    exp_p.train()
