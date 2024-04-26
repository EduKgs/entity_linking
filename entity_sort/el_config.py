import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoints/',
                            help='the output dir for model checkpoints')

        parser.add_argument('--bert_dir', default='./models/chinese-bert-wwm-ext/',
                            help='bert dir for uer')
        parser.add_argument('--data_dir', default='./data/ccks2019/',
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='./logs/',
                            help='log dir for uer')

        # other args
        parser.add_argument('--num_tags', default=2, type=int,
                            help='number of tags')
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='1',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--max_seq_len', default=256, type=int)

        parser.add_argument('--eval_batch_size', default=32, type=int)

        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')

        # train args
        parser.add_argument('--train_epochs', default=15, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        # 2e-5
        parser.add_argument('--lr', default=2e-5, type=float,
                            help='learning rate for the bert module')
        # 2e-3
        parser.add_argument('--other_lr', default=2e-4, type=float,
                            help='learning rate for the module except bert')
        # 0.5
        parser.add_argument('--max_grad_norm', default=1, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proporation', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=64, type=int)

        parser.add_argument('--ip', type=str,
                                    help='service_ip',
                                    default='0.0.0.0')
        parser.add_argument('--port', type=str,
                                    help='service_port',
                                    default=1080)
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
