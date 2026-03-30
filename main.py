import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

from utils.utils import set_seed, create_file, set_dataset


def main(gpu_id=0, pred_len_list=[96], result_dir='./outputs', model='iTransformer', activation='gelu', T=4,
         use_high_freq_balance=True, use_dynamic_beta=True):

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, default=1.0, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../data/MTSF_dataset/ETT-small/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')  # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7,
                        help='output size')  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default=0, help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False,
                        help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/',
                        help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False,
                        help='whether to use efficient_training (exp_name should be partial train)')  # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0,
                        help='the start index of variates for partial training, '
                             'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # MoFo
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='')
    parser.add_argument('--periodic', type=int, default=24, help='data loader num workers')
    parser.add_argument('--head', type=int, default=8, help='num of heads')
    parser.add_argument('--bias', type=int, default=1, help='num of heads')
    parser.add_argument('--cias', type=int, default=1, help='num of heads')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # ChannelTokenFormer
    parser.add_argument('--patch_lens', type=int, nargs='+', default=[16, 16, 16, 16, 16, 16, 16], help='patch lengths')
    parser.add_argument('--sampling_periods', type=float, nargs='+', default=[1, 1, 1, 1, 1, 1, 1],
                        help='sampling periods')
    parser.add_argument('--num_global_tokens', type=int, default=1, help='number of global tokens')
    parser.add_argument('--fix_seed', type=int, default=2026, help='gpu')
    parser.add_argument('--missing_ratio', type=float, default=0, help='missing ratio')
    parser.add_argument('--ckpt_dir', type=str, default='', help='checkpoint directory')
    parser.add_argument('--keep_prob', type=float, default=1, help='train time patch masking keep probability')

    # DeepBooTS
    parser.add_argument('--attn', type=int, default=1, help='use attention (1) or not (0)')
    parser.add_argument('--d_block', type=int, help='dimension of model')
    parser.add_argument('--gate', type=int, default=1, help='use gate (1) or not (0)')

    # SNNs
    parser.add_argument('--threshold', default=1.0, type=float, help='')
    parser.add_argument('--T', type=int, default=4, help='data loader num workers')

    # Frequency_LIF
    parser.add_argument('--season_factor', default=0.8, type=float, help='')
    parser.add_argument('--alpha', default=0.1, type=float, help='')
    parser.add_argument('--beta', default=0.7, type=float, help='')
    parser.add_argument('--beta_decay_factor', default=0.5, type=float, help='')
    parser.add_argument('--use_high_freq_balance', type=bool, default=True, help='data file')
    parser.add_argument('--use_dynamic_beta', type=bool, default=True, help='data file')

    args = parser.parse_args()

    args.gpu = gpu_id
    args.result_dir = result_dir
    args.model = model
    args.activation = activation
    args.T = T
    args.use_high_freq_balance = use_high_freq_balance
    args.use_dynamic_beta = use_dynamic_beta

    datasets = ['ETTh1', 'ETTh2']
    # datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange', 'weather', 'solar-energy', 'traffic']

    if args.exp_name == 'partial_train':  # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else:  # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast

    for dataset in datasets:
        args.dataset = dataset
        set_dataset(args)

        for pred_len in pred_len_list:
            args.pred_len = pred_len
            if args.d_block is None: args.d_block = args.pred_len

            args.out_path = args.result_dir + '/%s/%s/%s/' % (args.model, args.activation, args.dataset)
            args.output_file = create_file(args.out_path, '%s_%s_predlen%d.txt' % (args.model, args.dataset, pred_len), 'epoch,train_loss,vali_loss,test_loss')
            args.results_file = create_file(args.out_path, 'results.txt', 'statement,mse_loss,mae_loss', exist_create_flag=False)

            set_seed(2026)
            if args.is_training:
                for ii in range(args.itr):
                    # setting record of experiments
                    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}_{}_{}'.format(
                        args.model_id,
                        args.model,
                        args.dataset,
                        args.features,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        args.d_model,
                        args.n_heads,
                        args.e_layers,
                        args.d_layers,
                        args.d_ff,
                        args.factor,
                        args.embed,
                        args.distil,
                        args.des,
                        args.activation,
                        args.season_factor,
                        args.beta,
                        args.beta_decay_factor,
                        args.use_high_freq_balance,
                        args.use_dynamic_beta,
                        args.class_strategy, ii)

                    exp = Exp(args)  # set experiments
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.train(setting)

                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)

                    if args.do_predict:
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.predict(setting, True)

                    torch.cuda.empty_cache()
            else:
                ii = 0
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}_{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.dataset,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    args.activation,
                    args.season_factor,
                    args.beta,
                    args.beta_decay_factor,
                    args.use_high_freq_balance,
                    args.use_dynamic_beta,
                    args.class_strategy, ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()

if __name__ == '__main__':
    # model_list = ['iTransformer', 'MoFo', 'TimeXer', 'ChannelTokenFormer', 'DeepBooTS']
    model_list = ['iTransformer']

    for model in model_list:
        # main(gpu_id=0, pred_len_list=[96, 192, 336, 720], activation='original', T=1, model=model)
        # main(gpu_id=0, pred_len_list=[96, 192, 336, 720], activation='lif', T=4, model=model)
        # main(gpu_id=0, pred_len_list=[96, 192, 336, 720], activation='tclif', T=4, model=model)
        # main(gpu_id=0, pred_len_list=[96, 192, 336, 720], activation='tslif', T=4, model=model)
        main(gpu_id=0, pred_len_list=[96, 192, 336, 720], activation='Frequency_LIF', T=4, model=model)



