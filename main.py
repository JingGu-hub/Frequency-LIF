import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from utils.utils import set_seed


def main():

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status (1 for training, 0 for testing)')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model name, options: [iTransformer, Transformer, TimeXer, MoFo, DeepBooTS, ChannelTokenFormer]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../data/MTSF_dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size, applicable on arbitrary number of variates in inverted Transformers')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='Frequency_LIF', help='activation function')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus, e.g., "0,1,2"')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF', help='experiment name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='classifier strategy, options:[projection, average, cls_token]')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the target data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='target data csv file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient training, exp_name should be partial_train')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0,
                        help='the start index of variates for partial training, select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # MoFo
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--periodic', type=int, default=24, help='periodic length')
    parser.add_argument('--head', type=int, default=8, help='number of attention heads in MoFo')
    parser.add_argument('--bias', type=int, default=1, help='use bias in MoFo layers (1) or not (0)')
    parser.add_argument('--cias', type=int, default=1, help='use channel-wise attention in MoFo (1) or not (0)')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # ChannelTokenFormer
    parser.add_argument('--patch_lens', type=int, nargs='+', default=[16, 16, 16, 16, 16, 16, 16], help='patch lengths for each variate')
    parser.add_argument('--sampling_periods', type=float, nargs='+', default=[1, 1, 1, 1, 1, 1, 1], help='sampling periods for each variate')
    parser.add_argument('--num_global_tokens', type=int, default=1, help='number of global tokens')
    parser.add_argument('--fix_seed', type=int, default=2026, help='random seed')
    parser.add_argument('--missing_ratio', type=float, default=0, help='missing ratio for input data')
    parser.add_argument('--ckpt_dir', type=str, default='', help='checkpoint directory for loading pretrained model')
    parser.add_argument('--keep_prob', type=float, default=1, help='keep probability for patch masking during training')

    # DeepBooTS
    parser.add_argument('--attn', type=int, default=1, help='use attention in DeepBooTS (1) or not (0)')
    parser.add_argument('--d_block', type=int, help='dimension of block in DeepBooTS')
    parser.add_argument('--gate', type=int, default=1, help='use gate in DeepBooTS (1) or not (0)')

    # SNNs
    parser.add_argument('--T', type=int, default=4, help='time steps for SNN')

    # Frequency-LIF
    parser.add_argument('--frequency_factor', default=0.5, type=float, help='frequency factor for Frequency-LIF')
    parser.add_argument('--alpha', default=0.1, type=float, help='alpha parameter for Frequency-LIF')
    parser.add_argument('--beta', default=0.7, type=float, help='beta parameter for Frequency-LIF')
    parser.add_argument('--min_beta', default=0.5, type=float, help='minimum beta value for Frequency-LIF')
    parser.add_argument('--beta_decay_factor', default=0.5, type=float, help='beta decay factor for Frequency-LIF')

    args = parser.parse_args()

    if args.exp_name == 'partial_train':  # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else:  # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast

    set_seed(args)
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
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
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
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
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()


