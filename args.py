import argparse


def str2bool(v):
    return v.lower() in ['true']

def get_VAE_config():
    parser = argparse.ArgumentParser()

    # Quantum circuit configuration
    parser.add_argument('--quantum', type=bool, default=True, help='choose to use quantum vae')
    parser.add_argument('--patches', type=int, default=1, help='number of quantum circuit patches')
    parser.add_argument('--layer', type=int, default=1, help='number of repeated variational quantum layer')
    parser.add_argument('--qubits', type=int, default=20, help='number of qubits and dimension of domain labels')
    
    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128, 256, 512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]],
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for training D')
    parser.add_argument('--num_epochs_decay', type=int, default=2500, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=100, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Use either of these two datasets.
    parser.add_argument('--mol_data_dir', type=str, default='data/qm9_5k.sparsedataset')
    # parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')

    # Directories.
    parser.add_argument('--saving_dir', type=str, default='../exp_results/VAE_test/')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1)
    parser.add_argument('--lr_update_step', type=int, default=500)

    # For training
    config = parser.parse_args()
    config.mode = 'train'
    config.lambda_wgan = 1.0
    config.g_lr = config.d_lr = 1e-4
    config.model_save_step = 1
    config.batch_size = 128
    config.num_epochs = 150

    # For testing
    # config.mode = 'test'
    # config.saving_dir = 'exp_results/VAE/2020-06-03_13-38-00'
    # config.resume_epoch = 150

    return config
