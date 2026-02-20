import numpy as np
import h5py
import torch
from sklearn.model_selection import train_test_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchaudio.functional import edit_distance
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
from realtime_sim.ctc_decoder import greedy_decode_batch
from realtime_sim.realtime_datamodule import (CTCHeldOutDataModule,
                                              CTCHeldOutTargetValAlignDataModule)
from realtime_sim.realtime_nn_model import RealtimeRNNModel
import realtime_sim.augmentations as augs

N_SIL = 0
BLANK_TOKEN = 0
SIL_TOKEN = 10
PHON_DICT = {
    0: 'blank',
    1: 'a',
    2: 'ae',
    3: 'i',
    4: 'u',
    5: 'b',
    6: 'p',
    7: 'v',
    8: 'g',
    9: 'k',
    10: 'sil',
}

AUGS_DICT = {
    'time_warping': augs.time_warping,
    'time_masking': augs.time_masking,
    'time_shifting': augs.time_shifting,
    'noise_jitter': augs.noise_jitter,
    'scaling': augs.scaling,
}


@hydra.main(version_base=None, config_path="config", config_name="train_ctc_rnn_config")
def main(cfg: DictConfig) -> None:
    # Check for missing keys in config
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')
    
    print('Configuration:', flush=True)
    for key, value in cfg.items():
        print(f'  {key}: {value}', flush=True)
    
    # Setup logging
    logger = make_logger(cfg.paths.nn_log_dir,
                         cfg.target_pt,
                         cfg)
    
    # Convert path parameters to be Path objects
    cfg.paths.data_dir = Path(cfg.paths.data_dir).expanduser()
    cfg.paths.results_dir = Path(cfg.paths.results_dir).expanduser()
    cfg.paths.nn_data_dir = Path(cfg.paths.nn_data_dir).expanduser()
    cfg.paths.nn_log_dir = Path(cfg.paths.nn_log_dir).expanduser()

    # Load data from target patient
    X_train_tgt, y_train_tgt, X_test, y_test = load_data(
            cfg.paths.data_dir / cfg.paths.data_fname,
            cfg.target_pt,
            cfg.data_proc.tw_select,
            cfg.data_proc.tw_orig,
            zscore=cfg.data_proc.zscore,
            only_train=False,
            n_sil=N_SIL,
        )
    if cfg.data_proc.target_subsample < 1:
        # train test split to subsample training data with stratification
        try:
            X_train_tgt, _, y_train_tgt, _ = train_test_split(X_train_tgt,
                                        y_train_tgt,
                                        train_size=cfg.data_proc.target_subsample,
                                        stratify=y_train_tgt[:,0],
                                        shuffle=True)
        except ValueError:  # when fraction is too small to stratify
            X_train_tgt, _, y_train_tgt, _ = train_test_split(X_train_tgt,
                                        y_train_tgt,
                                        train_size=cfg.data_proc.target_subsample,
                                        shuffle=True)

    # (Optional) Pooling data from other patients
    X_train_cross = []
    y_train_cross = []
    if cfg.pool_train:        
        # Load in data from other patients
        other_pts = [pt for pt in cfg.train_pts if pt != cfg.target_pt]
        for pt in other_pts:
            # use ALL data from non-target patients (except S33 which only has 1 block of data)
            only_train = True if pt in ['S33'] else False
            load_all = False if pt in ['S33'] else True
            X_pt, y_pt, _, _ = load_data(cfg.paths.data_dir / cfg.paths.data_fname,
                                         pt,
                                         cfg.data_proc.tw_select,
                                         cfg.data_proc.tw_orig,
                                         zscore=cfg.data_proc.zscore,
                                         only_train=only_train,
                                         load_all=load_all,
                                         n_sil=N_SIL,
                                         )

            # Append to training data
            X_train_cross.append(X_pt)
            y_train_cross.append(y_pt)
    else:
        # Placeholders for cross-patient data in patient-specific case
        X_train_cross = None
        y_train_cross = None

    # Load optimal hyperparameters from tuning runs
    best_tune_cfg = load_hparams(cfg)
    print('Loaded hyperparameters:', best_tune_cfg, flush=True)

    # Train model over multiple iterations with optimal hyperparamters
    pers_all = []
    logits_all = []
    for i in range(cfg.training.n_iter):

        # reshuffle chance labels for each iteration
        if cfg.compute_chance:
            # Shuffle training labels by trial for chance performance
            rand_idx = np.random.permutation(y_train_tgt.shape[0])
            y_train_tgt = y_train_tgt[rand_idx]

        # Define data module
        augs_list = [AUGS_DICT[aug] for aug in cfg.training.augmentations]
        dm = select_datamodule(cfg, X_train_tgt, y_train_tgt, X_train_cross,
                            y_train_cross, X_test, y_test,
                            int(best_tune_cfg['batch_size']), cfg.training.val_size,
                            augs_list, cfg.paths.nn_data_dir)
        dm.setup()

        # Define model
        data_shapes = dm.get_data_shape()
        model = RealtimeRNNModel(
            input_size=data_shapes[-1]*cfg.model.win_size,
            hidden_size=int(best_tune_cfg['hidden_size']),
            n_layers=int(best_tune_cfg['n_layers']),
            n_classes=len(PHON_DICT),
            dropout=float(best_tune_cfg['dropout']),
            win_size=cfg.model.win_size,
            stride=cfg.model.stride,
            learning_rate=float(best_tune_cfg['learning_rate']),
            decay_steps=int(cfg.training['n_epochs']),
            weight_decay=float(best_tune_cfg['l2_reg']),
        )

        # Model training
        callbacks = [
            ModelCheckpoint(monitor='val_PER', mode='min')
        ]

        trainer = L.Trainer(
            accelerator='auto',
            max_epochs=cfg.training.n_epochs,
            gradient_clip_val=float(best_tune_cfg['gclip_val']),
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=cfg.paths.nn_log_dir,
        )
        trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        
        # Load best model for evaluation
        best_ckpt_path = callbacks[0].best_model_path
        print(f'Loading best model from {best_ckpt_path} for evaluation...')
        test_model = RealtimeRNNModel.load_from_checkpoint(best_ckpt_path)

        # Evaluate model on test data
        test_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_model.to(device)

        test_dataloader = dm.test_dataloader()
        X_batch, y_batch, _, _ = next(iter(test_dataloader))
        X_batch = X_batch.to(device)
        with torch.no_grad():
            logits = test_model(X_batch)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

        # compute edit distance
        norm_edit_dist = calc_norm_edit_distance(logits, y_batch)
        per = norm_edit_dist * 100  # convert to percentage
        print(f'Iteration {i+1} - Phoneme Error Rate for {cfg.target_pt}: {round(per,3)}', flush=True)

        pers_all.append(per)
        logits_all.append(logits.cpu().numpy())

        # save intermediate results to tmp directory to avoid data loss
        save_results(cfg.paths.nn_log_dir, cfg, cfg.target_pt, pers_all, logits_all, PHON_DICT, model.hparams)
        print(f'Intermediate results saved to {cfg.paths.nn_log_dir}', flush=True)

    # Save final results over iterations
    save_results(cfg.paths.results_dir, cfg, cfg.target_pt, pers_all, logits_all, PHON_DICT, model.hparams)
    print(f'Final results saved to {cfg.paths.results_dir}', flush=True)


def make_logger(log_dir, pt, cfg):
    log_str = f'{pt}'
    suffix = '_ptSpecific'
    if cfg.pool_train:
        if cfg.align_train:
            suffix = '_aligned'
        else:
            suffix= '_unaligned'
    if cfg.compute_chance:
        suffix = '_chance'
    log_str += suffix
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=f'{log_str}_ctcRnn',
    )
    return logger


def load_data(data_filename, pt, tw_select, tw_orig, zscore=False, only_train=False, load_all=False, n_sil=2):
    feat_key_train = 'norm_rt_HG_pow_z' if zscore else 'norm_rt_HG_pow'
    feat_key_test = 'norm_rt_HG_test_pow_z' if zscore else 'norm_rt_HG_test_pow'

    # load data
    with h5py.File(data_filename, 'r') as f:
        feats_train = f[f'{pt}/{feat_key_train}'][:].transpose(0, 2, 1)  # reshape to (n_trials, n_time, n_channels)
        labels_train = f[f'{pt}/labels_train'][:]
        if only_train:
            feats_test = None
            labels_test = None
        else:
            feats_test = f[f'{pt}/{feat_key_test}'][:].transpose(0, 2, 1)
            labels_test = f[f'{pt}/labels_test'][:]

    # select desired time window
    t_range_orig = np.linspace(tw_orig[0], tw_orig[1], feats_train.shape[1])
    t_select_mask = (t_range_orig >= tw_select[0]) & (t_range_orig <= tw_select[1])
    feats_train = feats_train[:, t_select_mask, :]
    if not only_train:
        feats_test = feats_test[:, t_select_mask, :]

    # process labels for CTC decoding
    for _ in range(n_sil):
        labels_train = np.insert(labels_train, 0, SIL_TOKEN, axis=1)
        labels_train = np.insert(labels_train, labels_train.shape[1], SIL_TOKEN, axis=1)
        if not only_train:
            labels_test = np.insert(labels_test, 0, SIL_TOKEN, axis=1)
            labels_test = np.insert(labels_test, labels_test.shape[1], SIL_TOKEN, axis=1)

    # (optional) combine train and test into larger dataset
    if load_all:
        feats_train = np.concatenate([feats_train, feats_test], axis=0)
        labels_train = np.concatenate([labels_train, labels_test], axis=0)
        feats_test = None
        labels_test = None
    
    return feats_train, labels_train, feats_test, labels_test


def select_datamodule(cfg, X_train_tgt, y_train_tgt, X_train_cross,
                      y_train_cross, X_test, y_test, batch_size,
                      val_size, augmentations, data_dir):
    if cfg.pool_train:
        dm = CTCHeldOutTargetValAlignDataModule(
            X_train_tgt,
            y_train_tgt,
            X_train_cross,
            y_train_cross,
            X_test,
            y_test,
            batch_size=batch_size,
            val_size=val_size,
            augmentations=augmentations,
            data_path=data_dir / f'{cfg.target_pt}_data',
            pool=cfg.pool_train,
            n_comp=cfg.data_proc.n_components,
            align=cfg.align_train,
        )
    else:
        dm = CTCHeldOutDataModule(
            X_train_tgt,
            y_train_tgt,
            X_test,
            y_test,
            batch_size=batch_size,
            val_size=val_size,
            augmentations=augmentations,
            data_path=data_dir / f'{cfg.target_pt}_data',
        )
    return dm


def load_hparams(cfg, hparam_dir='~/data/results/decoding/ctc_results_tuneRndm30CV_90varNoDel_computeAlign_augs/'):
    # get parameters from input configuration as default
    best_tune_cfg = {
        'batch_size': cfg.training.batch_size,
        'learning_rate': cfg.training.learning_rate,
        'gclip_val': cfg.training.gclip_val,
        'hidden_size': cfg.model.hidden_size,
        'n_layers': cfg.model.n_layers,
        'dropout': cfg.model.dropout,
        'l2_reg': cfg.model.l2_reg,
    }

    if cfg.pool_train:
        if cfg.align_train:
            context = 'aligned'
        else:
            context = 'unaligned'
    elif cfg.compute_chance:
        context = 'chance'
    else:
        context = 'ptSpecific'
    fname = Path(hparam_dir).expanduser() / cfg.target_pt / f'{cfg.target_pt}_ctcRNN_{context}_hp.h5'

    try:
        with h5py.File(fname, 'r') as f:
            # replace default keys if found in hparam file
            for k, v in f.items():
                if k in best_tune_cfg.keys():
                    best_tune_cfg[k] = v[()]
    except FileNotFoundError:
        print('Saved hyparameters not found! Using defaults from yaml config file.')

    return best_tune_cfg


def calc_norm_edit_distance(input_seqs, target_seqs):
    tot_dist = 0
    n_tokens = 0
    decoded_outputs = greedy_decode_batch(input_seqs)
    for i in range(len(input_seqs)):
        edit_dist = edit_distance(decoded_outputs[i], target_seqs[i])
        tot_dist += edit_dist
        n_tokens += len(target_seqs[i])
    norm_edit_dist = tot_dist / n_tokens
    return norm_edit_dist


def save_results(save_dir, cfg, pt, pers_all, logits_all, phon_dict, model_hparams):
    save_dir = Path(save_dir)
    save_fname = f'{pt}/{pt}_ctcRNN_decodeTW([{cfg.data_proc.tw_select[0]},{cfg.data_proc.tw_select[1]}])'
    suffix = '_ptSpecific'

    if cfg.pool_train:
        if cfg.align_train:
            suffix = '_aligned'
        else:
            suffix= '_unaligned'
    if cfg.compute_chance:
        suffix = '_chance'
    save_fname += suffix

    tgt_r = cfg.data_proc.target_subsample
    tgt_subsamp_str = f'_tgtSubsamp{(tgt_r*100):.0f}' if tgt_r < 1 else ''
    save_fname += tgt_subsamp_str

    save_path = save_dir / (save_fname + '.h5')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('phoneme_error_rate', data=np.array(pers_all))
        f.create_dataset('logits', data=np.array(logits_all))
        phon_keys = np.array(list(phon_dict.keys()), dtype=int)
        phon_vals = np.array(list(phon_dict.values()), dtype='S')  # store as bytes
        f.create_dataset('phon_keys', data=phon_keys)
        f.create_dataset('phon_vals', data=phon_vals)
        # save model hyperparameters as attributes
        model_grp = f.create_group('model_hparams')
        for key, val in model_hparams.items():
            model_grp.attrs[key] = val


if __name__ == '__main__':
    main()
