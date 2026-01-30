import numpy as np
import h5py
import torch
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
from realtime_sim.realtime_datamodule import CTCHeldOutDataModule, CTCHeldOutTargetValDataModule
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
    
    # Setup logging
    logger = make_logger(cfg.paths.nn_log_dir,
                         cfg.target_pt,
                         cfg)
    
    # Convert path parameters to be Path objects
    cfg.paths.data_dir = Path(cfg.paths.data_dir).expanduser()
    cfg.paths.results_dir = Path(cfg.paths.results_dir).expanduser()
    cfg.paths.nn_data_dir = Path(cfg.paths.nn_data_dir).expanduser()
    cfg.paths.nn_log_dir = Path(cfg.paths.nn_log_dir).expanduser()
    cfg.paths.pca_path = Path(cfg.paths.pca_path).expanduser()
    cfg.paths.cca_path = Path(cfg.paths.cca_path).expanduser()

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

    # (Optional) Pooling data from other patients
    pooled_data = []
    pooled_labels = []
    if cfg.pool_train:
        # project target data to latent space
        n_trs_train_tgt, n_time_tgt, n_chan_tgt = X_train_tgt.shape
        n_trs_test_tgt, _, _ = X_test.shape
        pca_xform_tgt = load_pca_xform(cfg.paths.pca_path, cfg.target_pt)

        X_train_r = X_train_tgt.reshape(-1, n_chan_tgt)
        X_train_r -= np.mean(X_train_r, axis=0, keepdims=True)
        X_train_r = np.dot(X_train_r, pca_xform_tgt).reshape(n_trs_train_tgt, n_time_tgt, -1)
        X_train_tgt = X_train_r

        X_test_r = X_test.reshape(-1, n_chan_tgt)
        X_test_r -= np.mean(X_test_r, axis=0, keepdims=True)
        X_test_r = np.dot(X_test_r, pca_xform_tgt).reshape(n_trs_test_tgt, n_time_tgt, -1)
        X_test = X_test_r

        # align target data to desired pt space if align pt is different from target pt
        if cfg.align_train and cfg.target_pt != cfg.align_pt:
            n_latents = X_train_tgt.shape[-1]

            cca_xform_tgt = load_cca_xform(cfg.paths.cca_path, cfg.align_pt, cfg.target_pt)
            X_train_algn = X_train_tgt.reshape(-1, n_latents)
            X_train_algn = np.dot(X_train_algn, cca_xform_tgt).reshape(n_trs_train_tgt, n_time_tgt, -1)
            X_train_tgt = X_train_algn

            X_test_algn = X_test.reshape(-1, n_latents)
            X_test_algn = np.dot(X_test_algn, cca_xform_tgt).reshape(n_trs_test_tgt, n_time_tgt, -1)
            X_test = X_test_algn
        
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
            n_trs_pt, n_time_pt, n_chan_pt = X_pt.shape

            # load offline PCA transform
            pca_xform = load_pca_xform(cfg.paths.pca_path, pt)

            # demean in realtime space instead of using saved mean from offline PCA
            X_pt_r = X_pt.reshape(-1, n_chan_pt)
            X_pt_r -= np.mean(X_pt_r, axis=0, keepdims=True)

            # transform to latent space
            X_pt_r = np.dot(X_pt_r, pca_xform).reshape(n_trs_pt, n_time_pt, -1)

            # (Optional) Align to desired patient space (not necessarily target patient)
            if cfg.align_train and pt != cfg.align_pt:
                n_latents = X_pt_r.shape[-1]

                cca_xform = load_cca_xform(cfg.paths.cca_path, cfg.align_pt, pt)
                X_pt_algn = X_pt_r.reshape(-1, n_latents)
                X_pt_algn = np.dot(X_pt_algn, cca_xform).reshape(n_trs_pt, n_time_pt, -1)
                X_pt_r = X_pt_algn

            # Append to training data
            pooled_data.append(X_pt_r)
            pooled_labels.append(y_pt)

        # Truncate unaligned data to consistent dimensionality across patients
        if not cfg.align_train:
            all_pt_data = [X_train_tgt] + pooled_data
            min_dim = min([data.shape[-1] for data in all_pt_data])

            X_train_tgt = X_train_tgt[:, : , :min_dim]
            X_test = X_test[:, :, :min_dim]
            pooled_data = [data[:, :, :min_dim] for data in pooled_data]

        X_train_cross = np.concatenate(pooled_data, axis=0)
        y_train_cross = np.concatenate(pooled_labels, axis=0)
    else:
        # Placeholders for cross-patient data in patient-specific case
        X_train_cross = None
        y_train_cross = None
        
    if cfg.compute_chance:
        # Shuffle training labels by trial for chance performance
        rand_idx = np.random.permutation(y_train_tgt.shape[0])
        y_train_tgt = y_train_tgt[rand_idx]

    # Define data module
    augs_list = [AUGS_DICT[aug] for aug in cfg.training.augmentations]
    dm = select_datamodule(cfg, X_train_tgt, y_train_tgt, X_train_cross,
                            y_train_cross, X_test, y_test,
                            cfg.training.batch_size, cfg.training.val_size,
                            augs_list, cfg.paths.nn_data_dir)
    dm.setup()

    pers_all = []
    logits_all = []
    for i in range(cfg.training.n_iter):

        # Define model
        model = RealtimeRNNModel(
            input_size=X_train_tgt.shape[-1]*cfg.model.win_size,
            hidden_size=cfg.model.hidden_size,
            n_layers=cfg.model.n_layers,
            n_classes=len(PHON_DICT),
            dropout=cfg.model.dropout,
            win_size=cfg.model.win_size,
            stride=cfg.model.stride,
            learning_rate=cfg.training.learning_rate,
            decay_steps=cfg.training.n_epochs,
            weight_decay=cfg.model.l2_reg,
        )

        # Model training
        callbacks = [
            ModelCheckpoint(monitor='val_PER', mode='min')
        ]

        trainer = L.Trainer(
            accelerator='auto',
            max_epochs=cfg.training.n_epochs,
            gradient_clip_val=cfg.training.gclip_val,
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
        dm = CTCHeldOutTargetValDataModule(
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


def make_chance_labels(n_trials, seq_length, n_phonemes=9, n_sil=2):
    labels = np.random.randint(1, n_phonemes + 1, size=(n_trials, seq_length - 2 * n_sil))
    for _ in range(n_sil):
        labels = np.insert(labels, 0, SIL_TOKEN, axis=1)
        labels = np.insert(labels, labels.shape[1], SIL_TOKEN, axis=1)
    return labels


def load_pca_xform(pca_path, pt):
    with h5py.File(pca_path, 'r') as f:
        # load PCA components - transpose for projection to latent space
        pca_xform = f[f'{pt}/components'][:].T
    return pca_xform


def load_cca_xform(cca_path, target_pt, source_pt):
    with h5py.File(cca_path, 'r') as f:
        cca_xform = f[f'{source_pt}_to_{target_pt}/components'][:]
    return cca_xform


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
