"""Hyperparameter tuning for CTC-RNN models with online CCA alignment.

Uses Ray Tune with random search or BOHB to optimize RNN hyperparameters
for CTC-based phoneme decoding with compute-time CCA alignment of
cross-patient data. After tuning, retrains the best configuration over
multiple iterations and evaluates phoneme error rate on held-out test data.
"""

import numpy as np
import h5py
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import ray
from ray import tune
from ray.util import ActorPool
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ConfigSpace import (ConfigurationSpace, UniformFloatHyperparameter,
                         UniformIntegerHyperparameter,
                         CategoricalHyperparameter)
from torchaudio.functional import edit_distance
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

import warnings
warnings.filterwarnings('ignore')

# sys.path.append(str(Path('~/repos/cross_patient_speech_decoding/aligned_decoding').expanduser()))
sys.path.append(str(Path('~/repos/cross_patient_micro/cross_patient_speech_decoding/aligned_decoding').expanduser()))
# import realtime_sim
# import alignment
from realtime_sim.ctc_decoder import greedy_decode_batch
from realtime_sim.realtime_datamodule import (CTCHeldOutDataModule,
                                              CTCHeldOutTargetValAlignDataModule,
                                              CTCHeldOutTargetValAlignCVDataModule)
from realtime_sim.realtime_nn_model import RealtimeRNNModel
import realtime_sim.augmentations as augs

# ray.init(runtime_env={'py_modules': [realtime_sim]}, num_cpus=5, num_gpus=1, object_store_memory=16e9, include_dashboard=False, _temp_dir=str(Path('~/workspace/ray_tmp').expanduser()), ignore_reinit_error=True)
ray.init(num_cpus=10, num_gpus=1, object_store_memory=16e9, include_dashboard=False, _temp_dir=str(Path('~/workspace/ray_tmp').expanduser()), ignore_reinit_error=True)
# ray.init(runtime_env={'py_modules': [realtime_sim, alignment]}, num_cpus=10, num_gpus=1, object_store_memory=16e9, include_dashboard=False, _temp_dir=str(Path('E:/workspace/ray_tmp').expanduser()), ignore_reinit_error=True)

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


@hydra.main(version_base=None, config_path="config", config_name="tune_ctc_rnn_config")
def main(cfg: DictConfig) -> None:
    """Entry point for CTC-RNN tuning with online CCA alignment.

    Loads patient data (keeping per-patient arrays separate for online
    alignment), runs Ray Tune hyperparameter optimization, retrains the
    best configuration over multiple iterations, and saves results.

    Args:
        cfg: Hydra DictConfig with paths, data processing, model,
            training, and tuning parameters.

    Raises:
        RuntimeError: If required configuration keys are missing.
    """
    # Check for missing keys in config
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')
    
    print('Configuration:')
    for key, value in cfg.items():
        print(f'  {key}: {value}')
    
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

    ##### Hyperparameter optimization #####
    if cfg.tuning.algo == 'random':
        search_space = {
            'hidden_size': tune.choice([128, 256, 512]),
            'n_layers': tune.choice([2, 3, 4, 5]),
            'dropout': tune.choice([0.2, 0.3, 0.4]),
            # 'learning_rate': tune.loguniform(1e-4, 5e-3),
            'learning_rate': tune.choice([1e-4, 5e-4, 1e-3, 5e-3]),
            'batch_size': tune.choice([128, 256]),
            # 'l2_reg': tune.loguniform(1e-6, 1e-3),
            'l2_reg': tune.choice([1e-5, 1e-4, 1e-3]),
            'gclip_val': tune.choice([5.0]),
            }
    elif cfg.tuning.algo == 'bohb':
        search_space = ConfigurationSpace({
            'hidden_size': UniformIntegerHyperparameter('hidden_size', lower=128, upper=512),
            'n_layers': UniformIntegerHyperparameter('n_layers', lower=2, upper=5),
            'dropout': UniformFloatHyperparameter('dropout', lower=0.1, upper=0.5),
            'learning_rate': UniformFloatHyperparameter('learning_rate', lower=1e-5, upper=1e-2, log=True),
            'batch_size': CategoricalHyperparameter('batch_size', choices=[64, 128, 256, 512]),
            'l2_reg': UniformFloatHyperparameter('l2_reg', lower=1e-6, upper=1e-3, log=True),
            'gclip_val': CategoricalHyperparameter('gclip_val', choices=[5.0]),
        })
    else:
        raise ValueError(f'Unknown tuning algorithm: {cfg.tuning.algo}')

    tune_results = tune_func(search_space,
                             cfg=cfg,
                             X_train_tgt=X_train_tgt,
                             y_train_tgt=y_train_tgt,
                             X_train_cross=X_train_cross,
                             y_train_cross=y_train_cross,
                             X_test=X_test,
                             y_test=y_test,
                             algo=cfg.tuning.algo,
                             cv=cfg.tuning.do_cv,
                            )
    best_tune = tune_results.get_best_result(metric='val_PER', mode='min')
    best_tune_cfg = best_tune.config
    print(f'Best hyperparameters from tuning: {best_tune_cfg}')
    ############

    # Retrain model over multiple iterations with optimal hyperparamters
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
            hidden_size=best_tune_cfg['hidden_size'],
            n_layers=best_tune_cfg['n_layers'],
            n_classes=len(PHON_DICT),
            dropout=best_tune_cfg['dropout'],
            win_size=cfg.model.win_size,
            stride=cfg.model.stride,
            learning_rate=best_tune_cfg['learning_rate'],
            decay_steps=cfg.training['n_epochs'],
            weight_decay=best_tune_cfg['l2_reg'],
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

    # base_dir = Path(cfg.paths.nn_log_dir)
    # base_dir.mkdir(parents=True, exist_ok=True)

    # completed_iters = get_completed_iters(base_dir, cfg.training.n_iter)
    # pending_iters = [
    #     i for i in range(cfg.training.n_iter)
    #     if i not in completed_iters
    # ]

    # print(
    #     f"[Resume] Found {len(completed_iters)} completed / "
    #     f"{cfg.training.n_iter} total iterations",
    #     flush=True,
    # )

    # futures = [
    #     run_single_iteration.remote(
    #         i,
    #         cfg,
    #         best_tune_cfg,
    #         X_train_tgt,
    #         y_train_tgt,
    #         X_train_cross,
    #         y_train_cross,
    #         X_test,
    #         y_test,
    #     )
    #     for i in pending_iters
    # ]

    # if futures:
    #     pool = ActorPool(futures)
    #     for result in pool.map_unordered(lambda a, v: a.get_result()):
    #         print(
    #             f"[Done] Iter {result['iter']} "
    #             f"PER={result['per']:.3f}",
    #             flush=True,
    #         )

    # results = []
    # for i in range(cfg.training.n_iter):
    #     result_file = base_dir / f"iter_{i}" / "result.pt"
    #     if not result_file.exists():
    #         raise RuntimeError(f"Missing result for iteration {i}")
    #     results.append(torch.load(result_file))

    # pers_all = [r["per"] for r in results]
    # logits_all = [r["logits"] for r in results]

    # save_results(
    #     cfg.paths.nn_log_dir,
    #     cfg,
    #     cfg.target_pt,
    #     pers_all,
    #     logits_all,
    #     PHON_DICT,
    #     results[0]["hparams"],
    # )

    # print(
    #     f"[Success] All {cfg.training.n_iter} iterations complete.",
    #     flush=True,
    # )
    # print(f'Final results saved to {cfg.paths.results_dir}', flush=True)


class TuneReportBestMetricCallback(L.Callback):
    """Lightning callback that reports the best metric value to Ray Tune.

    Tracks the running best of a monitored metric and reports it to
    Ray Tune after each validation epoch for early stopping decisions.

    Attributes:
        metric: Name of the metric to monitor.
        mode: One of 'min' or 'max'.
        best: Current best metric value observed.
    """

    def __init__(self, metric="val_PER", mode="min"):
        """Initialize the callback.

        Args:
            metric: Name of the metric to track.
            mode: 'min' to track the lowest value, 'max' for highest.
        """
        super().__init__()
        self.metric = metric
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def on_validation_end(self, trainer, pl_module):
        """Update best metric and report to Ray Tune.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module being trained.
        """
        metrics = trainer.callback_metrics
        if self.metric not in metrics:
            return

        current = metrics[self.metric].item()
        if (self.mode == "min" and current < self.best) or \
           (self.mode == "max" and current > self.best):
            self.best = current

        tune.report({self.metric: self.best})

class BestMetricCallback(L.Callback):
    """Lightning callback that tracks the best metric value locally.

    Used during cross-validation folds to record the best validation
    metric without reporting to Ray Tune.

    Attributes:
        metric: Name of the metric to monitor.
        mode: One of 'min' or 'max'.
        best: Current best metric value observed.
    """

    def __init__(self, metric="val_PER", mode="min"):
        """Initialize the callback.

        Args:
            metric: Name of the metric to track.
            mode: 'min' to track the lowest value, 'max' for highest.
        """
        self.metric = metric
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def on_validation_end(self, trainer, pl_module):
        """Update the best metric value if improved.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module being trained.
        """
        metrics = trainer.callback_metrics
        if self.metric not in metrics:
            return

        current = metrics[self.metric].item()
        if (self.mode == "min" and current < self.best) or \
           (self.mode == "max" and current > self.best):
            self.best = current


def train_func(hp_config, cfg=None, X_train_tgt=None, y_train_tgt=None, X_train_cross=None, y_train_cross=None, X_test=None, y_test=None):
    """Ray Tune trainable for a single hyperparameter trial with alignment.

    Builds an alignment-aware data module, trains the CTC-RNN model,
    and reports the best validation PER to Ray Tune.

    Args:
        hp_config: Dictionary of hyperparameters from the search space.
        cfg: Hydra DictConfig with model and training settings.
        X_train_tgt: Target patient training features.
        y_train_tgt: Target patient training labels.
        X_train_cross: List of cross-patient training feature arrays.
        y_train_cross: List of cross-patient training label arrays.
        X_test: Test features array.
        y_test: Test labels array.
    """
    # Setup data module
    trial_dir = Path(tune.get_context().get_trial_dir())
    dm_path = trial_dir

    aug_list = [AUGS_DICT[aug] for aug in cfg.training.augmentations]

    # shuffle chance labels for each CV run
    if cfg.compute_chance:
        # Shuffle training labels by trial for chance performance
        rand_idx = np.random.permutation(y_train_tgt.shape[0])
        y_train_tgt = y_train_tgt[rand_idx]

    dm = select_datamodule(cfg, X_train_tgt, y_train_tgt, X_train_cross,
                            y_train_cross, X_test, y_test,
                            int(hp_config['batch_size']), cfg.training.val_size,
                            aug_list, dm_path)
    dm.setup()

    # Define model
    data_shapes = dm.get_data_shape()
    model = RealtimeRNNModel(
        input_size=data_shapes*cfg.model.win_size,
        hidden_size=hp_config['hidden_size'],
        n_layers=hp_config['n_layers'],
        n_classes=len(PHON_DICT),
        dropout=hp_config['dropout'],
        win_size=cfg.model.win_size,
        stride=cfg.model.stride,
        learning_rate=hp_config['learning_rate'],
        decay_steps=cfg.training.n_epochs,
        weight_decay=hp_config['l2_reg'],
    )

    # Model training
    callbacks = [
        # TuneReportCheckpointCallback()
        TuneReportBestMetricCallback(metric='val_PER', mode='min')
    ]

    trainer = L.Trainer(
        accelerator='auto',
        max_epochs=cfg.training.n_epochs,
        gradient_clip_val=float(hp_config['gclip_val']),
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader())


def train_func_cv(hp_config, cfg=None, X_train_tgt=None, y_train_tgt=None, X_train_cross=None, y_train_cross=None, X_test=None, y_test=None):
    """Ray Tune trainable with cross-validated evaluation and alignment.

    Trains the CTC-RNN model across multiple folds using an
    alignment-aware CV data module, aggregates the best validation PER
    per fold, and reports the mean to Ray Tune.

    Args:
        hp_config: Dictionary of hyperparameters from the search space.
        cfg: Hydra DictConfig with model, training, and tuning settings.
        X_train_tgt: Target patient training features.
        y_train_tgt: Target patient training labels.
        X_train_cross: List of cross-patient training feature arrays.
        y_train_cross: List of cross-patient training label arrays.
        X_test: Test features array.
        y_test: Test labels array.
    """

    trial_dir = Path(tune.get_context().get_trial_dir())
    dm_path = trial_dir

    aug_list = [AUGS_DICT[aug] for aug in cfg.training.augmentations]

    # shuffle chance labels for each CV run
    if cfg.compute_chance:
        # Shuffle training labels by trial for chance performance
        rand_idx = np.random.permutation(y_train_tgt.shape[0])
        y_train_tgt = y_train_tgt[rand_idx]

    dm = CTCHeldOutTargetValAlignCVDataModule(
            X_train_tgt,
            y_train_tgt,
            X_train_cross,
            y_train_cross,
            X_test,
            y_test,
            batch_size=int(hp_config['batch_size']),
            n_folds=cfg.tuning.n_folds,
            augmentations=aug_list,
            data_path=dm_path / f'{cfg.target_pt}_data',
            pool=cfg.pool_train,
            align=cfg.align_train,
        )
    dm.setup()

    fold_metrics = []

    for fold in range(dm.n_folds):
        dm.set_fold(fold)

        data_shapes = dm.get_data_shape()
        model = RealtimeRNNModel(
            input_size=data_shapes[-1] * cfg.model.win_size,
            hidden_size=hp_config['hidden_size'],
            n_layers=hp_config['n_layers'],
            n_classes=len(PHON_DICT),
            dropout=hp_config['dropout'],
            win_size=cfg.model.win_size,
            stride=cfg.model.stride,
            learning_rate=hp_config['learning_rate'],
            decay_steps=cfg.training.n_epochs,
            weight_decay=hp_config['l2_reg'],
        )

        best_cb = BestMetricCallback(metric="val_PER", mode="min")

        trainer = L.Trainer(
            accelerator="auto",
            max_epochs=cfg.training.n_epochs,
            gradient_clip_val=float(hp_config['gclip_val']),
            callbacks=[best_cb],
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(
            model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        fold_metrics.append(best_cb.best)
    
    # report aggregate across folds
    fold_metrics = torch.tensor(fold_metrics)
    tune.report({
        # just call this val_PER for consistency with non-CV case
        "val_PER": fold_metrics.mean().item(),  
    })

    
def tune_func(search_space, cfg, X_train_tgt, y_train_tgt, X_train_cross, y_train_cross, X_test, y_test, algo='random', cv=False):
    """Launch Ray Tune hyperparameter optimization.

    Args:
        search_space: Ray Tune or ConfigSpace search space definition.
        cfg: Hydra DictConfig with training and tuning parameters.
        X_train_tgt: Target patient training features.
        y_train_tgt: Target patient training labels.
        X_train_cross: List of cross-patient training feature arrays.
        y_train_cross: List of cross-patient training label arrays.
        X_test: Test features array.
        y_test: Test labels array.
        algo: Tuning algorithm, either 'random' or 'bohb'.
        cv: If True, use cross-validated training function.

    Returns:
        ray.tune.ResultGrid: Results from the tuning run.

    Raises:
        ValueError: If an unknown tuning algorithm is specified.
    """
    if cv:
        train_fn = train_func_cv
    else:
        train_fn = train_func

    trainable = tune.with_resources(
                                   tune.with_parameters(
                                       train_fn,
                                       cfg=cfg,
                                       X_train_tgt=X_train_tgt,
                                       y_train_tgt=y_train_tgt,
                                       X_train_cross=X_train_cross,
                                       y_train_cross=y_train_cross,
                                       X_test=X_test,
                                       y_test=y_test,
                                   ), 
                                   resources={'CPU': 1, 'GPU': 1},
                )

    if algo == 'random':
        tuner = create_rndm_tuner(search_space, cfg, trainable)
    elif algo == 'bohb':
        tuner = create_bohb_tuner(search_space, cfg, trainable)
    else:
        raise ValueError(f'Unknown tuning algorithm: {algo}')

    return tuner.fit()

def create_rndm_tuner(search_space, cfg, trainable):
    """Create a Ray Tune Tuner with random search.

    Args:
        search_space: Dictionary defining the hyperparameter search space.
        cfg: Hydra DictConfig with tuning.n_trials.
        trainable: Ray Tune trainable (with resources and parameters).

    Returns:
        tune.Tuner: Configured tuner for random search.
    """
    # scheduler = ASHAScheduler(max_t=cfg.training.n_epochs,
    #                           grace_period=cfg.tuning.burn_in,
    #                           reduction_factor=2,
    #                          )
    
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric='val_PER',
            mode='min',
            num_samples=cfg.tuning.n_trials,
            # scheduler=scheduler,
            trial_dirname_creator=lambda trial: f'trial_{trial.trial_id}',
        ),
        run_config=tune.RunConfig(
            storage_path=Path('~/workspace/ray_results').expanduser(),
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=50,
                checkpoint_score_attribute='val_PER',
                checkpoint_score_order='min',
            ),
        ),
    )
    return tuner
    

def create_bohb_tuner(search_space, cfg, trainable):
    """Create a Ray Tune Tuner with BOHB (Bayesian Optimization HyperBand).

    Args:
        search_space: ConfigurationSpace defining the search space.
        cfg: Hydra DictConfig with tuning.n_trials and
            training.n_epochs.
        trainable: Ray Tune trainable (with resources and parameters).

    Returns:
        tune.Tuner: Configured tuner for BOHB search.
    """
    scheduler = HyperBandForBOHB(
        max_t=cfg.training.n_epochs,
    )
    algo = TuneBOHB(
        search_space,
        metric='val_PER',
        mode='min',
    )
    
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric='val_PER',
            mode='min',
            num_samples=cfg.tuning.n_trials,
            search_alg=algo,
            scheduler=scheduler,
        ),
        run_config=tune.RunConfig(
            storage_path=Path('~/workspace/ray_results').expanduser(),
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=50,
                checkpoint_score_attribute='val_PER',
                checkpoint_score_order='min',
            ),
        ),
    )
    return tuner


@ray.remote(num_gpus=0.1, num_cpus=1, max_retries=2, retry_exceptions=True)
def run_single_iteration(
    iter_idx,
    cfg,
    best_tune_cfg,
    X_train_tgt,
    y_train_tgt,
    X_train_cross,
    y_train_cross,
    X_test,
    y_test,
):
    """Ray remote task that trains and evaluates a single iteration.

    Builds its own DataModule with alignment support, trains the model
    with the best hyperparameters, evaluates on test data, and saves
    the result checkpoint to disk.

    Args:
        iter_idx: Iteration index for file naming.
        cfg: Hydra DictConfig with model, training, and path settings.
        best_tune_cfg: Dictionary of best hyperparameters from tuning.
        X_train_tgt: Target patient training features.
        y_train_tgt: Target patient training labels.
        X_train_cross: List of cross-patient training feature arrays.
        y_train_cross: List of cross-patient training label arrays.
        X_test: Test features array.
        y_test: Test labels array.

    Returns:
        dict: Result with keys 'iter', 'per', 'logits', 'hparams'.
    """
    # IMPORTANT: each worker must build its own DataModule
    dm = select_datamodule(
        cfg,
        X_train_tgt,
        y_train_tgt,
        X_train_cross,
        y_train_cross,
        X_test,
        y_test,
        best_tune_cfg["batch_size"],
        cfg.training.val_size,
        [AUGS_DICT[aug] for aug in cfg.training.augmentations],
        cfg.paths.nn_log_dir / f"iter_{iter_idx}",
    )
    dm.setup()

    data_shapes = dm.get_data_shape()
    model = RealtimeRNNModel(
        input_size=data_shapes[-1] * cfg.model.win_size,
        hidden_size=best_tune_cfg["hidden_size"],
        n_layers=best_tune_cfg["n_layers"],
        n_classes=len(PHON_DICT),
        dropout=best_tune_cfg["dropout"],
        win_size=cfg.model.win_size,
        stride=cfg.model.stride,
        learning_rate=best_tune_cfg["learning_rate"],
        decay_steps=cfg.training.n_epochs,
        weight_decay=best_tune_cfg["l2_reg"],
    )

    ckpt_cb = ModelCheckpoint(monitor="val_PER", mode="min")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.training.n_epochs,
        gradient_clip_val=float(best_tune_cfg["gclip_val"]),
        callbacks=[ckpt_cb],
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=cfg.paths.nn_log_dir,
    )

    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    # Load best checkpoint
    best_model = RealtimeRNNModel.load_from_checkpoint(ckpt_cb.best_model_path)
    best_model.eval().cuda()

    test_loader = dm.test_dataloader()
    X_batch, y_batch, _, _ = next(iter(test_loader))
    X_batch = X_batch.cuda()

    with torch.no_grad():
        logits = torch.nn.functional.log_softmax(
            best_model(X_batch), dim=-1
        )

    per = calc_norm_edit_distance(logits, y_batch) * 100

    result = {
        "iter": iter_idx,
        "per": per,
        "logits": logits.cpu().numpy(),
        "hparams": model.hparams,
    }

    out_dir = cfg.paths.nn_log_dir / f"iter_{iter_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(result, out_dir / "result.pt")

    return result


def make_logger(log_dir, pt, cfg):
    """Create a TensorBoard logger with a context-aware experiment name.

    Args:
        log_dir: Directory for TensorBoard log files.
        pt: Patient identifier string.
        cfg: Hydra DictConfig with pool_train, align_train, and
            compute_chance flags.

    Returns:
        TensorBoardLogger: Configured logger instance.
    """
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


def get_completed_iters(base_dir, n_iter):
    """Find which training iterations have already been completed.

    Args:
        base_dir: Path to the base directory containing iteration
            subdirectories.
        n_iter: Total number of expected iterations.

    Returns:
        set[int]: Indices of iterations with saved result files.
    """
    completed = set()
    for i in range(n_iter):
        result_file = base_dir / f"iter_{i}" / "result.pt"
        if result_file.exists():
            completed.add(i)
    return completed    


def load_data(data_filename, pt, tw_select, tw_orig, zscore=False, only_train=False, load_all=False, n_sil=2):
    """Load neural feature data and labels from an HDF5 file.

    Args:
        data_filename: Path to the HDF5 data file.
        pt: Patient identifier string.
        tw_select: Two-element sequence [start, end] for the desired time
            window in seconds.
        tw_orig: Two-element sequence [start, end] of the original time
            window in the data.
        zscore: If True, load z-scored features.
        only_train: If True, skip loading test data.
        load_all: If True, concatenate train and test into a single
            training set.
        n_sil: Number of silence tokens to prepend/append to labels.

    Returns:
        tuple: (feats_train, labels_train, feats_test, labels_test).
            Test arrays are None when only_train or load_all is True.
    """
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


def make_chance_labels(n_trials, seq_length, n_phonemes=9, n_sil=2):
    """Generate random phoneme label sequences for chance-level evaluation.

    Args:
        n_trials: Number of trials to generate.
        seq_length: Total sequence length including silence tokens.
        n_phonemes: Number of distinct phoneme classes (excluding silence).
        n_sil: Number of silence tokens to prepend/append.

    Returns:
        np.ndarray: Random label array of shape (n_trials, seq_length).
    """
    labels = np.random.randint(1, n_phonemes + 1, size=(n_trials, seq_length - 2 * n_sil))
    for _ in range(n_sil):
        labels = np.insert(labels, 0, SIL_TOKEN, axis=1)
        labels = np.insert(labels, labels.shape[1], SIL_TOKEN, axis=1)
    return labels


def select_datamodule(cfg, X_train_tgt, y_train_tgt, X_train_cross,
                      y_train_cross, X_test, y_test, batch_size,
                      val_size, augmentations, data_dir):
    """Instantiate the appropriate alignment-aware Lightning data module.

    Args:
        cfg: Hydra DictConfig with pool_train, align_train, and
            target_pt fields.
        X_train_tgt: Target patient training features array.
        y_train_tgt: Target patient training labels array.
        X_train_cross: List of cross-patient training feature arrays.
        y_train_cross: List of cross-patient training label arrays.
        X_test: Test features array.
        y_test: Test labels array.
        batch_size: Training batch size.
        val_size: Fraction of training data for validation.
        augmentations: List of augmentation functions.
        data_dir: Path for caching data module state.

    Returns:
        LightningDataModule: Configured data module for CTC training
            with alignment.
    """
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


def calc_norm_edit_distance(input_seqs, target_seqs):
    """Compute token-level normalized edit distance via greedy CTC decoding.

    Args:
        input_seqs: Log-softmax output tensor of shape
            (batch, time, n_classes).
        target_seqs: Ground-truth label sequences.

    Returns:
        float: Total edit distance divided by total target tokens.
    """
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
    """Save decoding results and model hyperparameters to an HDF5 file.

    Args:
        save_dir: Base directory for saving results.
        cfg: Hydra DictConfig with data_proc, pool_train, align_train,
            and compute_chance fields.
        pt: Patient identifier string.
        pers_all: List of phoneme error rates across iterations.
        logits_all: List of logit arrays across iterations.
        phon_dict: Mapping from token indices to phoneme strings.
        model_hparams: Dictionary of model hyperparameters to store.
    """
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
