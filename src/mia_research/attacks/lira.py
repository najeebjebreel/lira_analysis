"""
Module implementing the LiRA membership inference pipeline: generating logits, scoring, and plotting.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mia_research.models.model_utils import get_model
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import roc_curve, auc, precision_score
# Font compatibility for PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from mia_research.data.data_utils import load_dataset_for_mia_inference

class LiRA:
    """
    LiRA pipeline: inference, scoring, and plotting.
    Enhanced with Google-style spatial augmentations.
    """
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.info("Initializing Enhanced LiRA...")
        self.config = config
      
        # Experiment directory
        self.experiment_dir = config.get('experiment', {}).get('checkpoint_dir', 'experiments')
        self.logger.info(f"Experiment directory: {self.experiment_dir}")

        # Load dataset
        self.logger.info("Loading dataset...")
        self.full_dataset, self.labels = load_dataset_for_mia_inference(config)
        self.logger.info("Dataset loaded...")
        self.logger.info("full_dataset size: %d", len(self.full_dataset))
        
        # Enhanced augmentation configuration
        attack_cfg = config.get('attack', {})
        self.target_model = attack_cfg.get('target_model', 'best')
        self.prior = attack_cfg.get('prior', 0.5)
        
        # Parse augmentation config with spatial parameters
        aug_config = config.get('inference_data_augmentations', {})
        if isinstance(aug_config, str):
            aug_config = {'type': aug_config}
        
        self.aug_type = aug_config.get('type', 'none')
        self.spatial_shift = aug_config.get('spatial_shift', 0)  # How many pixels to shift
        self.spatial_stride = aug_config.get('spatial_stride', 1)  # Stride for spatial grid
        self.use_horizontal_flip = aug_config.get('horizontal_flip', True)
        
        self.logger.info(f"Augmentation type: {self.aug_type}")
        self.logger.info(f"Spatial shift: {self.spatial_shift}, stride: {self.spatial_stride}")
        self.logger.info(f"Horizontal flip: {self.use_horizontal_flip}")
        
        # Get device
        use_cuda = config.get('use_cuda', True)
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # DataLoader (use num_workers from config)
        train_cfg    = config.get('training', {})
        batch_size   = train_cfg.get('batch_size', 128)
        num_workers  = train_cfg.get('num_workers', 0)
        self.data_loader = DataLoader(
            self.full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Model
        model_cfg = config.get('model', {})
        num_classes = config.get('dataset', {}).get('num_classes', 10)
        self.model = get_model(num_classes, **model_cfg)
        self.model = self.model.to(self.device)

        # Shadow models
        self.num_shadow_models = train_cfg.get('num_shadow_models', 1)
        self.logger.info(f"Using {self.num_shadow_models} shadow models")

        self.keep_indices_path = os.path.join(self.experiment_dir, 'keep_indices.npy')
        self.keep_indices = np.load(self.keep_indices_path) 
        self.logger.info(f"Loaded keep_indices from {self.keep_indices_path}")


    def get_model_logits(self, data_loader=None):
        """
        Run inference under configured augmentations, return logits [N, A, C].
        """
        if data_loader is None:
            data_loader = self.data_loader
        
        all_logits = []
    
        self.model.eval()   # Standard evaluation mode
        
        with torch.no_grad():
            for images, _ in tqdm(data_loader, desc="Processing batches"):
                images = images.to(self.device)
                batch_logits = []
                
                # Always start with original images (no augmentation)
                original_logits = self.model(images)
                batch_logits.append(original_logits)
                
                # Apply horizontal flip if enabled
                if self.use_horizontal_flip:
                    flipped = torch.flip(images, dims=[3])
                    flipped_logits = self.model(flipped)
                    batch_logits.append(flipped_logits)
                
                # Apply spatial augmentations if enabled
                if self.spatial_shift > 0:
                    # Apply reflection padding
                    padded = F.pad(images, 
                                [self.spatial_shift, self.spatial_shift,  # left, right
                                self.spatial_shift, self.spatial_shift], # top, bottom
                                mode='reflect')
                    
                    # Generate spatial grid augmentations
                    for dx in range(0, 2 * self.spatial_shift + 1, self.spatial_stride):
                        for dy in range(0, 2 * self.spatial_shift + 1, self.spatial_stride):
                            # Skip the center crop if it's the same as original
                            if dx == self.spatial_shift and dy == self.spatial_shift:
                                continue
                                
                            cropped = padded[:, :, 
                                        dy:dy + images.shape[2], 
                                        dx:dx + images.shape[3]]
                            cropped_logits = self.model(cropped)
                            batch_logits.append(cropped_logits)
                            
                            # Also apply horizontal flip to spatial augmentations if enabled
                            if self.use_horizontal_flip:
                                cropped_flipped = torch.flip(cropped, dims=[3])
                                cropped_flipped_logits = self.model(cropped_flipped)
                                batch_logits.append(cropped_flipped_logits)
                
                # Stack all augmentations for this batch: [B, A, C]
                batch_logits = torch.stack(batch_logits, dim=1)
                all_logits.append(batch_logits)

        result = torch.cat(all_logits, dim=0)  # [N, A, C]
        num_augs = result.shape[1]
        self.logger.info(f"Generated logits with {num_augs} augmentations per sample")
        return result

    def generate_logits(self):
        """
        Inference for each shadow model; save logits under model_i/logits/logits.npy
        """
        if self.target_model == 'best':
            target_model_prefix = 'best_model.pth'
        else:
            target_model_prefix = f"checkpoint_epoch{self.target_model}.pth"

        for i in range(self.num_shadow_models):
            self.logger.info(f"Inference using shadow model {i}...")
            ckpt = os.path.join(self.experiment_dir, f'model_{i}', target_model_prefix)
            if not os.path.exists(ckpt):
                self.logger.warning(f"Checkpoint not found: {ckpt}")
                continue

            logits_dir = os.path.join(self.experiment_dir, f'model_{i}', 'logits')
            os.makedirs(logits_dir, exist_ok=True)
            logits_path = os.path.join(logits_dir, 'logits.npy')
            if os.path.exists(logits_path) and self.config.get('experiment', {}).get('overwrite_logits', False) is False:
                self.logger.info(f"Logits already exist for model {i}, skipping")
                continue

            try:
                self.logger.info(f"Loading checkpoint from {ckpt}")
                checkpoint = torch.load(ckpt, map_location=self.device)
                state = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state)

                aug_desc = f"{self.aug_type}"
                if self.spatial_shift > 0:
                    aug_desc += f" (shift={self.spatial_shift}, stride={self.spatial_stride})"
                if self.use_horizontal_flip:
                    aug_desc += " + horizontal_flip"
                    
                self.logger.info(f"Running inference with augmentations: {aug_desc}")
                logits = self.get_model_logits()
                logits = logits.cpu().unsqueeze(1).numpy()  # [N,1,A,C]
                self.logger.info(f"Logits shape: {logits.shape}")
                np.save(logits_path, logits)  # [N,1,A,C]
                self.logger.info(f"Saved logits to {logits_path}")

            except Exception as e:
                self.logger.error(f"Error on model {i}: {e}")
        self.logger.info("Completed logits generation for all models")
        
    def compute_scores(self):
        """
        For each shadow model i:
        - load logits from model_i/logits/logits.npy ([N,1,A,C] or [N,A,C])
        - compute softmax over classes
        - compute y_true = average true‐class probability over all augs
        - compute y_wrong = average of remaining probability mass over all augs
        - score = log(y_true) - log(y_wrong)
        - save score array (shape [N]) to model_i/scores/scores.npy
        """

        for i in range(self.num_shadow_models):
            # where the logits live
            logits_path = os.path.join(self.experiment_dir,
                                    f"model_{i}",
                                    "logits",
                                    "logits.npy")
            if not os.path.exists(logits_path):
                self.logger.warning(f"No logits for model_{i}, skipping score computation")
                continue

            # prepare output dir
            scores_dir = os.path.join(self.experiment_dir,
                                    f"model_{i}",
                                    "scores")
            os.makedirs(scores_dir, exist_ok=True)
            if os.path.exists(os.path.join(scores_dir, "scores.npy")) and self.config.get('experiment', {}).get('overwrite_scores', False) is False:
                self.logger.info(f"Scores already exist for model {i}, skipping")
                continue

            # load logits
            try:
                opredictions = np.load(logits_path)  # [N,1,A,C] or [N,A,C]
                if opredictions.shape[2] > 2:
                    opredictions = opredictions[:, :, 0:opredictions.shape[2], :]
                self.logger.info(f"Loaded logits for model_{i}: shape {opredictions.shape}")
            except Exception as e:
                self.logger.error(f"Error loading logits for model_{i}: {e}")
                continue        

            ## Be exceptionally careful.
            ## Numerically stable everything, as described in the paper.
            predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
            predictions = np.array(np.exp(predictions), dtype=np.float64)
            predictions = predictions/np.sum(predictions,axis=3,keepdims=True)

            COUNT = predictions.shape[0]
            #  x num_examples x num_augmentations x logits
            y_true = predictions[np.arange(COUNT),:,:,self.labels[:COUNT]]
  
            predictions[np.arange(COUNT),:,:,self.labels[:COUNT]] = 0
            y_wrong = np.sum(predictions, axis=3)

            logit = (np.log(y_true.mean((1))+1e-45) - np.log(y_wrong.mean((1))+1e-45))

            # save out
            outpath = os.path.join(scores_dir, "scores.npy")
            np.save(outpath, logit)
            self.logger.info(f"Saved scores for model_{i} to {outpath}")

        self.logger.info("Completed score computation for all shadow models")

    
    def plot(self, ntest=1, metric='auc'):
        """
        Main plotting function that handles different evaluation modes.
        """
        attack_cfg = self.config.get('attack', {})
        evaluation_mode = attack_cfg.get('evaluation_mode', 'single')
        
        if evaluation_mode == 'single':
            self._plot_single_target(ntest=ntest, metric=metric)
        elif evaluation_mode == 'leave_one_out':
            self._plot_leave_one_out(metric=metric)
        elif evaluation_mode == 'both':
            self._plot_single_target(ntest=ntest, metric=metric)
            self._plot_leave_one_out(metric=metric)
        else:
            raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")

    def _plot_single_target(self, ntest=1, metric='auc'):
        """
        Original single target evaluation mode.
        """
        self.logger.info(f"Running single target evaluation with ntest={ntest}")
        
        # --- load scores and keep_indices ---
        scores_list = []
        for i in range(self.num_shadow_models):
            path = os.path.join(self.experiment_dir,
                                f"model_{i}", "scores", "scores.npy")
            if os.path.exists(path):
                scores_list.append(np.load(path))
            else:
                self.logger.warning(f"No scores for model_{i}")
        if not scores_list:
            raise RuntimeError("No score files found in any shadow model directories.")
        scores = np.stack(scores_list)          # shape (M_used, N, A)
        self.keep_indices = np.array(self.keep_indices)[:len(scores_list)]  # shape (M_used, N)
        self.logger.info(f"Loaded {scores.shape[0]} models, scores shape: {scores.shape}")

        # First, compute train/test loss & acc stats:
        stats_path = os.path.join(self.experiment_dir, 'train_test_stats.csv')
        self.compute_train_test_stats(save_csv_path=stats_path)
    

        # --- split train/test once ---
        train_keep   = self.keep_indices[:-ntest]
        train_scores = scores[:-ntest]
        test_keep    = self.keep_indices[-ntest:]
        test_scores  = scores[-ntest:]

        # --- run and plot each attack ---
        plt.figure(figsize=(4, 3))
        attacks = [
            (self.generate_ours,           "LiRA (online)",             {'color': 'C0', 'fix_variance': False}),
            (self.generate_ours,           "LiRA (online, fixed var)",  {'color': 'C1', 'fix_variance': True}),
            (self.generate_ours_offline,   "LiRA (offline)",            {'color': 'C2', 'fix_variance': False}),
            (self.generate_ours_offline,   "LiRA (offline, fixed var)", {'color': 'C3', 'fix_variance': True}),
            (self.generate_global,         "Global threshold",          {'color': 'C4'}),
        ]

        results = {}
        tfprs = self.config.get('attack', {}).get('target_fprs', [0.001])
        if not isinstance(tfprs, (list,tuple,np.ndarray)):
            tfprs = [tfprs]

        for fn, label, opts in attacks:
            results[label] = {}
            opts_local = opts.copy()
            fix_var = opts_local.pop('fix_variance', False)
            if label == "Global threshold":
                preds, ans = fn(train_keep, train_scores, test_keep, test_scores)
            else:
                preds, ans = fn(train_keep, train_scores, test_keep, test_scores, fix_variance=fix_var)
            fpr, tpr, auc_v, acc_v, thresholds = self.sweep(np.array(preds), np.array(ans, dtype=bool))
            
            results[label] = {
                'Acc':  acc_v*100,
                'AUC':  auc_v*100,
                'FPRs': [],
                'TPRs': [],
                'Precs': []
            }

            metric_text = f"AUC={auc_v*100:.2f}" if metric == 'auc' else f"Acc={acc_v*100:.2f}"
            plt.plot(fpr, tpr, label=f"{label} ({metric_text})", **opts_local)

            # compute & log TPR@each target FPR
            for tfpr in tfprs:
                idx   = np.where(fpr <= tfpr)[0]
                if idx.size > 0:
                    tpr_at = float(tpr[idx[-1]])
                    actual_fpr = float(fpr[idx[-1]])
                    prec = (self.prior*tpr_at) / (self.prior*tpr_at + (1-self.prior)*actual_fpr + 1e-30)
                else:
                    tpr_at = 0.0
                    actual_fpr = 0.0  # Add this line!
                    prec = 0.0

                self.logger.info(
                    f"{label}: AUC={auc_v*100:.2f}, Acc={acc_v*100:.2f}, "
                    f"TPR@{tfpr*100:.4f}FPR={tpr_at*100:.3f}, "
                    f"Prec@{tfpr*100:.4f}FPR={prec*100:.2f}" 
                )
                
                results[label]['FPRs'].append(tfpr*100)
                results[label]['TPRs'].append(tpr_at*100)
                results[label]['Precs'].append(prec*100)
                
            self.logger.info("---------------------------------------------------------------------")

        plt.semilogx()
        plt.semilogy()
        plt.xlim(1e-5, 1)
        plt.ylim(1e-5, 1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot([1e-5, 1], [1e-5, 1], ls='--', color='gray')
        plt.subplots_adjust(bottom=0.18, left=0.18, top=0.96, right=0.96)
        plt.legend(fontsize=8)

        roc_curve_path = os.path.join(self.experiment_dir, 'roc_curve_single.pdf')
        plt.savefig(roc_curve_path, bbox_inches='tight', dpi=300)
        self.logger.info(f"Saved single target plot to {roc_curve_path}")
        # plt.show()

        # Save results
        rows = []
        for label, v in results.items():
            row = {'Attack': label, 'Acc': np.round(v['Acc'], 2), 'AUC': np.round(v['AUC'], 2)}
            for tfpr, tpr_at, prec_at in zip(v['FPRs'], v['TPRs'], v['Precs']):
                row[f"TPR@{tfpr:.4f}%FPR"]  = np.round(tpr_at, 3)
                row[f"Prec@{tfpr:.4f}%FPR"] = np.round(prec_at, 2)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.experiment_dir, 'attack_results_single.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved single target results to {csv_path}")

    def _plot_leave_one_out(self, metric='auc'):
        """
        Leave-one-out evaluation mode where each shadow model is used as target.
        Computes likelihood ratios for ALL attack variants and saves them efficiently.
        """
        self.logger.info("Running leave-one-out evaluation")
        
        # Load scores and keep_indices
        scores_list = []
        for i in range(self.num_shadow_models):
            path = os.path.join(self.experiment_dir, f"model_{i}", "scores", "scores.npy")
            if os.path.exists(path):
                scores_list.append(np.load(path))
            else:
                self.logger.warning(f"No scores for model_{i}")
        
        if not scores_list:
            raise RuntimeError("No score files found in any shadow model directories.")
        
        scores = np.stack(scores_list)  # shape (M, N, A)
        keep_indices = np.array(self.keep_indices)[:len(scores_list)]  # shape (M, N)
        M, N = keep_indices.shape
        
        self.logger.info(f"Leave-one-out: {M} models, {N} samples each")

        # Get target FPRs from config
        tfprs = self.config.get('attack', {}).get('target_fprs', [0.001])
        if not isinstance(tfprs, (list, tuple, np.ndarray)):
            tfprs = [tfprs]

        # Initialize storage for results across all target models
        all_results = {
            'LiRA (online)': {'AUCs': [], 'Accs': [], 'FPRs': [], 'TPRs': [], 'Precs': []},
            'LiRA (online, fixed var)': {'AUCs': [], 'Accs': [], 'FPRs': [], 'TPRs': [], 'Precs': []},
            'LiRA (offline)': {'AUCs': [], 'Accs': [], 'FPRs': [], 'TPRs': [], 'Precs': []},
            'LiRA (offline, fixed var)': {'AUCs': [], 'Accs': [], 'FPRs': [], 'TPRs': [], 'Precs': []},
            'Global threshold': {'AUCs': [], 'Accs': [], 'FPRs': [], 'TPRs': [], 'Precs': []}
        }

        # Storage for hard membership labels and attack scores (each attack produces different types of scores)
        membership_labels = np.zeros((M, N), dtype=bool)  # M_models x N_samples
        attack_scores = {
            'LiRA (online)': np.zeros((M, N)),           # likelihood ratios
            'LiRA (online, fixed var)': np.zeros((M, N)),  # likelihood ratios
            'LiRA (offline)': np.zeros((M, N)),          # log probabilities
            'LiRA (offline, fixed var)': np.zeros((M, N)),  # log probabilities
            'Global threshold': np.zeros((M, N))         # raw scores
        }
        
        # Storage for threshold information per target FPR
        threshold_info = []

        # Leave-one-out loop
        for target_idx in range(M):
            self.logger.info(f"Evaluating target model {target_idx}/{M-1}")
            
            # Split: all other models as training, current model as target
            train_indices = [i for i in range(M) if i != target_idx]
            train_keep = keep_indices[train_indices]
            train_scores = scores[train_indices]
            test_keep = keep_indices[target_idx:target_idx+1]
            test_scores = scores[target_idx:target_idx+1]
            
            # Store hard labels for this target model
            membership_labels[target_idx] = keep_indices[target_idx]

            # Attack definitions for this target
            attacks = [
                ('LiRA (online)', lambda: self.generate_ours(train_keep, train_scores, test_keep, test_scores, fix_variance=False)),
                ('LiRA (online, fixed var)', lambda: self.generate_ours(train_keep, train_scores, test_keep, test_scores, fix_variance=True)),
                ('LiRA (offline)', lambda: self.generate_ours_offline(train_keep, train_scores, test_keep, test_scores, fix_variance=False)),
                ('LiRA (offline, fixed var)', lambda: self.generate_ours_offline(train_keep, train_scores, test_keep, test_scores, fix_variance=True)),
                ('Global threshold', lambda: self.generate_global(train_keep, train_scores, test_keep, test_scores))
            ]
            
            for attack_name, attack_fn in attacks:
                preds, ans = attack_fn()
                
                # Store attack scores with "higher = member" semantics
                # All attack functions return scores where "lower = member", so negate to flip
                attack_scores[attack_name][target_idx] = -np.array(preds)

                # Compute ROC metrics
                fpr, tpr, auc_v, acc_v, thresholds = self.sweep(np.array(preds), np.array(ans, dtype=bool))
                
                # Store results
                all_results[attack_name]['AUCs'].append(auc_v * 100)
                all_results[attack_name]['Accs'].append(acc_v * 100)

                # Compute TPR and precision at target FPRs
                for tfpr in tfprs:
                    idx = np.where(fpr <= tfpr)[0]
                    if idx.size > 0:
                        tpr_at = float(tpr[idx[-1]])
                        actual_fpr = float(fpr[idx[-1]])
                        threshold = float(thresholds[idx[-1]])
                        prec = (self.prior * tpr_at) / (self.prior * tpr_at + (1 - self.prior) * actual_fpr + 1e-30)
                    else:
                        tpr_at = 0.0
                        actual_fpr = 0.0
                        threshold = 0.0
                        prec = 0.0

                    all_results[attack_name]['FPRs'].append(tfpr * 100)
                    all_results[attack_name]['TPRs'].append(tpr_at * 100)
                    all_results[attack_name]['Precs'].append(prec * 100)

                    # Store threshold info for all attacks
                    threshold_info.append({
                        'attack': attack_name,
                        'target_model': target_idx,
                        'target_fpr': tfpr,
                        'actual_fpr': actual_fpr,
                        'threshold': threshold,
                        'tpr': tpr_at,
                        'precision': prec
                    })

        # Compute statistics across all target models
        self.logger.info("\n" + "="*80)
        self.logger.info("LEAVE-ONE-OUT RESULTS (Mean ± Std across target models)")
        self.logger.info("="*80)

        summary_results = []
        for attack_name, results in all_results.items():
            auc_mean = np.mean(results['AUCs'])
            auc_std = np.std(results['AUCs'], ddof=1)
            acc_mean = np.mean(results['Accs'])
            acc_std = np.std(results['Accs'], ddof=1)

            row = {
                'Attack': attack_name,
                'AUC Mean': np.round(auc_mean, 2),
                'AUC Std': np.round(auc_std, 2),
                'Acc Mean': np.round(acc_mean, 2),
                'Acc Std': np.round(acc_std, 2)
            }

            log_msg = f"{attack_name}: AUC={auc_mean:.2f}±{auc_std:.2f}, Acc={acc_mean:.2f}±{acc_std:.2f}"

            # Group FPRs, TPRs, and Precs by target FPR value
            unique_fprs = sorted(set(results['FPRs']))
            for tfpr in unique_fprs:
                # Find all indices for this target FPR
                indices = [i for i, fpr in enumerate(results['FPRs']) if fpr == tfpr]
                
                # Extract TPRs and Precs for this target FPR
                tprs_for_fpr = [results['TPRs'][i] for i in indices]
                precs_for_fpr = [results['Precs'][i] for i in indices]
                
                tpr_mean = np.mean(tprs_for_fpr)
                tpr_std = np.std(tprs_for_fpr, ddof=1)
                prec_mean = np.mean(precs_for_fpr)
                prec_std = np.std(precs_for_fpr, ddof=1)

                row[f"TPR@{tfpr:.4f}%FPR Mean"] = np.round(tpr_mean, 3)
                row[f"TPR@{tfpr:.4f}%FPR Std"] = np.round(tpr_std, 3)
                row[f"Prec@{tfpr:.4f}%FPR Mean"] = np.round(prec_mean, 2)
                row[f"Prec@{tfpr:.4f}%FPR Std"] = np.round(prec_std, 2)

                log_msg += f", TPR@{tfpr:.4f}%FPR={tpr_mean:.4f}±{tpr_std:.4f}"
                log_msg += f", Prec@{tfpr:.4f}%FPR={prec_mean:.2f}±{prec_std:.2f}"

            self.logger.info(log_msg)
            summary_results.append(row)

        # Save summary results
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(self.experiment_dir, 'attack_results_leave_one_out_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        self.logger.info(f"Saved leave-one-out summary to {summary_csv_path}")

        # Save hard membership labels
        membership_labels_path = os.path.join(self.experiment_dir, 'membership_labels.npy')
        np.save(membership_labels_path, membership_labels)
        self.logger.info(f"Saved hard labels (shape {membership_labels.shape}) to {membership_labels_path}")

        # Save attack scores with simple, clear names
        score_type_map = {
            'LiRA (online)': 'online_scores',
            'LiRA (online, fixed var)': 'online_fixed_scores',
            'LiRA (offline)': 'offline_scores',
            'LiRA (offline, fixed var)': 'offline_fixed_scores',
            'Global threshold': 'global_scores'
        }
        
        for attack_name, score_array in attack_scores.items():
            score_filename = score_type_map[attack_name]
            score_path = os.path.join(self.experiment_dir, f'{score_filename}_leave_one_out.npy')
            np.save(score_path, score_array)
            self.logger.info(f"Saved {attack_name} scores (shape {score_array.shape}) to {score_path}")

        # Save threshold information as CSV (now includes all attacks)
        threshold_df = pd.DataFrame(threshold_info)
        threshold_csv_path = os.path.join(self.experiment_dir, 'threshold_info_leave_one_out.csv')
        threshold_df.to_csv(threshold_csv_path, index=False)
        self.logger.info(f"Saved threshold information to {threshold_csv_path}")

        self.logger.info("="*80)
        self.logger.info("Leave-one-out evaluation completed!")
        self.logger.info("="*80)

    # --- helper to compute ROC metrics ---
    def sweep(self, score, truth):
        fpr, tpr, thresholds = roc_curve(truth, -score)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        return fpr, tpr, auc(fpr, tpr), acc, thresholds

    # --- attack definitions ---
    def generate_ours(self, keep, scores, check_keep, check_scores, fix_variance=False):
        N = scores.shape[1]
        dat_in  = [scores[keep[:,j], j]   for j in range(N)]
        dat_out = [scores[~keep[:,j], j]  for j in range(N)]
        in_size  = min(map(len, dat_in))
        out_size = min(map(len, dat_out))
        dat_in   = np.array([x[:in_size]  for x in dat_in])
        dat_out  = np.array([x[:out_size] for x in dat_out])

        mu_in  = np.median(dat_in,  axis=1)
        mu_out = np.median(dat_out, axis=1)
        if fix_variance:
            sigma_in  = np.std(dat_in)
            sigma_out = np.std(dat_in)
        else:
            sigma_in  = np.std(dat_in,  axis=1)
            sigma_out = np.std(dat_out, axis=1)

        preds, ans = [], []
        for mask, sc in zip(check_keep, check_scores):
            pin  = -scipy.stats.norm.logpdf(sc, mu_in,  sigma_in + 1e-30)
            pout = -scipy.stats.norm.logpdf(sc, mu_out, sigma_out + 1e-30)
            score   = pin - pout
            preds.extend(score.mean(1))
            ans.extend(mask)
        return preds, ans

    def generate_ours_offline(self, keep, scores, check_keep, check_scores, fix_variance=False):
        N = scores.shape[1]
        dat_out = [scores[~keep[:,j], j] for j in range(N)]
        out_size = min(map(len, dat_out))
        dat_out = np.array([x[:out_size] for x in dat_out])

        mu_out = np.median(dat_out, axis=1)
        if fix_variance:
            sigma = np.std(dat_out)
        else:
            sigma = np.std(dat_out, axis=1)

        preds, ans = [], []
        for mask, sc in zip(check_keep, check_scores):
            score = scipy.stats.norm.logpdf(sc, mu_out, sigma + 1e-30)
            preds.extend(score.mean(1))
            ans.extend(mask)
        return preds, ans

    def generate_global(self, keep, scores, check_keep, check_scores):
        preds, ans = [], []
        for mask, sc in zip(check_keep, check_scores):
            preds.extend((-sc).mean(1))
            ans.extend(mask)
        return preds, ans
    

    def compute_train_test_stats(self, save_csv_path=None):
        """
        For each shadow model i:
        - load logits [N,1,A,C] or [N,A,C]
        - extract ORIGINAL (non-augmented) logits → [N,C]
        - compute per‐sample cross‐entropy loss & accuracy on original inputs
        - split into Train/Test via keep_indices[i]
        Then compute across‐models mean & std of loss & acc, and save CSV:
            Set, Loss Mean, Loss STD, Acc (%) Mean, Acc (%) STD
        """
        train_losses, train_accs = [], []
        test_losses, test_accs   = [], []

        for i in range(self.num_shadow_models):
            lp = os.path.join(self.experiment_dir,
                            f"model_{i}", "logits", "logits.npy")
            if not os.path.exists(lp):
                self.logger.warning(f"Missing logits for model_{i}, skipping")
                continue

            logits = np.load(lp)             # [N,1,A,C] or [N,A,C]
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:,0]         # ⇒ [N,A,C]

            # --- 1) Extract ORIGINAL (first) logits only ---
            # The first augmentation (index 0) should be the original, non-augmented input
            original_logits = logits[:, 0, :]  # [N,C] - only the original inputs

            # --- 2) compute per‐sample cross‐entropy loss on original inputs ---
            # CE = logsumexp - logit_true
            N, C = original_logits.shape
            labels = self.labels[:N]

            # stable log-sum-exp
            mx    = np.max(original_logits, axis=1, keepdims=True)            # [N,1]
            lse   = mx + np.log(np.exp(original_logits - mx).sum(axis=1, keepdims=True))  # [N,1]

            # Cross-entropy loss: logsumexp - logit_true
            losses = lse[:,0] - original_logits[np.arange(N), labels]         # [N]

            # --- 3) accuracy calculation on original inputs ---
            preds = original_logits.argmax(axis=1)
            accs  = (preds == labels).astype(float)                          # [N]

            # --- 4) split by Train/Test using keep_indices[i] ---
            # keep_indices[i] should indicate which samples were in training set for model i
            keep = self.keep_indices[i]
            if keep.ndim > 1: keep = keep.reshape(-1)
            keep = keep[:N]

            # Training set statistics (samples that were used to train model i)
            train_losses.append( losses[keep].mean() )
            train_accs.append(   accs[keep].mean()   )
            
            # Test set statistics (samples that were NOT used to train model i)
            test_losses.append( losses[~keep].mean() )
            test_accs.append(   accs[~keep].mean()   )

        # --- 5) aggregate across models & log ---
        rows = []
        for tag, (ls, ac) in [
            ('Train', (train_losses, train_accs)),
            ('Test',  (test_losses, test_accs))
        ]:
            if len(ls) == 0:  # Handle empty lists
                continue
                
            loss_mean = float(np.mean(ls))
            loss_std  = float(np.std(ls, ddof=1)) if len(ls) > 1 else 0.0
            acc_mean  = float(np.mean(ac) * 100)
            acc_std   = float(np.std(ac, ddof=1) * 100) if len(ac) > 1 else 0.0

            rows.append({
                'Set':             tag,
                'Loss Mean':       np.round(loss_mean, 4),
                'Loss STD':        np.round(loss_std, 4),
                'Acc (%) Mean':    np.round(acc_mean, 2),
                'Acc (%) STD':     np.round(acc_std, 2),
            })

            self.logger.info(
                f"{tag}: Loss={loss_mean:.4f}±{loss_std:.4f}, "
                f"Acc={acc_mean:.2f}%±{acc_std:.2f}%"
            )

        # --- 6) save CSV ---
        if rows:  # Only save if we have data
            df = pd.DataFrame(rows)
            if save_csv_path:
                df.to_csv(save_csv_path, index=False)
                self.logger.info(f"Saved train/test stats to {save_csv_path}")
            return df
        else:
            self.logger.warning("No data to save")
            return pd.DataFrame()