
# ============================================================
# TRAINER
# ============================================================

class MultiViewTrainer(DefaultTrainer):
    """Trainer for multi-view Mask2Former."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Register Gradient Clipping Hook to the inner trainer (AMPTrainer/SimpleTrainer)
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            clip_value = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            try:
                clip_type = cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE
            except AttributeError:
                clip_type = "norm"
            try:
                norm_type = cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE
            except AttributeError:
                norm_type = 2.0
            
            # Create the hook
            clipping_hook = GradientClippingHook(clip_value, clip_type, norm_type)
            
            # Register it
            self._trainer.register_hooks([clipping_hook])
            logger.info(f"Registered GradientClippingHook (value={clip_value}, type={clip_type})")

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(NaNLossCheckHook())
        # Add gradient diagnostics hook (every 100 iters)
        hooks.append(GradientDiagnosticsHook(log_period=100))
        # Add CSV logging hook
        hooks.append(CSVMetricsLogger(
            output_dir=self.cfg.OUTPUT_DIR,
            log_period=20,
        ))
        return hooks
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator for panoptic segmentation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOPanopticEvaluator(dataset_name, output_folder)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Run evaluation and log results to CSV.
        
        Overrides DefaultTrainer.test to capture evaluation results.
        """
        results = super().test(cfg, model, evaluators)
        
        # Log evaluation results to CSV
        if comm.is_main_process() and results:
            eval_csv_path = os.path.join(cfg.OUTPUT_DIR, "evaluation_metrics.csv")
            
            # Check if file exists, if not create with headers
            if not os.path.exists(eval_csv_path):
                with open(eval_csv_path, 'w') as f:
                    headers = ['iteration', 'PQ', 'SQ', 'RQ', 'PQ_th', 'SQ_th', 'RQ_th', 'PQ_st', 'SQ_st', 'RQ_st']
                    f.write(','.join(headers) + '\n')
            
            # Extract panoptic metrics
            for dataset_name, dataset_results in results.items():
                if 'panoptic_seg' in dataset_results:
                    metrics = dataset_results['panoptic_seg']
                    
                    pq = metrics.get('PQ', 0.0)
                    sq = metrics.get('SQ', 0.0)
                    rq = metrics.get('RQ', 0.0)
                    pq_th = metrics.get('PQ_th', 0.0)
                    sq_th = metrics.get('SQ_th', 0.0)
                    rq_th = metrics.get('RQ_th', 0.0)
                    pq_st = metrics.get('PQ_st', 0.0)
                    sq_st = metrics.get('SQ_st', 0.0)
                    rq_st = metrics.get('RQ_st', 0.0)
                    
                    # Try to get current iteration (may not be available during eval_only)
                    try:
                        iteration = cfg.SOLVER.MAX_ITER  # Use as fallback
                    except Exception:
                        iteration = 0
                    
                    with open(eval_csv_path, 'a') as f:
                        values = [
                            str(iteration),
                            f'{pq:.4f}', f'{sq:.4f}', f'{rq:.4f}',
                            f'{pq_th:.4f}', f'{sq_th:.4f}', f'{rq_th:.4f}',
                            f'{pq_st:.4f}', f'{sq_st:.4f}', f'{rq_st:.4f}',
                        ]
                        f.write(','.join(values) + '\n')
                    
                    print(f"\n{'='*60}")
                    print(f"Panoptic Segmentation Evaluation Results:")
                    print(f"  PQ={pq:.2f}  SQ={sq:.2f}  RQ={rq:.2f}")
                    print(f"  PQ_th (things)={pq_th:.2f}  PQ_st (stuff)={pq_st:.2f}")
                    print(f"{'='*60}\n")
        
        return results
    
    @classmethod
    def run_view_analysis(cls, cfg, model):
        """
        Run detailed per-view analysis on evaluation dataset.
        
        This analyzes:
        - Per-view losses (reference vs targets)
        - Class probability consistency across views
        - Query prediction similarity
        
        Results are logged to view_analysis_metrics.csv
        """
        logger = logging.getLogger(__name__)
        logger.info("Running detailed view analysis...")
        
        # Build data loader
        data_loader = cls.build_test_loader(cfg, cfg.DATASETS.TEST[0])
        
        # Create view analysis CSV
        output_dir = cfg.OUTPUT_DIR
        view_csv_path = os.path.join(output_dir, "view_analysis_metrics.csv")
        if not os.path.exists(view_csv_path):
            with open(view_csv_path, 'w') as f:
                headers = [
                    'iteration', 'scene_id', 'view_type', 'view_idx',
                    'loss_ce', 'loss_mask', 'loss_dice', 'total_loss',
                    'class_prob_kl', 'top_class_agreement'
                ]
                f.write(','.join(headers) + '\n')
        
        model.eval()
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(data_loader, desc="View Analysis")):
                if idx >= cfg.TEST.get('VIEW_ANALYSIS_MAX_SAMPLES', 100):
                    break
                
                # Get scene ID from batch
                scene_id = batch[0].get('scene_id', f'scene_{idx}')
                
                # Compute view metrics
                try:
                    view_metrics = model.compute_view_metrics(
                        batch[0],
                        compute_class_probs=True
                    )
                    
                    # Log to CSV
                    iteration = 0  # Can be updated if called during training
                    with open(view_csv_path, 'a') as f:
                        # Log reference view
                        ref_losses = view_metrics['ref_view_losses']
                        values = [
                            str(iteration),
                            str(scene_id),
                            'ref',
                            '0',
                            f"{ref_losses['loss_ce']:.6f}",
                            f"{ref_losses['loss_mask']:.6f}",
                            f"{ref_losses['loss_dice']:.6f}",
                            f"{sum(ref_losses.values()):.6f}",
                            '0.0',
                            '1.0',
                        ]
                        f.write(','.join(values) + '\n')
                        
                        # Log target views
                        target_losses = view_metrics['target_view_losses']
                        kl_divs = view_metrics.get('class_prob_kl_divergence', {}).get('per_view', [])
                        agreements = view_metrics.get('top_class_agreement', {}).get('per_view', [])
                        
                        for i, tgt_loss in enumerate(target_losses):
                            values = [
                                str(iteration),
                                str(scene_id),
                                'target',
                                str(tgt_loss['view_idx']),
                                f"{tgt_loss['loss_ce']:.6f}",
                                f"{tgt_loss['loss_mask']:.6f}",
                                f"{tgt_loss['loss_dice']:.6f}",
                                f"{tgt_loss['total']:.6f}",
                                f"{kl_divs[i] if i < len(kl_divs) else 0.0:.6f}",
                                f"{agreements[i] if i < len(agreements) else 0.0:.6f}",
                            ]
                            f.write(','.join(values) + '\n')
                    
                    # Print summary
                    if idx % 10 == 0:
                        kl_mean = view_metrics.get('class_prob_kl_divergence', {}).get('mean', 0.0)
                        agree_mean = view_metrics.get('top_class_agreement', {}).get('mean', 0.0)
                        logger.info(
                            f"Scene {scene_id}: "
                            f"KL div={kl_mean:.4f}, "
                            f"Class agreement={agree_mean:.4f}"
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze scene {scene_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        logger.info(f"View analysis complete. Results saved to {view_csv_path}")
        
        return view_csv_path
    
    def after_train(self):
        """Close TensorBoard writer after training completes."""
        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
                logger.info("TensorBoard writer closed successfully")
            except Exception as e:
                print(f"Warning: Failed to close TensorBoard writer: {e}")
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build a dataloader for multi-view training.
        """
        # We need to construct the dataset mapper manually to pass custom args
        from mask2former.data.dataset_mappers.scannetpp_multiview_dataset_mapper import (
            ScanNetPPMultiViewDatasetMapper
        )
        
        # Get dataset name
        dataset_name = cfg.DATASETS.TRAIN[0]
        
        # Build mapper
        mapper = ScanNetPPMultiViewDatasetMapper(cfg, is_train=True)
        
        # Build dataloader using Detectron2's builder
        return build_detection_train_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Build a dataloader for multi-view testing/evaluation.
        """
        from mask2former.data.dataset_mappers.scannetpp_multiview_dataset_mapper import (
            ScanNetPPMultiViewDatasetMapper
        )
        
        mapper = ScanNetPPMultiViewDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def main(args):
    """Main training function."""
    cfg = setup_cfg(args)
    
    if args.eval_only:
        model = MultiViewTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        
        # Run view analysis if requested
        if hasattr(args, 'view_analysis') and args.view_analysis:
            res = MultiViewTrainer.run_view_analysis(cfg, model)
        else:
            res = MultiViewTrainer.test(cfg, model)
        return res
    
    trainer = MultiViewTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    
    # View analysis mode
    parser.add_argument(
        "--view-analysis",
        action="store_true",
        help="Run detailed per-view analysis during evaluation"
    )
    
    # ========================================
    # PANOPTIC LABEL GENERATION
    # ========================================
    parser.add_argument(
        "--generate-labels",
        action="store_true",
        help="Generate 2D panoptic labels from ScanNet++ 3D mesh before training"
    )
    parser.add_argument(
        "--generate-labels-only",
        action="store_true",
        help="Only generate labels, don't start training"
    )
    parser.add_argument(
        "--label-gen-workers",
        type=int,
        default=4,
        help="Number of parallel workers for label generation"
    )
    
    # ========================================
    # DATASET PATHS
    # ========================================
    parser.add_argument(
        "--scannetpp-root",
        type=str,
        default=None,
        help="Path to ScanNet++ data directory (contains scene folders)"
    )
    parser.add_argument(
        "--panoptic-root",
        type=str,
        default=None,
        help="Path to rasterized panoptic annotations directory"
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Path to directory containing split files (nvs_sem_train.txt, etc.)"
    )
    
    # ========================================
    # DATASET OPTIONS
    # ========================================
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of semantic classes in the dataset (ScanNet++ has 1000 classes)"
    )
    parser.add_argument(
        "--image-type",
        type=str,
        default="dslr",
        choices=["dslr", "iphone"],
        help="Image type to use (dslr or iphone)"
    )
    parser.add_argument(
        "--use-undistorted",
        action="store_true",
        default=True,
        help="Use undistorted images (default: True)"
    )
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
