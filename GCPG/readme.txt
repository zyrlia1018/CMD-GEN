训练
python train_chembl_baseline.py <output_dir> --show_progressbar

python generate.py data/phar_PARP1.posp gen_result/ result/rs_mapping/fold0_epoch64.pth result/rs_mapping/tokenizer_r_iso.pkl --filter --device cpu

python test_generation.py results/rs_mapping/fold0_epoch32.pth results/rs_mapping/tokenizer_r_iso.pkl --device cuda --show_progressbar

python generate_dock.py data/phar_PARP1_5.posp gen_dock_result/ results_docking_epoch1/rs_mapping_parp1/fold0_epoch32.pth results_docking_epoch1/rs_mapping_parp1/tokenizer_r_iso.pkl --filter --device cpu
