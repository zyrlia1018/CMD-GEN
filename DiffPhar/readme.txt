训练

python -u train.py --config <config>.yml
python generate_phars.py ./checkpoints/best-model-epoch\=epoch\=281.ckpt --num_nodes_phar 10 --pdbfile ./generated/7ONS.pdb --ref_ligand A:1101
python test.py checkpoints/ca_atom/best-model-epoch\=epoch\=472.ckpt --test_dir ./data/processed_crossdock_noH_ca_only_temp/test/ --sanitize

