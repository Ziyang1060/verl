# ["rel_train_process", "rel_tiny_random_process", "rel_tiny_uniform_process", "rel_tiny_longtail_process", "rel_tiny_knowledge_process"]:
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/rl_v3_4w9_random_unbiased_20250806.csv" "rel_train_process_reverse"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/rl/rl_v3p1_2w8_uniform_unbiased_20250806.csv" "rel_train_process_reverse"

python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/testset/tiny/0226_tiny_random_sample_wtax_unbiased_20250719.csv" "rel_tiny_random_process_reverse"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/testset/tiny/0226_tiny_uniform_sample_wtax_biased.csv" "rel_tiny_uniform_process_reverse"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/testset/tiny/0320_tiny_longtail_sample_wtax_biased.csv" "rel_tiny_longtail_process_reverse"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/testset/tiny/knowledge_good_click_note_test_1k_biased.csv" "rel_tiny_knowledge_process_reverse"