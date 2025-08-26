# ["rel_train_process", "rel_tiny_random_process", "rel_tiny_uniform_process", "rel_tiny_longtail_process", "rel_tiny_knowledge_process"]:
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/caption/add_captionrl_v3p1_2w8_uniform_biased.csv" "rel_train_process"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/caption/add_captionrl_v3_4w9_random_unbiased_20250806.csv" "rel_train_process"

python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/caption/add_caption0226_tiny_random_sample_wtax_unbiased.csv" "rel_tiny_random_process"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/caption/add_caption0226_tiny_uniform_sample_wtax.csv" "rel_tiny_uniform_process"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/caption/add_caption0320_tiny_longtail_sample_wtax.csv" "rel_tiny_longtail_process"
python /mnt/ali-sh-1/usr/huaan1/ocean/code/verl/examples/data_preprocess/rel_train.py "/mnt/ali-sh-1/usr/huaan1/ocean/data/caption/add_caption_tiny_knowledge_good_click_note_test.csv" "rel_tiny_knowledge_process"