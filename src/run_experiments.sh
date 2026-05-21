

CUDA_VISIBLE_DEVICES=2 python ./compare_experiment/evaluate_compare.py \
    --output_dir /HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/Docproject/project/compare_experiment/Qwen3-VL-4b-instruct/new_results


CUDA_VISIBLE_DEVICES=3 python ./ablation_text_only/evaluate_ablation.py \
    --output_dir ./ablation_text_only/results \
    --start_idx 513
CUDA_VISIBLE_DEVICES=3 python ./ablation_page_only/evaluate_ablation.py \
    --output_dir ./ablation_page_only/results_k2 \
    --start_idx 513


