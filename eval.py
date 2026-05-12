import os
import torch
import random
import numpy as np
from tqdm import tqdm  
import json
from argparse import ArgumentParser
from LLMs import init_llm
import gc
import sys
sys.setrecursionlimit(10000)

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_eval_datasets(datasets_str):
    return datasets_str.split(',')

def main():
    parser = ArgumentParser()
    parser.add_argument('--eval_datasets', type=parse_eval_datasets, default=['MMMU'],
                    help='name of eval dataset')
    parser.add_argument('--datasets_path', type=str, default="benchmarks",
                    help='path of eval dataset')
    parser.add_argument('--output_path', type=str, default='eval_results/Qwen2-VL-7B-Instruct',
                        help='name of saved json')
    parser.add_argument('--model_name', type=str, default='Qwen2-VL-7B-Instruct',
                        help='name of model')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_visible_devices', type=str, default=None)
    parser.add_argument('--tensor_parallel_size', type=str, default="1")
    parser.add_argument('--use_vllm', type=str, default="True")
    parser.add_argument('--reasoning', type=str, default="False")

    parser.add_argument('--num_chunks', type=str, default="1")
    parser.add_argument('--chunk_idx', type=str, default="0")

    parser.add_argument('--max_image_num', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--repetition_penalty', type=float, default=1)

    parser.add_argument('--test_times', type=int, default=1)
    parser.add_argument('--enable_pca_viz', type=str, default="False")
    parser.add_argument('--enable_cknna_viz', type=str, default="False")
    parser.add_argument('--pca_max_samples', type=int, default=8)
    parser.add_argument('--pca_output_subdir', type=str, default="pca_viz")
    parser.add_argument('--pca_layer_ids', type=str, default="")
    parser.add_argument('--pca_visual_source', type=str, default="global")
    parser.add_argument('--cknna_topk', type=int, default=10)

    parser.add_argument(
        '--dq_cross_attn_visualize',
        type=str,
        default="False",
        help='True: save cross-attn debug PNGs during eval (rank 0). Uses DQLLaVAhpc hooks.',
    )
    parser.add_argument(
        '--dq_cross_attn_visualize_dir',
        type=str,
        default="",
        help='Cross-attn PNG root; empty = {output_path}/cross_attn_viz. Otherwise expand as given.',
    )
    parser.add_argument(
        '--dq_cross_attn_visualize_layers',
        type=str,
        default="",
        help='Comma-separated decoder layer ids, e.g. 0,5,10,15. Empty = all cross-attn layers.',
    )
    parser.add_argument(
        '--dq_cross_attn_visualize_prefill_only',
        type=str,
        default="True",
        help='If True, only plot on prefill step (recommended).',
    )
    parser.add_argument(
        '--dq_self_attn_visualize',
        type=str,
        default="False",
        help='True: save decoder self-attn PNGs during eval (rank 0), last query token to vision keys.',
    )
    parser.add_argument(
        '--dq_self_attn_visualize_dir',
        type=str,
        default="",
        help='Self-attn PNG root; empty = {output_path}/self_attn_viz.',
    )
    parser.add_argument(
        '--dq_self_attn_visualize_layers',
        type=str,
        default="",
        help=(
            'Comma-separated decoder layer ids for self-attn PNGs. Empty=all layers. '
            'For joint DQ eval with dq_cross_attn_visualize: filters decoder self-attn only; '
            'cross-attn layers still follow dq_cross_attn_visualize_layers.'
        ),
    )
    parser.add_argument(
        '--dq_self_attn_visualize_agg',
        type=str,
        default="last",
        help='last | mean — mean: average attention over non-vision query positions (prefill).',
    )
    parser.add_argument(
        '--dq_self_attn_value_stats',
        type=str,
        default="False",
        help='True: log self-attn V (v_proj) per-token sum/mean; aggregate vision vs text; JSONL under output.',
    )
    parser.add_argument(
        '--dq_self_attn_value_stats_dir',
        type=str,
        default="",
        help='Directory for self_attn_value_stats.jsonl; empty = {output_path}/self_attn_value_stats.',
    )
    parser.add_argument(
        '--dq_self_attn_value_stats_layers',
        type=str,
        default="",
        help='Comma-separated decoder layer ids for V stats; empty = all layers.',
    )
    parser.add_argument(
        '--dq_self_attn_value_stats_prefill_only',
        type=str,
        default="True",
        help='If True, only log V stats on prefill (past_key_value is None per layer).',
    )
    parser.add_argument(
        '--dq_self_attn_value_stats_save_npz',
        type=str,
        default="False",
        help='True: save per-token token_sum/token_mean + mask as npz under stats_dir/npz/ for plotting.',
    )
    parser.add_argument(
        '--dq_vis_text_sim_viz',
        type=str,
        default="False",
        help='True: per V_pool layer, heatmap [N_vis_patch × N_text] of vis–text similarity (PNG per layer + summary JSON).',
    )
    parser.add_argument(
        '--dq_vis_text_sim_viz_dir',
        type=str,
        default="",
        help='Output dir for vis_text_sim_*.{png,json}; empty = {output_path}/vis_text_sim_viz.',
    )
    parser.add_argument(
        '--dq_vis_text_sim_viz_prefill_only',
        type=str,
        default="True",
        help='If True, only compute on prefill (full sequence forward).',
    )
    parser.add_argument(
        '--dq_vis_text_sim_use_cosine',
        type=str,
        default="True",
        help='True: cosine between each vision patch row and each text row; False: raw dot product.',
    )
    parser.add_argument(
        '--dq_vis_text_sim_max_layers',
        type=str,
        default="",
        help='Optional cap on first K V_pool layers to plot (e.g. 12); empty = all layers in V_pool.',
    )
    parser.add_argument(
        '--dq_cross_attn_visualize_agg',
        type=str,
        default="last",
        help='last | mean — mean: average cross-attn over positions True in dq scope mask (prefill).',
    )
    parser.add_argument(
        '--max_eval_samples',
        type=int,
        default=0,
        help='If >0, cap dataset size (IU_XRAY etc. via MAX_EVAL_SAMPLES env). 0 = no limit.',
    )

    parser.add_argument('--use_llm_judge', type=str, default="False")
    parser.add_argument('--judge_model_type', type=str, default="openai")
    parser.add_argument('--judge_model', type=str,default = "None")
    parser.add_argument('--api_key', type=str,default = "None")
    parser.add_argument('--base_url', type=str, default= "None")

    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"

    # llm judge setting
    if args.api_key == "None" and args.use_llm_judge == "True":
        raise ValueError("If you want to use llm judge, please set the openai api key")
    
    os.environ["use_llm_judge"] = args.use_llm_judge
    os.environ["judge_model_type"] = args.judge_model_type
    os.environ["judge_model"] = args.judge_model
    os.environ["api_key"] = args.api_key
    os.environ["base_url"] = args.base_url

    # eval modle setting
    os.environ["REASONING"] = args.reasoning
    os.environ["use_vllm"] = args.use_vllm

    os.environ["max_image_num"] = str(args.max_image_num)
    os.environ["ENABLE_PCA_VIZ"] = args.enable_pca_viz
    os.environ["ENABLE_CKNNA_VIZ"] = args.enable_cknna_viz
    os.environ["PCA_MAX_SAMPLES"] = str(args.pca_max_samples)
    os.environ["PCA_OUTPUT_SUBDIR"] = args.pca_output_subdir
    os.environ["PCA_LAYER_IDS"] = args.pca_layer_ids
    os.environ["PCA_VISUAL_SOURCE"] = args.pca_visual_source
    os.environ["CKNNA_TOPK"] = str(args.cknna_topk)

    if int(getattr(args, "max_eval_samples", 0) or 0) > 0:
        os.environ["MAX_EVAL_SAMPLES"] = str(int(args.max_eval_samples))

    # vllm and torch setting
    # Keep allocator config optional: older torch versions do not support
    # "expandable_segments" and will crash during CUDA initialization.
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.use_vllm == "True":
        os.environ["tensor_parallel_size"] = args.tensor_parallel_size
        if int(args.tensor_parallel_size) > 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        torch.multiprocessing.set_start_method('spawn')
        os.environ["num_chunks"] = args.num_chunks
        os.environ["chunk_idx"] = args.chunk_idx


    os.makedirs(args.output_path, exist_ok=True)

    print('initializing LLM...')
    model = init_llm(args)
    
    total_results_path = os.path.join(args.output_path, args.model_name, 'total_results.json')

    from benchmarks import prepare_benchmark

    for eval_dataset in tqdm(args.eval_datasets):
        set_seed(args.seed)
        print(f'evaluating on {eval_dataset}...')

        eval_dataset_path = os.path.join(args.datasets_path,eval_dataset) if args.datasets_path != "hf" else None
        eval_output_path = os.path.join(args.output_path,args.model_name, eval_dataset)
        os.makedirs(eval_output_path, exist_ok=True)
        benchmark = prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path)
        benchmark.load_data()
        final_results = benchmark.eval() if benchmark else {}
        print(f'final results on {eval_dataset}: {final_results}')
        if final_results is not None:
            if os.path.exists(total_results_path):
                with open (total_results_path,"r") as f:
                    total_results = json.load(f)
            else:
                total_results = {}
            total_results[eval_dataset] = final_results

            with open(os.path.join(args.output_path,args.model_name,'total_results.json'),'w') as f:
                json.dump(total_results,f,indent=4)
        gc.collect()

if __name__ == '__main__':
    main()