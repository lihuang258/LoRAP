from utils.log import exported_logger as logger
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer,LlamaForCausalLM,LlamaConfig
from importlib.metadata import version
def get_model():
    logger.info("Start print log")
    logger.info('torch:%s', version('torch'))
    logger.info('transformers:%s', version('transformers'))
    logger.info('accelerate:%s', version('accelerate'))
    ######Record the module version#######
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/Llama-1/7B/", help="Model to be pruned.")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--dataset", type=str, default="c4", choices=["c4", "PTB", "wikitest2","bookcorpus"],help="Dataset for calibration.")
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--calibration_seqlen', type=int, default=128, help='The length of calibration data used during the compression process')
    parser.add_argument('--eval_seqlen', type=int, default=128, help='The length of evaluation data')
    parser.add_argument('--sparsity_ratio', type=float, default=0.2, help='Sparsity level')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--eval_results', type=str, default="result/eval", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default="pruned_model", help='Path to save the pruned model.')
    parser.add_argument('--para_allocate', type=float, default=3, help='The parameter ratio of (Wv+Wo):(Wq+Wk)')
    parser.add_argument('--mlp_compress_method', type=str, default="prune", choices=["prune","decom"],help='The method to compress mlp prune the channel or decompose the weight matrix')
    parser.add_argument('--mlp_least_save_ratio', type=float, default=0.01, help='The retained ratio of least important in mlp')
    parser.add_argument('--real_com', type=bool, default=True, help='The real compression or not')
    parser.add_argument("--deco_method", type=str, default="AWSVD", choices=["AWSVD", "AFM","SVD"])
    parser.add_argument("--sublayer", type=str, default=["self_attn", "mlp"], choices=["self_attn", "mlp"],
                        help="Sublayer to be compressed ,you can choose self_attn or mlp individually or both")
    args = parser.parse_args()
    logger.info(args)
    if "Llama" or "vicuna" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(args.model)
        logger.info("tokenizer load finished")
        model = LlamaForCausalLM.from_pretrained(args.model, device_map="balanced", torch_dtype=torch.float16)
        logger.info("model load finished")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info("tokenizer load finished")
        model = AutoModelForCausalLM.from_pretrained(args.model,device_map="balanced", torch_dtype=torch.float16)
        logger.info("model load finished")
    model.seqlen = args.calibration_seqlen
    return model,tokenizer,args
