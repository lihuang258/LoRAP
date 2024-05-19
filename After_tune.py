
######Due to the limited support of LoRA fine-tuning,
# currently only torch.nn.Linear and Conv1D are supported.
# Therefore, we opt to merge and perform SVD decomposition again after fine-tuning.
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from utils.compression import after_tune_SVD
from utils.data import get_loaders,test_loaders
from utils.eval import llama_eval
from LLMPruner.peft import PeftModel, PeftConfig
from utils.convert import convert_safe_to_bin
import argparse
def test_time(model,device):
    model.eval()
    with torch.no_grad():
        inputs = torch.randint(0, 32000, (1, 128)).to(device)
        total_time = 0

        for i in range(100):
            begin = time.time()
            lm_logits = model(inputs).logits
            end = time.time()
            total_time += end - begin

        average_time = total_time / 100
        print("Average time for 100 runs: %f seconds" % average_time)
def test_result(model,test_length,device):
    # print("evaluate sequence length", model.seqlen)
    model.seqlen= test_length
    _, test_loader = test_loaders("ptb", tokenizer, seq_len=model.seqlen, batch_size=8)
    _, test_loader2 = test_loaders("wikitext2", tokenizer, seq_len=model.seqlen, batch_size=8)
    ppl = llama_eval(model, test_loader, device)
    print("ptb ppl:%f" % ppl)
    ppl = llama_eval(model, test_loader2, device)
    print("wikitext2 ppl:%f" % ppl)
def save_merged_model(model,save_path,sparsity_ratio,allocate_ratio):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path)
    model=after_tune_SVD(model,8,sparsity_ratio,allocate_ratio)
    torch.save({'model': model, 'tokenizer': tokenizer, }, save_name)
    print("save model to %s" % save_name)
def get_model(model_path,use_lora,lora_path):
    pruned_dict = torch.load(model_path)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    print("base model load finished")
    if use_lora:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("merge finished")
    return model,tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="pruned_model_path", help='The path of model')
    parser.add_argument('--use_lora', type=bool, default=False, help='Whether to use LoRA')
    parser.add_argument('--lora_path', type=str, default="tuned_model_path", help='The path of LoRA model')
    parser.add_argument('--para_allocate', type=float, default=3, help='The parameter ratio of (Wv+Wo):(Wq+Wk)')
    parser.add_argument('--sparsity_ratio', type=float, default=0.2, help='Sparsity level')
    parser.add_argument('--eval_seqlen', type=int, default=128, help='The length of evaluation data')
    parser.add_argument('--save_path', type=int, default="tuned_model", help='The size of model')
    args = parser.parse_args()
    use_lora=args.use_lora
    sparsity_ratio = args.sparsity_ratio
    allocate_ratio = args.para_allocate
    eval_seqlen = args.eval_seqlen
    device = torch.device("cuda:0")
    # convert_safe_to_bin(sparsity_ratio,model_size)
    model_path=args.model_path
    lora_path=args.lora_path
    model, tokenizer= get_model(model_path,use_lora,lora_path)
    print("lora weights merged")
    test_result(model,eval_seqlen,device)
    saved_path=args.save_path
    save_merged_model(model, tokenizer, saved_path,sparsity_ratio,allocate_ratio)
