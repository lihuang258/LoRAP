
######Due to the limited support of LoRA fine-tuning,
# currently only torch.nn.Linear and Conv1D are supported.
# Therefore, we opt to merge and perform SVD decomposition again after fine-tuning.
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from utils.compression import after_tune_SVD
from utils.data import get_loaders,test_loaders
from utils.eval import eval_ppl,eval_ppl_wikitext,model_eval,llama_eval
from peft import PeftModel, PeftConfig
from utils.convert import convert_safe_to_bin

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
def save_merged_model(model,save_path,sparsity_ratio):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path)
    model=after_tune_SVD(model,8,sparsity_ratio)
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
    use_lora=False  #or True
    sparsity_ratio = "0.5"
    model_size = "7B"
    device = torch.device("cuda:0")
    # convert_safe_to_bin(sparsity_ratio,model_size)
    model_path=f"pruned_model_path"
    lora_path=f"tuned_model_path"
    model, tokenizer= get_model(model_path,use_lora,lora_path)
    print("merge finished")
    test_result(model,128,device)
    saved_path=f"saved_path"
    save_merged_model(model, tokenizer, saved_path,sparsity_ratio)
