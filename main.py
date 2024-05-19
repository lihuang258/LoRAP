import os
import torch
cuda_config = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_config
from utils.model_load import get_model
import time
import gc
from utils.log import exported_logger as logger
from utils.compression import lorap
from utils.data import get_loaders,test_loaders
from utils.eval import llama_eval
model,tokenizer,args=get_model()
device = torch.device("cuda:0")
model.to(device)
########compress model########
lorap(args, model, tokenizer, device)
########save pruned model########
if args.save_model!=None:
    save_path = args.save_model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, f"model-{args.sparsity_ratio}.bin")
    torch.save({'model': model, 'tokenizer': tokenizer, }, save_name)
    print("finish save model to %s" % save_name)

model.eval()
model.seqlen=args.eval_seqlen
print("evaluate sequence length",model.seqlen)
_, test_loader=test_loaders("ptb", tokenizer, seq_len=model.seqlen, batch_size=8)
_, test_loader2=test_loaders("wikitext2", tokenizer, seq_len=model.seqlen, batch_size=8)
ptb_ppl=llama_eval(model, test_loader, model.device)
logger.info("ptb ppl:%f" % ptb_ppl)
wiki_ppl=llama_eval(model, test_loader2, model.device)
logger.info("wikitext2 ppl:%f"%wiki_ppl)
