from datasets import load_dataset
import os
from liger_kernel.transformers import apply_liger_kernel_to_llama, LigerTiledSwiGLUMLP
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import LlamaForCausalLM
from tqdm import trange


MODEL_NAME = "unsloth/Llama-3.1-8B"
DEVICE = torch.device("cuda")


class WikitextDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len):
        self.texts = [row["text"] for row in dataset if len(row["text"].strip()) > 0]
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=self.seq_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def change_mlp(model):
    for layer in model.model.layers:
        if not hasattr(layer, "mlp"):
            continue

        new_mlp = LigerTiledSwiGLUMLP(model.config)
        new_mlp.gate_proj = layer.mlp.gate_proj
        new_mlp.up_proj = layer.mlp.up_proj
        new_mlp.down_proj = layer.mlp.down_proj

        layer.mlp = new_mlp


def train(
    n_iters: int = 20,
    batch_size: int = 1,
    seq_len: int = 1024,
    loss_type: str = "torch",
    liger_model: bool = False,
    tiled_mlp: bool = False
):
    torch.manual_seed(0xDEADBEEF)
    torch.cuda.manual_seed(0xDEADBEEF)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE,
    )

    # can be also be spawned by direct call via 
    #
    # model = LlamaForCausalLM.from_pretrained(
    #     MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE, **kwargs
    # )
    
    if liger_model and loss_type in ["ligerfusedlinear", "liger"]:
        is_fused_ce = (loss_type == "ligerfusedlinear")
        apply_liger_kernel_to_llama(cross_entropy=(not is_fused_ce), fused_linear_cross_entropy=is_fused_ce, model=model)
    elif liger_model and loss_type == "torch":
        apply_liger_kernel_to_llama(cross_entropy=False, fused_linear_cross_entropy=False, model=model)
    elif not liger_model and loss_type in ["ligerfusedlinear", "liger"]:
        is_fused_ce = (loss_type == "ligerfusedlinear")
        apply_liger_kernel_to_llama(
            rope=False,
            cross_entropy=(not is_fused_ce),
            fused_linear_cross_entropy=is_fused_ce,
            rms_norm=False,
            swiglu=False,
            model=model
        )
    elif not liger_model and loss_type == "torch":
        pass

    if tiled_mlp:
        change_mlp(model)
        
    model.enable_input_require_grads()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    scaler = torch.amp.grad_scaler.GradScaler(DEVICE)

    vocab_size = model.vocab_size

    if not liger_model and loss_type == "torch":
        loss_fn = torch.nn.CrossEntropyLoss()

    data = load_dataset("wikitext", "wikitext-2-v1")['train']
    dataset = WikitextDataset(data, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history()

    timings = []
    losses = []

    model.train()
    dataloader_iter = iter(dataloader)

    for _ in trange(n_iters):
        batch = next(dataloader_iter)

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        optimizer.zero_grad()

        with torch.amp.autocast(str(DEVICE), dtype=torch.float16):
            output = model(**batch, labels=labels)
            if loss_type != "torch" or liger_model:
                loss = output.loss
            else:
                loss = loss_fn(output.logits[:, :-1].reshape(-1, vocab_size), labels[:,1:].reshape(-1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

        losses.append(loss.item())

    snapshot_name = model.__repr__().split('\n')[0][:-1]
    model_type = "liger" if liger_model else "hf"
    name = f"{snapshot_name}__bs_{batch_size}__ce_{loss_type}__model_{model_type}"
    
    if tiled_mlp:
        name += "__tiled"

    torch.cuda.memory._dump_snapshot(f"logs/{name}.pickle")
    torch.save(
        {
            "max_memory_allocated": torch.cuda.max_memory_allocated(),
            "losses": losses,
            "timings": timings,
        },
        f"logs/{name}.pt"
    )

    
if __name__ == "__main__":
    N_ITERS = int(os.getenv("N_ITERS", 100))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
    LOSS_TYPE = os.getenv("LOSS_TYPE", "torch")
    LIGER_MODEL = (os.getenv("LIGER_MODEL", "NO").upper() == "YES")
    TILED_MLP = (os.getenv("TILED_MLP", "NO").upper() == "YES")

    train(
        n_iters=N_ITERS,
        batch_size=BATCH_SIZE,
        loss_type=LOSS_TYPE,
        liger_model=LIGER_MODEL,
        tiled_mlp=TILED_MLP,
    )  
