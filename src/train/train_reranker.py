import os
import sys

import torch
import random
from accelerate.utils import set_seed
# from model import Embedding
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
# from data import EmbeddingDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from ..model import SFR, SFRReranker
from .trainer import Trainer
from ..arguments import parse_args
from ..utils.general_utils import set_seed


def create_adamw_optimizer(
        model,
        lr,
        weight_decay=1e-2,
        no_decay_keywords=('bias', 'LayerNorm', 'layernorm'),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if n.startswith('mlp.')],
            'weight_decay': weight_decay,
            'lr': 0.001,
        },
        
        {
            'params': [p for n, p in parameters if (not n.startswith('mlp.')) and not any(nd in n for nd in no_decay_keywords)],
            'weight_decay': weight_decay,
            'lr': lr,
        },
        {
            'params': [p for n, p in parameters if (not n.startswith('mlp.')) and any(nd in n for nd in no_decay_keywords)],
            'weight_decay': 0.0,
            'lr': lr,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

class RerankDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, query_max_len, passage_max_len):
        self._data = data
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        if self.tokenizer.pad_token is None:
            print('Setting pad token to eos token')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        

    def __getitem__(self, idx):
        example = self._data[idx]
        query = self.tokenizer(
            example['query'], 
            padding='max_length',
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt"
        )
        docs = self.tokenizer(example['passages']['passage_text'],
            padding='max_length',  # Pads all sequences to the same length
            truncation=True,  # Truncate sequences to the model's max length
            max_length=self.passage_max_len,
            return_tensors="pt"  # Return PyTorch tensors directly
        )
        query['input_ids'] = query['input_ids'][0]
        query['attention_mask'] = query['attention_mask'][0]
        labels = torch.tensor(example['passages']['is_selected'], dtype=torch.long)
        
        max_size = 10
        if docs['input_ids'].shape[0] < 10:
            repeat_count = (max_size + docs['input_ids'].shape[0] - 1) // docs['input_ids'].shape[0]
            docs['input_ids'] = torch.tile(docs['input_ids'], (repeat_count, 1))
            docs['attention_mask'] = torch.tile(docs['attention_mask'], (repeat_count, 1))
            labels = torch.tile(labels, (repeat_count, ))
            
            

        return {
            'query_input_ids': query['input_ids'], # (Length of Query, )
            'query_attention_mask': query['attention_mask'],
            'doc_input_ids': docs['input_ids'][:max_size],  # (Length of Doc (padded), )
            'doc_attention_mask': docs['attention_mask'][:max_size],
            'labels': labels[:max_size],
        }
        
    def __len__(self):
        return len(self._data)



# def parse_args():
#     import yaml
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str)
#     parser.add_argument("--model_name_or_path", default='hfl/chinese-roberta-wwm-ext')

#     parser.add_argument("--train_dataset", help='trainset')
#     parser.add_argument('--neg_nums', type=int, default=15)
#     parser.add_argument('--query_max_len', type=int, default=128)
#     parser.add_argument('--passage_max_len', type=int, default=512)

#     parser.add_argument('--output_dir', help='output dir')
#     parser.add_argument('--save_on_epoch_end', type=int, default=1, help='if save_on_epoch_end')
#     parser.add_argument('--num_max_checkpoints', type=int, default=5)


#     parser.add_argument('--epochs', type=int, default=2, help='epoch nums')
#     parser.add_argument("--lr", type=float, default=5e-5)
#     parser.add_argument('--batch_size', type=int, help='batch size')
#     parser.add_argument('--seed', type=int, default=666)
#     parser.add_argument("--warmup_proportion", type=float, default=0.05)
#     parser.add_argument("--temperature", type=float, default=0.02)
#     parser.add_argument('--mixed_precision', default='fp16', help='')
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

#     parser.add_argument("--log_with", type=str, default='wandb', help='wandb,tensorboard')
#     parser.add_argument("--log_interval", type=int, default=10)

#     parser.add_argument('--use_mrl', action='store_true', help='if use mrl loss')
#     parser.add_argument('--mrl_dims', type=str, help='list of mrl dims', default='128, 256, 512, 768, 1024, 1280, 1536, 1792')

#     args = parser.parse_args()

#     with open(args.config, "r", encoding="utf-8") as file:
#         config = yaml.safe_load(file)

#     for key, value in config.items():
#         setattr(args, key, value)

#     return args


def main():
    DISTRIBUTED_TRAINING = False

    if DISTRIBUTED_TRAINING:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler


    args = parse_args('train')

    set_seed(args.seed)
    project_dir = os.path.abspath(os.path.join(args.output_dir, ".."))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=project_dir,
        device_placement=True,
        log_with="wandb"
    )

    # accelerator.init_trackers('embedding', config=vars(args))
    accelerator.init_trackers(
        project_name="extremerag",
        config=vars(args),
        init_kwargs={
            "wandb": {
                "name": "train reranker", 
                "tags": ["experiment_1", "debug"]
                }
            }
    )
    
    
    
    accelerator.print(f"Train Args from User Input: {vars(args)}")
    
    sfr_kwargs = {
        "torch_dtype": torch.bfloat16
    }
    if "checkpoint" in args.model_name_or_path:
        sfr_kwargs["local_files_only"] = True
    
    sfr = SFR.from_pretrained(args.model_name_or_path, **sfr_kwargs)
    
    tokenizer_kwargs = {}
    if "checkpoint" in args.model_name_or_path:
        tokenizer_kwargs["local_files_only"] = True
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)

    model = SFRReranker(sfr, tokenizer, batch_size=args.per_device_train_batch_size, device=args.device)
    
    # To load, use:
    # model.mlp.load_state_dict(torch.load(os.path.join(os.environ.get('SARA_CHECKPOINTS_DIR','checkpoints'),'reranker','mlp.pth'), map_location=args.device))
    for n, p in (model.named_parameters()):
        if n.startswith("embed_model"):
            p.requires_grad = False
            
        else:
            p.requires_grad = True
            print(n)
    
    train_dataset = datasets.load_dataset("microsoft/ms_marco", "v2.1", split='train')
    train_dataset = RerankDataset(list(train_dataset), tokenizer, query_max_len=32, passage_max_len=256)
    train_sampler = DistributedSampler(train_dataset) if DISTRIBUTED_TRAINING else None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        # collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    dev_dataset = datasets.load_dataset("microsoft/ms_marco", "v2.1", split='validation')
    dev_dataset = RerankDataset(list(dev_dataset)[:2000], tokenizer, query_max_len=32, passage_max_len=256)
    dev_sampler = DistributedSampler(dev_dataset) if DISTRIBUTED_TRAINING else None
    
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.per_device_eval_batch_size,
        # collate_fn=train_dataset.collate_fn,
        sampler=dev_sampler,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
    )
    

    accelerator.print(f'train_dataloader total is : {len(train_dataloader)}')

    optimizer = create_adamw_optimizer(
        model, lr=float(args.learning_rate)
    )
    assert 0 <= args.warmup_proportion < 1
    total_steps = (
        len(train_dataloader) * args.num_train_epochs
    ) // accelerator.gradient_state.num_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.num_train_epochs * total_steps),
        num_training_steps=total_steps,
    )


    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)

    accelerator.wait_for_everyone()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=dev_dataloader,
        accelerator=accelerator,
        epochs=args.num_train_epochs,
        lr_scheduler=lr_scheduler,
        log_interval=args.log_interval * accelerator.gradient_state.num_steps,
        save_on_epoch_end=args.save_on_epoch_end,
        tokenizer=tokenizer,
    )

    accelerator.print(f'Start training for {args.num_train_epochs} epochs')
    trainer.train()

    accelerator.print('Training finished. Saving model ...')

    unwrapped_model = accelerator.unwrap_model(model)
    save_dir = os.path.join(args.output_dir, "..", "checkpoints", 'reranker')
    os.makedirs(save_dir, exist_ok=True)
    # Save the state dictionary of the `mlp` module
    torch.save(unwrapped_model.mlp.state_dict(), os.path.join(save_dir, 'mlp.pth'))


if __name__ == "__main__":
    main()
