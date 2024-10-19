from tqdm.auto import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from og_transformer import Transformer
from tokenizers import ByteLevelBPETokenizer

def collate_fn(batch, tokenizer):
    """
        endoder input:
        [<s>, <hello>, <world>, </s>]
        The encoder will use the entire sentence 
        with omnidirectional attention, ie., attention that
        can look both forward and back for each word in the sentence. 

        translation
        [<s>, <hallo>, <wereld>, </s>]

        decoder input 
        [  <s>,   <hallo>, <wereld>]
            |        |         |
            v        v         v
        [<hallo>, <wereld>,  </s>  ]
        y_true^

        the decoder attantion is causal, meaning that
        it can only attend to words it has already seen.  
    """
    batch.sort(key=lambda x: len(x['fr']), reverse=True)

    src_txt = [item['en'] for item in batch]
    tgt_txt = [item['fr'] for item in batch]

    src_encodings = tokenizer(
        src_txt,
        max_length=512,
        padding=True,
        truncation=True,
    )

    tgt_encodings = tokenizer(
        tgt_txt,
        max_length=512,
        padding=True,
        truncation=True,
    )

    input_ids = src_encodings['input_ids']
    labels = tgt_encodings['input_ids']

    max_len = max(len(ids) for ids in input_ids + labels)
    
    pad_id = tokenizer.pad_token_id

    padded_input_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids]
    padded_labels = [ids + [pad_id] * (max_len - len(ids)) for ids in labels]

    # Shift decoder inputs to the right by one (prepend <s> token)
    decoder_input_ids = [[tokenizer.bos_token_id] + ids[:-1] for ids in padded_labels]

    # Shift labels to the left by one (remove the first token)
    shifted_labels = [ids[1:] + [pad_id] for ids in padded_labels]

    # Replace padding token id's in labels with -100 so they're ignored in loss calculation
    shifted_labels = [
        [label if label != pad_id else -100 for label in seq]
        for seq in shifted_labels
    ]

    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    decoder_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
    labels = torch.tensor(shifted_labels, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'decoder_ids': decoder_ids,
        'labels': labels
    }

def translate_sentence(sentence, model, tokenizer, device, max_length=50):
    model.eval()
    with torch.no_grad():
        # Tokenize and encode the input sentence
        input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
        
        # Prepare the target input (start with <s> token)
        decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        # Loop to generate tokens
        for _ in range(max_length):
            # Get the model outputs
            outputs = model(input_ids, decoder_input_ids)
            
            # Get the logits of the last generated token
            logits = outputs[:, -1, :]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get the token with the highest probability
            next_token = probs.argmax(dim=-1)
            
            # Append the predicted token to decoder_input_ids
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if the model predicts the end-of-sentence token
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Decode the generated tokens (excluding the first <s> token)
        generated_ids = decoder_input_ids[0, 1:]
        translated_sentence = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return translated_sentence


class Trainer:
    def __init__(
            self, 
            n_epochs,
            batch_size, 
            train_dataset,
            tokenizer
    ):
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer = Transformer().to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.transformer.parameters(),
            betas=[0.9, 0.98],
            eps=10e-9,
        )
        self.tokenizer = tokenizer
        self.loss_fn = torch.nn.CrossEntropyLoss() 

        
    def train(self):
        self.transformer.train()

        dataloader = DataLoader(
            self.train_dataset,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
            batch_size=self.batch_size,
            shuffle=True,
        )

        total_steps = self.epochs * len(dataloader)
        global_step = 0

        epoch_pbar = tqdm(range(self.epochs), desc="Epochs", position=0)
        step_pbar = tqdm(total=total_steps, desc="Steps", position=1)

        for _ in epoch_pbar:
            epoch_loss = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                decoder_ids = batch['decoder_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                output_blv = self.transformer(input_ids, decoder_ids)
                # here the output shape is b, l, vocab
                # convince myself that here we have b sequences of length l, 
                # with for each l tokens vocab logits of token n+1 
                # and the labels are b, l
                output_blv = output_blv.contiguous().view(-1, output_blv.size(-1))
                labels = labels.contiguous().view(-1)

                # Cross etnropy loss tkaes (N, C) where
                # n is the batch and C the classes.    
                loss = self.loss_fn(output_blv, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                # Update step progress bar
                step_pbar.update(1)
                step_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(dataloader)
            epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

        step_pbar.close()
        epoch_pbar.close()
        print("Training completed!")
        torch.save(self.transformer.state_dict(), 'model.pth')


def train_tokenizer(d):
    """
    Train a ByteLevel BPE tokenizer on the dataset.
    """
    # Get all texts from 'en' and 'fr' columns
    def batch_iterator(bs=1000):
        for i in range(0, len(d), bs):
            batch = d[i: i+bs]
            en_txt = [item['en'] for item in batch] 
            fr_txt = [item['fr'] for item in batch]

            txt = en_txt + fr_txt
            yield txt
    
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iterator=batch_iterator(),
        vocab_size=52000,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
    )
    tokenizer.save_model("tokenizer")
    return tokenizer


if __name__ == "__main__":

    from datasets import load_dataset
    from transformers import RobertaTokenizerFast
    
    # load the dataset
    ds = load_dataset("wmt/wmt14", "fr-en")

    #tokenizer = train_tokenizer(ds['train']['translation'])
    # print("trained tokenizer")
    #train = ds['train']

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "tokenizer", 
        max_len=512,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    trainer = Trainer(
        n_epochs=1,
        batch_size=32,
        train_dataset=ds['train']['translation'],
        tokenizer=tokenizer
    )
    
    trainer.train()
 
    device = torch.device('cuda')
    model = Transformer(
        block_size=512,
        d_model=512,
        in_vocab_s=len(tokenizer),
        out_vocab_s=len(tokenizer)
    ).to(device)

    model.load_state_dict(torch.load('model.pth', map_location=device))

    # Input sentence
    input_sentence = input("Enter a sentence to translate: ")
    
    # Translate the sentence
    translation = translate_sentence(input_sentence, model, tokenizer, device)
    
    print("Translation:", translation)

