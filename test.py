import torch
import torch.nn as nn
from config import Config
from other.BLEU import bleu_stats, bleu, get_bleu

def evaluate(model, valid_iter, criterion, tokenizer):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(valid_iter):
            src = batch['input_ids'].to(Config.device)
            trg = batch['labels'].to(Config.device)

            output = model(src, trg)
            output = output[:, :-1, :]
            output_reshape = output.contiguous().view(-1, output.shape[-1]).to(Config.device, non_blocking=True)
            trg = trg[:, 1:]
            trg_reshape = trg.contiguous().view(-1).to(Config.device, non_blocking=True)
            loss = criterion(output_reshape, trg_reshape)
            epoch_loss += loss.item()

            total_bleu = []
            output = output.argmax(dim=2)
            for j in range(len(src)):
                try:
                    trg_words = tokenizer.decode(trg[j], skip_special_tokens=True)
                    output_words = tokenizer.decode(output[j], skip_special_tokens=True)
                    bleu = get_bleu(hypothesis=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass
              
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
    
    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(valid_iter), batch_bleu

def inference(model, test_iter, tokenizer, pad_id):
    bleu = []
    model.eval()
    for i, batch in enumerate(test_iter):
        src = batch['input_ids'].to(Config.device)
        trg = batch['labels'].to(Config.device)

        src_mask = model.make_src_mask(src)
        
        decoder_output = torch.full((Config.batch_size, Config.max_len), pad_id)
        decoder_output[:, 0] = tokenizer.bos_token_id

        encoder_output = model.encoder(src, src_mask)
        
        for j in range(1, Config.max_len):
            trg_mask = model.make_trg_mask(decoder_output)
            output = model.decoder(encoder_output, decoder_output, src_mask, trg_mask)
            output = output.argmax(dim=2)
            output = output[:, j-1]
            decoder_output[:, j] = output
            
        for j in range(Config.batch_size):
            decoder_text = tokenizer.decode(decoder_output[j], skip_special_tokens=True)
            trg_text = tokenizer.decode(trg[j], skip_special_tokens=True)
            single_bleu = get_bleu(hypothesis=decoder_text.split(), reference=trg_text.split())
            bleu.append(single_bleu)

    return bleu