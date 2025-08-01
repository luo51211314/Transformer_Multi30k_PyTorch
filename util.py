import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_attention_weights(model, src, trg, tokenizer, writer, epoch):
    model.eval()
    with torch.no_grad():
        # 类型和维度处理
        src_mask = model.make_src_mask(src)
        trg_mask = model.make_trg_mask(trg)
        
        enc = model.encoder(src, src_mask)
        output = model.decoder(enc, trg, src_mask, trg_mask)
        
        # Get attention weights from the first decoder layer
        decoder_layer = model.decoder.layers[0]
        if trg.size(-1) != model.d_model:
            trg = nn.Linear(trg.size(-1), model.d_model).to(trg.device)(trg.float())
            
        _, self_attn = decoder_layer.self_attention(decoder_layer.self_attention.w_q(trg.float()),
                                                  decoder_layer.self_attention.w_k(trg.float()),
                                                  decoder_layer.self_attention.w_v(trg.float()),
                                                  mask=trg_mask)
        
        # Plot attention weights
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self_attn[0, 0].cpu().numpy(), cmap='viridis')
        plt.colorbar(im)
        ax.set_title('Self-Attention Weights (Head 0)')
        ax.set_xticks(range(len(trg[0])))
        ax.set_yticks(range(len(trg[0])))
        writer.add_figure('attention/self_attention', fig, epoch)
        plt.close()