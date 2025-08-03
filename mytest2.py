from other.BLEU_2 import Evaluator
from model.Transformer import Transformer
from other.dataloader_wmt import DataLoaderWMT
import torch
from config import Config
from datetime import datetime
import logging
import math

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='test_log_sacrebleu.txt',
    filemode='a'
)
logger = logging.getLogger(__name__)

def load_model():
    """加载训练好的Transformer模型"""
    dataloader = DataLoaderWMT(Config.model_name, Config.max_len, Config.batch_size, Config.special_token)
    tokenizer = dataloader.tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': [Config.special_token]})
    
    model = Transformer(
        pad_idx=tokenizer.pad_token_id,
        enc_voc_size=len(tokenizer.get_vocab()),
        dec_voc_size=len(tokenizer.get_vocab()),
        d_model=Config.d_model,
        max_len=Config.max_len,
        batch_size=1,
        n_head=Config.n_head,
        n_layers=Config.n_layers,
        ffn_hidden=Config.ffn_hidden,
        drop_prob=Config.drop_prob,
        device=Config.device
    )
    
    state_dict = torch.load("best_model.pt")
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
    return model.to(Config.device), tokenizer

def translate_with_ppl(model, tokenizer, src_text):
    """翻译函数（修正版PPL计算）"""
    src = tokenizer(
        src_text, 
        padding="max_length",
        truncation=True,
        max_length=Config.max_len,
        return_tensors="pt"
    )['input_ids'].to(Config.device)
    
    decoder_output = torch.full((1, Config.max_len), tokenizer.pad_token_id, device=Config.device)
    decoder_output[:, 0] = tokenizer.convert_tokens_to_ids(Config.special_token)
    
    total_log_prob = 0.0
    valid_tokens = 0
    generated_length = 0
    
    with torch.no_grad():
        encoder_output = model.encoder(src, model.make_src_mask(src))
        for j in range(1, Config.max_len):
            # 获取当前步的预测logits
            output = model.decoder(
                encoder_output, 
                decoder_output, 
                model.make_src_mask(src),
                model.make_trg_mask(decoder_output)
            )
            
            # 生成下一个token（使用贪心解码）
            logits = output[:, j-1, :]
            next_token = logits.argmax(dim=-1)
            decoder_output[:, j] = next_token
            
            # 计算当前生成token的log概率
            log_probs = torch.log_softmax(logits, dim=-1)
            current_log_prob = log_probs[0, next_token].item()
            #print(f"curr_log_prob:{current_log_prob}", current_log_prob)
            
            # 仅计算已生成部分的PPL（排除pad和未生成部分）
            if next_token.item() != tokenizer.pad_token_id:
                total_log_prob += current_log_prob
                valid_tokens += 1
                generated_length = j  # 记录实际生成长度
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 使用数值稳定的方式计算PPL
    #print(f"valid_tokens:{valid_tokens}", valid_tokens)
    avg_log_prob = total_log_prob / max(valid_tokens, 1)  # 防止除以0
    #print(f"avg_prob:{avg_log_prob}", avg_log_prob)
    ppl = math.exp(-avg_log_prob) if valid_tokens > 0 else float('inf')
    
    # 截断到实际生成长度
    final_output = decoder_output[:, :generated_length+1]
    return tokenizer.decode(final_output[0], skip_special_tokens=True), ppl

def interactive_test():
    """交互式测试"""
    model, tokenizer = load_model()
    evaluator = Evaluator()
    
    print("\nTransformer翻译系统（德语->英语）[SacreBLEU+PPL]")
    print("输入'quit'退出\n")
    
    while True:
        src_text = input("请输入德语文本: ").strip()
        if src_text.lower() == 'quit':
            break
            
        start_time = datetime.now()
        translated_text, ppl = translate_with_ppl(model, tokenizer, src_text)
        process_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n翻译结果: {translated_text}")
        print(f"生成PPL: {ppl:.2f}")
        
        if ref_text := input("请输入参考英语翻译（可选，回车跳过）: ").strip():
            eval_result = evaluator.evaluate_all(
                hypothesis=[translated_text],
                references=[ref_text],
                src_text=src_text
            )
            print(f"BLEU: {eval_result['bleu']:.2f}")
            logger.info(
                f"Input: {src_text} | Output: {translated_text} | "
                f"BLEU: {eval_result['bleu']:.2f} | PPL: {ppl:.2f} | "
                f"Time: {process_time:.2f}s"
            )
        print("-" * 50)

if __name__ == "__main__":
    try:
        interactive_test()
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")