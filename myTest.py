import torch
from config import Config
from model.Transformer import Transformer
from other.dataloader import DataLoaderHF
from other.BLEU import get_bleu
from transformers import AutoTokenizer
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='test_log.txt',
    filemode='a'
)
logger = logging.getLogger(__name__)

def load_model():
    # 初始化数据加载器
    dataloader = DataLoaderHF(Config.model_name, Config.max_len, Config.batch_size, Config.special_token)
    tokenizer = dataloader.tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': [Config.special_token]})
    pad_id = tokenizer.pad_token_id
    
    # 初始化模型
    model = Transformer(
        pad_idx=pad_id,
        enc_voc_size=len(tokenizer.get_vocab()),
        dec_voc_size=len(tokenizer.get_vocab()),
        d_model=Config.d_model,
        max_len=Config.max_len,
        batch_size=1,  # 交互模式下batch_size设为1
        n_head=Config.n_head,
        n_layers=Config.n_layers,
        ffn_hidden=Config.ffn_hidden,
        drop_prob=Config.drop_prob,
        device=Config.device
    )
    
    # 加载预训练权重（处理torch.compile前缀）
    state_dict = torch.load("best_model.pt")
    fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict)
    model.to(Config.device)
    model.eval()
    
    return model, tokenizer

def translate(model, tokenizer, src_text):
    # 预处理输入文本
    src = tokenizer(
        src_text, 
        padding="max_length",
        truncation=True,
        max_length=Config.max_len,
        return_tensors="pt"
    )['input_ids'].to(Config.device)
    
    # 初始化decoder输入
    decoder_output = torch.full((1, Config.max_len), tokenizer.pad_token_id, device=Config.device)
    decoder_output[:, 0] = tokenizer.convert_tokens_to_ids(Config.special_token)
    
    # 生成mask
    src_mask = model.make_src_mask(src)
    
    # 编码器前向传播
    with torch.no_grad():
        encoder_output = model.encoder(src, src_mask)
        
        # 自回归生成
        for j in range(1, Config.max_len):
            trg_mask = model.make_trg_mask(decoder_output)
            output = model.decoder(encoder_output, decoder_output, src_mask, trg_mask)
            output = output.argmax(dim=2)
            output = output[:, j-1]
            decoder_output[:, j] = output
            
            # 如果生成了结束标记则提前终止
            if output.item() == tokenizer.eos_token_id:
                break
    
    # 解码输出
    translated_text = tokenizer.decode(
        decoder_output[0], 
        skip_special_tokens=True
    )
    
    return translated_text

def interactive_test(model, tokenizer):
    print("\n欢迎使用Transformer翻译系统(德语->英语)")
    print("输入'quit'退出程序\n")
    
    while True:
        src_text = input("请输入德语文本: ")
        
        if src_text.lower() == 'quit':
            print("感谢使用，再见！")
            break
            
        if not src_text.strip():
            print("输入不能为空，请重新输入！")
            continue
            
        # 获取参考翻译（可选）
        ref_text = input("请输入参考英语翻译(可选，直接回车跳过): ")
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 进行翻译
        translated_text = translate(model, tokenizer, src_text)
        
        # 计算处理时间
        process_time = (datetime.now() - start_time).total_seconds()
        
        # 输出结果
        print("\n翻译结果:")
        print(f"德语原文: {src_text}")
        print(f"英语译文: {translated_text}")
        
        # 如果有参考翻译，计算BLEU分数
        if ref_text.strip():
            bleu_score = get_bleu(
                hypothesis=translated_text.split(),
                reference=ref_text.split()
            )
            print(f"BLEU分数: {bleu_score:.2f}")
        else:
            bleu_score = None
        
        # 记录日志
        log_entry = {
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_text": src_text,
            "translated_text": translated_text,
            "reference_text": ref_text if ref_text.strip() else None,
            "bleu_score": bleu_score,
            "processing_time": f"{process_time:.2f}s"
        }
        
        logger.info(
            f"Input: {src_text} | "
            f"Output: {translated_text} | "
            f"Reference: {ref_text if ref_text.strip() else 'None'} | "
            f"BLEU: {bleu_score if bleu_score is not None else 'N/A'} | "
            f"Time: {process_time:.2f}s"
        )
        
        print("-" * 50 + "\n")

if __name__ == "__main__":
    try:
        print("正在加载模型...")
        model, tokenizer = load_model()
        print("模型加载完成！")
        
        interactive_test(model, tokenizer)
    except Exception as e:
        logger.error(f"程序出错: {str(e)}", exc_info=True)
        print(f"发生错误: {str(e)}")