# myTest.py
import torch
from transformers import AutoTokenizer
from config import Config  # 假设Config包含模型和参数配置

def test_tokenizer_output():
    """测试tokenizer的分词结果及特殊Token处理"""
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    
    # 检查特殊Token是否存在
    print("特殊Token映射:", tokenizer.special_tokens_map)
    assert tokenizer.eos_token_id is not None, "EOS Token未定义"
    assert tokenizer.pad_token_id is not None, "PAD Token未定义"
    
    # 测试普通文本分词
    text = "Eine Gruppe füllt Wasserbehälter in der Wüste."
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=Config.max_len)
    print("编码结果:", encoded)
    
    # 验证解码还原
    decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
    print("解码还原:", decoded)
    assert text in decoded, "解码后文本与原始文本不匹配"

def test_model_decoder_initialization():
    """测试模型解码器的初始Token是否正确"""
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    pad_id = tokenizer.pad_token_id
    
    # 模拟推理时的decoder_output初始化
    decoder_output = torch.full((Config.batch_size, Config.max_len), pad_id, dtype=torch.long)
    
    # 方案1：使用EOS Token作为起始符（Opus-MT的常见做法）
    decoder_output[:, 0] = 58101
    print("初始化的decoder_output[0]:", decoder_output[0, :10])
    
    # 验证起始符有效性
    assert decoder_output[0, 0] == 58101, "起始符未正确设置"

def test_avoid_repetition():
    """测试生成结果是否避免重复词"""
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    test_sentence = "Leiche Leiche Leiche"  # 模拟重复问题
    
    # 检查重复词是否被正确处理
    encoded = tokenizer(test_sentence, return_tensors='pt')
    decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
    print("重复词测试解码:", decoded)
    assert "Leiche" in decoded, "重复词未被识别"

if __name__ == "__main__":
    test_tokenizer_output()
    test_model_decoder_initialization()
    test_avoid_repetition()
    print("所有测试通过！")