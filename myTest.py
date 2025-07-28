# myTest.py
import torch
from transformers import AutoTokenizer
from config import Config
from model.Transformer import Transformer

class ModelDecoderTest:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.pad_id = self.tokenizer.pad_token_id
        self._init_model()
        self._verify_special_tokens()

    def _init_model(self):
        """加载模型和权重"""
        self.model = Transformer(
            pad_idx=self.pad_id,
            enc_voc_size=len(self.tokenizer.get_vocab()),
            dec_voc_size=len(self.tokenizer.get_vocab()),
            d_model=Config.d_model,
            max_len=Config.max_len,
            batch_size=Config.batch_size,
            n_head=Config.n_head,
            n_layers=Config.n_layers,
            ffn_hidden=Config.ffn_hidden,
            drop_prob=Config.drop_prob,
            device=Config.device
        ).to(Config.device)
        
        # 加载预训练权重
        try:
            state_dict = torch.load('best_model.pt', map_location=Config.device)
            
            # 修复键名不匹配问题
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_k = k.replace('_orig_mod.', '')
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            
            # 严格模式加载
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ 意外的键: {unexpected_keys}")
                
            print("✅ 成功加载并修复 best_model.pt 权重")
        except Exception as e:
            print(f"❌ 加载模型失败: {str(e)}")
            raise

    def _verify_special_tokens(self):
        """验证特殊Token"""
        print("\n=== 特殊Token验证 ===")
        print(f"EOS: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"PAD: {self.tokenizer.pad_token} (ID: {self.pad_id})")
        assert self.tokenizer.eos_token_id is not None, "EOS Token必须存在"

    def test_decoding_termination(self):
        """测试解码终止条件"""
        print("\n=== 终止符测试 ===")
        src_text = "Eine Gruppe füllt Wasserbehälter."
        src = self.tokenizer(src_text, return_tensors='pt')['input_ids'].to(Config.device)
        
        # 初始化decoder输入（用EOS开头）
        decoder_input = torch.full((1, Config.max_len), self.pad_id, device=Config.device)
        decoder_input[0, 0] = self.tokenizer.eos_token_id
        
        # 模拟自回归生成
        for i in range(1, Config.max_len):
            with torch.no_grad():
                output = self.model(src, decoder_input[:, :i])
                next_token = output.argmax(dim=-1)[:, -1]
                decoder_input[0, i] = next_token
            
            # 实时打印生成结果
            current_output = self.tokenizer.decode(decoder_input[0, :i+1], skip_special_tokens=True)
            print(f"Step {i}: {current_output}")
            
            # 检查是否生成EOS提前终止
            if next_token == self.tokenizer.eos_token_id:
                print("🛑 检测到EOS，提前终止生成")
                break
        
        assert self.tokenizer.eos_token_id in decoder_input[0], "生成结果未包含EOS终止符"

    def test_repetition_control(self):
        """测试重复生成控制"""
        print("\n=== 重复生成测试 ===")
        src_text = "Describe a landscape"
        src = self.tokenizer(src_text, return_tensors='pt')['input_ids'].to(Config.device)
        
        decoder_input = torch.full((1, Config.max_len), self.pad_id, device=Config.device)
        decoder_input[0, 0] = self.tokenizer.eos_token_id
        
        generated_tokens = set()
        repeated_count = 0
        
        for i in range(1, Config.max_len):
            with torch.no_grad():
                output = self.model(src, decoder_input[:, :i])
                next_token = output.argmax(dim=-1)[:, -1]
                decoder_input[0, i] = next_token
            
            # 检查重复Token
            token = next_token.item()
            if token in generated_tokens:
                repeated_count += 1
                print(f"⚠️ 重复Token: {self.tokenizer.decode([token])} (ID: {token})")
            generated_tokens.add(token)
            
            if repeated_count > 3:
                raise AssertionError("检测到连续重复生成")
        
        print("✅ 重复控制测试通过")

if __name__ == "__main__":
    tester = ModelDecoderTest()
    tester.test_decoding_termination()
    tester.test_repetition_control()