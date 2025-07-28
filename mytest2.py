import torch


def inspect_model_file(file_path="best_model.pt"):
    """检查 .pt 文件内容"""
    try:
        # 加载文件内容
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

        print(f"文件 '{file_path}' 包含:")

        # 检查是否是完整模型还是仅 state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("类型: 完整检查点（包含模型和训练状态）")
            state_dict = checkpoint['state_dict']
            print(f"包含的训练信息: {list(checkpoint.keys())}")
        else:
            print("类型: 仅模型 state_dict")
            state_dict = checkpoint

        # 打印参数键和形状
        print("\n模型参数键:")
        for key in list(state_dict.keys())[:10]:  # 只显示前10个
            shape = state_dict[key].shape
            print(f"- {key}: {shape}")

        print(f"\n总共 {len(state_dict)} 个参数张量")

        # 显示参数示例
        sample_key = list(state_dict.keys())[0]
        print(f"\n参数 '{sample_key}' 示例值:")
        print(state_dict[sample_key])

        return state_dict
    except Exception as e:
        print(f"加载文件出错: {e}")
        return None


if __name__ == "__main__":
    inspect_model_file("best_model.pt")