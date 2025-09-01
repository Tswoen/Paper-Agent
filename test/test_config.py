from src.core.config import config

# 测试配置加载
print("===== 测试配置加载 =====")

# 测试环境变量加载
print("\n--- 环境变量 ---")
print(f"SILICONFLOW_API_KEY: {'已加载' if 'SILICONFLOW_API_KEY' in config else '未加载'}")
print(f"DASHSCOPE_API_KEY: {'已加载' if 'DASHSCOPE_API_KEY' in config else '未加载'}")
print(f"OPENAI_API_KEY: {'已加载' if 'OPENAI_API_KEY' in config else '未加载'}")

# 测试YAML配置加载
print("\n--- YAML配置 ---")
print(f"model-provider: {config.get('model-provider')}")

# 测试模型提供商配置
print("\n--- 模型提供商详细配置 ---")
for provider in config.get('model-provider', []):
    if provider in config:
        print(f"\n{provider}:")
        provider_config = config[provider]
        for key, value in provider_config.items():
            print(f"  {key}: {value}")

# 测试字典访问方式
try:
    print("\n--- 字典访问方式测试 ---")
    # 直接访问
    print(f"siliconflow api-key: {config['siliconflow']['api-key']}")
    # 使用get方法带默认值
    print(f"不存在的键: {config.get('non_existent_key', '默认值')}")
    print("字典访问方式测试通过！")
except Exception as e:
    print(f"字典访问方式测试失败: {e}")

print("\n===== 测试完成 =====")