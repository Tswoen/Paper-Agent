import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

class Config(dict):
    def __init__(self):
        super().__init__()
        # 加载环境变量
        self._load_env()
        # 加载YAML配置
        self._load_yaml_config()
        # 解析配置变量引用
        self._resolve_config_references()
    
    def _load_env(self):
        """加载.env文件中的环境变量"""
        # 查找.env文件的路径
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # 将所有环境变量添加到配置中
        for key, value in os.environ.items():
            self[key] = value
    
    def _load_yaml_config(self):
        """加载models.yaml文件中的配置"""
        yaml_path = Path(__file__).parent / "models.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                # 将YAML配置合并到字典中
                for key, value in yaml_config.items():
                    self[key] = value
    
    def _resolve_config_references(self):
        """解析配置中的环境变量引用，如SILICONFLOW_API_KEY"""
        for provider in self.get('model-provider', []):
            if provider in self and 'api-key' in self[provider]:
                api_key_ref = self[provider]['api-key']
                # 如果api-key是一个环境变量引用且在环境变量中存在
                if api_key_ref in self:
                    self[provider]['api-key'] = self[api_key_ref]
            
    def get(self, key, default=None):
        """重写get方法，提供默认值"""
        return super().get(key, default)
    
# 创建全局配置实例
config = Config()