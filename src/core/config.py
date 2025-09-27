import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, Optional, Union

class Config:
    """配置管理类，使用单例模式确保全局只有一个配置实例"""
    _instance: Optional['Config'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 防止重复初始化
        if self._initialized:
            return
        
        self._config: Dict[str, Any] = {}
        # 加载环境变量
        self._load_env()
        # 加载YAML配置
        self._load_yaml_config()
        # 解析配置变量引用
        self._resolve_config_references()
        
        self._initialized = True
    
    def _load_env(self) -> None:
        """加载.env文件中的环境变量并存储到配置字典中"""
        # 查找.env文件的路径
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
        else:
            print(f"警告: 未找到.env文件: {env_path}")
        
        # 将所有环境变量添加到配置中
        for key, value in os.environ.items():
            self._config[key] = value
    
    def _load_yaml_config(self) -> None:
        """加载models.yaml文件中的配置"""
        yaml_path = Path(__file__).parent / "models.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        # 深度合并YAML配置到字典中
                        self._merge_config(self._config, yaml_config)
            except yaml.YAMLError as e:
                print(f"错误: 解析models.yaml文件失败: {e}")
            except Exception as e:
                print(f"错误: 加载models.yaml文件时出错: {e}")
        else:
            print(f"警告: 未找到models.yaml文件: {yaml_path}")
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """深度合并配置字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def _resolve_config_references(self) -> None:
        """解析配置中的环境变量引用，如SILICONFLOW_API_KEY"""
        try:
            for provider in self._config.get('model-provider', []):
                if provider in self._config and 'api_key' in self._config[provider]:
                    api_key_ref = self._config[provider]['api_key']
                    # 如果api_key是一个环境变量引用且在环境变量中存在
                    if api_key_ref in self._config:
                        self._config[provider]['api_key'] = self._config[api_key_ref]
        except Exception as e:
            print(f"警告: 解析配置引用时出错: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点表示法访问嵌套配置
        
        例如: config.get('siliconflow.api_key')
        
        Args:
            key: 配置键，可以使用点表示法访问嵌套配置
            default: 键不存在时返回的默认值
        
        Returns:
            配置值或默认值
        """
        # 支持点表示法访问嵌套配置
        if '.' in key:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if not isinstance(value, dict) or k not in value:
                    return default
                value = value[k]
            return value
        
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值，支持点表示法访问嵌套配置
        
        Args:
            key: 配置键，可以使用点表示法访问嵌套配置
            value: 要设置的配置值
        """
        if '.' in key:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                if k not in config or not isinstance(config[k], dict):
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
        else:
            self._config[key] = value
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔类型的配置值"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'y', 't')
        return bool(value)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数类型的配置值"""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数类型的配置值"""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """获取列表类型的配置值"""
        if default is None:
            default = []
        
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # 处理逗号分隔的字符串
            return [item.strip() for item in value.split(',') if item.strip()]
        return default
    
    def __getitem__(self, key: str) -> Any:
        """支持字典风格的配置访问"""
        if '.' in key:
            return self.get(key)
        
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """检查配置中是否包含指定的键"""
        if '.' in key:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if not isinstance(value, dict) or k not in value:
                    return False
                value = value[k]
            return True
        
        return key in self._config
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        # 过滤敏感信息
        filtered_config = self._filter_sensitive_info(self._config.copy())
        return yaml.dump(filtered_config, allow_unicode=True, default_flow_style=False)
    
    def _filter_sensitive_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """过滤配置中的敏感信息"""
        sensitive_keys = ['api_key', 'password', 'secret', 'token']
        
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = self._filter_sensitive_info(value)
            elif any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                config[key] = '****'
        
        return config

# 创建全局配置实例
config = Config()