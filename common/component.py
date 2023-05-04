import inspect
import logging
from registrable import Registrable

import json
import logging
from pathlib import Path
from typing import Union

from pyhocon import ConfigFactory, ConfigTree
from pyhocon.converter import HOCONConverter
logger = logging.getLogger(__name__)


class Con:
    """
    统一的配置文件
    配置编写使用 HOCON 支持变量/导入等
    序列化保存使用 JSON
    """

    @classmethod
    def load(cls, file: Union[str, Path], **kwargs) -> ConfigTree:
        if isinstance(file, Path):
            path: Path = file
        else:
            path = Path(file).expanduser().resolve()

        text = open(str(file)).read()
        if path.suffix == ".json":
            conf = ConfigFactory.from_dict(json.loads(text))
        elif path.suffix == ".conf":
            conf = ConfigFactory.parse_file(path)
        else:
            raise ValueError(f"不支持配置文件格式 {path.suffix}")

        conf.update(**kwargs)
        logger.info(f"从 {path} 读取配置\n{cls.render(conf)}")
        return conf

    @classmethod
    def dump(cls, config: ConfigTree, file: Union[str, Path]) -> None:
        logger.info(f"配置文件写入 {file}")
        with open(file, "w", encoding="utf-8") as f:
            f.write(cls.render(config))

    @classmethod
    def render(cls, config: ConfigTree) -> str:
        return HOCONConverter.to_json(config, indent=2)

class Component(Registrable):
    """
    避免BaseClass中的一些boilerplate，自动根据配置文件来初始化相应的子类实现
    """

    @classmethod
    def from_kwargs(cls, **kwargs):
        if isinstance(kwargs, str):
            kwargs = {"impl": kwargs}

        registered_subclasses = Registrable._registry.get(cls)
        if registered_subclasses and "impl" in kwargs:
            impl = kwargs.pop("impl")
            subclass = registered_subclasses[impl]
            logger.debug(f"通过 impl: {impl} 找到注册组件: {subclass}")
            if hasattr(subclass, "from_kwargs"):
                return subclass.from_kwargs(impl=impl, **kwargs)
            else:
                return subclass.__init__(**kwargs)
        else:
            kwargs.pop("impl", None)
            kwargs = cls.build_nested_kwargs(cls.__init__, **kwargs)
            logger.info(f"初始化: {cls.__name__} 参数: {kwargs}")
            return cls(**kwargs)

    @classmethod
    def build_nested_kwargs(cls, func, **kwargs) -> dict:
        spec = inspect.getfullargspec(func)
        for arg_name, arg_type in spec.annotations.items():
            param_type = type(kwargs[arg_name])
            if arg_type == param_type:
                continue
            if hasattr(arg_type, "from_kwargs") and param_type in (dict, Con):
                obj = arg_type.from_kwargs(**kwargs[arg_name])
                kwargs[arg_name] = obj
            elif param_type == list:
                objs = [arg_type.from_kwargs(**kw) for kw in kwargs[arg_name]]
                kwargs[arg_name] = objs

        # output = {name: default for name, default in zip(spec.args[start_idx:], defaults)}
        return kwargs
