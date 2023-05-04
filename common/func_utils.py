import inspect
from collections import Counter
from typing import List, NamedTuple


class CheckRes(NamedTuple):
    missing: List[str]
    unused: List[str]
    duplicated: List[str]
    required: List[str]
    all_needed: List[str]
    varargs: List[str]


class CheckError(Exception):
    def __init__(self, check_res: CheckRes, func_signature: str):
        errs = [f"调用函数出现问题, 函数签名: `{func_signature}`"]

        if check_res.varargs:
            errs.append(f"\t可变参数: {check_res.varargs}")
        if check_res.missing:
            errs.append(f"\t缺失参数: {check_res.missing}")
        if check_res.duplicated:
            errs.append(f"\t重复参数: {check_res.duplicated}")
        if check_res.unused:
            errs.append(f"\t无用参数: {check_res.unused}")

        Exception.__init__(self, "\n".join(errs))

        self.check_res = check_res
        self.func_signature = func_signature



def refine_args(func, **kwargs) -> dict:
    """
    根据func的签名，从kwargs中选择func需要的参数
    """
    spec = inspect.getfullargspec(func)
    if spec.varkw is not None:
        return kwargs
    needed_args = set(spec.args)
    defaults = []
    if spec.defaults is not None:
        defaults = [arg for arg in spec.defaults]
    start_idx = len(spec.args) - len(defaults)
    output = {name: default for name, default in zip(spec.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output


def check_args(func, args) -> CheckRes:
    if isinstance(args, dict):
        arg_dict_list = [args]
    else:
        arg_dict_list = args
    assert callable(func) and isinstance(arg_dict_list, (list, tuple))
    assert len(arg_dict_list) > 0 and isinstance(arg_dict_list[0], dict)
    spec = inspect.getfullargspec(func)
    all_args = set([arg for arg in spec.args if arg != "self"])
    defaults = []
    if spec.defaults is not None:
        defaults = [arg for arg in spec.defaults]
    start_idx = len(spec.args) - len(defaults)
    default_args = set(spec.args[start_idx:])
    require_args = all_args - default_args
    input_arg_count = Counter()
    for arg_dict in arg_dict_list:
        input_arg_count.update(arg_dict.keys())
    duplicated = [name for name, val in input_arg_count.items() if val > 1]
    input_args = set(input_arg_count.keys())
    missing = list(require_args - input_args)
    unused = list(input_args - all_args)
    varargs = [] if not spec.varargs else [spec.varargs]
    return CheckRes(
        missing=missing,
        unused=unused,
        duplicated=duplicated,
        required=list(require_args),
        all_needed=list(all_args),
        varargs=varargs,
    )


def get_func_signature(func) -> str:
    """

    Given a function or method, return its signature.
    For example:

    1 function::

        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'

    2 method::

        class Demo:
            def __init__(self):
                xxx
            def forward(self, a, b='a', **args)
        demo = Demo()
        get_func_signature(demo.forward) # 'Demo.forward(self, a, b='a', **args)'

    :param func: a function or a method
    :return: str or None
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = "(self, "
        else:
            _self = "(self"
        signature_str = class_name + "." + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str
