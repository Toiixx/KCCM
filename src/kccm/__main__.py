"""支持 ``python -m kccm``，与安装后的 ``kccm`` 命令行等价。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
