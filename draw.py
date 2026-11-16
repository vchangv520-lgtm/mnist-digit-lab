"""打开你亲手训练的 Linear model 画板：.venv/bin/python draw.py。"""

from app import main
from model import DEFAULT_LINEAR_MODEL


if __name__ == "__main__":
    # 从保存的 state_dict 恢复你的 model；这个入口只进行 prediction。
    main(default_model=DEFAULT_LINEAR_MODEL, default_port=7861)
