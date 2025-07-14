import os
import sys


def add_project_root_to_sys_path():
    """
    向上遍历目录树，查找包含 'pyproject.toml' 文件的项目根目录，
    并将其添加到 sys.path 中。
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = None
    # 向上查找最多5层目录，以避免无限循环并限制搜索范围
    for _ in range(5):
        if os.path.exists(os.path.join(current_script_dir, 'pyproject.toml')):
            project_root = current_script_dir
            break
        parent_dir = os.path.dirname(current_script_dir)
        if parent_dir == current_script_dir:  # 达到文件系统根目录
            break
        current_script_dir = parent_dir

    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"项目根目录 '{project_root}' 已添加到 sys.path。")
    elif not project_root:
        print("未找到包含 'pyproject.toml' 的项目根目录。")
