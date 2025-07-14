# use wsl
  * 不建议在wsl内启动vscode,因为mcp工具不能正常使用
  * 不建议在windows路径上运行项目,因为性能差,建议移动到wsl的 ~/ 路径
  * 访问`\\wsl.localhost\Ubuntu\home\hxse\jax_demo`,鼠标右键通过code打开
  * 在powershell运行命令,roocode可正常识别
    * `wsl -d Ubuntu bash -i -c 'cd /home/hxse/jax_demo && uv run python main.py'`
