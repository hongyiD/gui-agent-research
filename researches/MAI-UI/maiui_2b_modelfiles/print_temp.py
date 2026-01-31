import json
from pathlib import Path
from jinja2 import Template

# 1. 读取 sample_data.json 文件
json_file_path = Path(__file__).parent / "sample_data.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 读取 chat_template.jinja 模板内容
template_path = Path(__file__).parent / "chat_template.jinja"
with open(template_path, "r", encoding="utf-8") as f:
    template_str = f.read()

# 3. 加载模板
template = Template(template_str)

# 4. 准备传入模板的变量（从 JSON 数据中提取）
render_vars = {
    "messages": data.get("messages", []),
    "tools": data.get("tools", None),
    "add_vision_id": data.get("add_vision_id", False),
    "add_generation_prompt": data.get("add_generation_prompt", False)
}

# 5. 渲染模板，得到最终字符串
result = template.render(**render_vars)

# 6. 打印结果
print("渲染后的字符串：")
print("-" * 50)
print(result)