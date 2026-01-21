# NumPy 版本兼容性问题修复指南

## 问题描述

运行 `sft_trainer.py` 时出现以下错误：

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.4.1 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

错误原因：NumPy 2.x 与使用 NumPy 1.x 编译的模块（如 `pyarrow`、`pandas`）不兼容。

## 解决方案

### 方法 1：重新安装依赖（推荐）

```bash
# 卸载当前 NumPy
pip uninstall numpy -y

# 安装兼容版本的 NumPy
pip install "numpy>=1.24.0,<2.0.0"

# 重新安装其他可能受影响的依赖
pip install --force-reinstall pandas pyarrow datasets
```

### 方法 2：使用更新后的 requirements.txt

```bash
# 重新安装所有依赖（确保使用更新后的 requirements.txt）
pip install -r requirements.txt --force-reinstall
```

### 方法 3：创建新的虚拟环境（最彻底）

```bash
# 创建新环境
conda create -n mai_ui_fixed python=3.11
conda activate mai_ui_fixed

# 安装依赖
pip install -r requirements.txt
```

## 验证修复

运行以下命令验证 NumPy 版本：

```bash
python -c "import numpy; print(numpy.__version__)"
# 应该输出类似: 1.26.x（而不是 2.x）

python -c "import pandas; import pyarrow; print('OK')"
# 应该输出: OK（无错误）
```

## 技术说明

- **根本原因**：NumPy 2.0 引入了破坏性变更，需要依赖库重新编译
- **当前状态**：`pyarrow`、`pandas` 等库尚未完全支持 NumPy 2.x
- **长期方案**：等待依赖库更新后，再升级到 NumPy 2.x

## 相关链接

- [NumPy 2.0 迁移指南](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [PyArrow NumPy 兼容性](https://arrow.apache.org/docs/python/numpy.html)
