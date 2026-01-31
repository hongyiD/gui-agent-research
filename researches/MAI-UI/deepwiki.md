https://deepwiki.com/search/scalefactor-999_cf84e5f1-9ce8-491e-99dd-7df679f3dc49?mode=fast

# 我问
SCALE_FACTOR 这个值定义了999，是为啥？


# deepwiki
SCALE_FACTOR 定义为 999 是用于坐标归一化的缩放因子。 [1](#0-0) [2](#0-1) 

## 详细说明

### 用途
SCALE_FACTOR 主要用于将模型输出的坐标从 [0, 999] 范围归一化到 [0, 1] 范围：

1. **坐标归一化**：在 `parse_grounding_response` 函数中，将模型返回的坐标值除以 SCALE_FACTOR 得到归一化坐标 [3](#0-2) 

2. **动作解析**：在 `parse_action_to_structure_output` 函数中同样使用 SCALE_FACTOR 进行坐标归一化 [4](#0-3) 

3. **历史记录转换**：在生成历史响应时，将归一化坐标乘以 SCALE_FACTOR 转换回原始范围 [5](#0-4) 

### 为什么选择 999
选择 999 而不是 1000 可能有以下考虑：
- **避免边界问题**：999 作为最大值，坐标范围是 [0, 999]，避免了 1000 可能带来的边界混淆
- **浮点精度**：在某些浮点运算中，999 可能比 1000 有更好的数值特性
- **模型兼容性**：可能是训练模型时使用的特定标准化参数

## Notes
SCALE_FACTOR 在两个不同的 agent 文件中都有相同的定义，表明这是整个系统的统一坐标标准化方案。这个值的选择可能是基于模型训练时的具体需求和数值稳定性考虑。

### Citations

**File:** src/mai_grounding_agent.py (L34-34)
```python
SCALE_FACTOR = 999
```

**File:** src/mai_grounding_agent.py (L74-77)
```python
                # Normalize coordinates from SCALE_FACTOR range to [0, 1]
                point_x = coordinates[0] / SCALE_FACTOR
                point_y = coordinates[1] / SCALE_FACTOR
                result["coordinate"] = [point_x, point_y]
```

**File:** src/mai_naivigation_agent.py (L38-38)
```python
SCALE_FACTOR = 999
```

**File:** src/mai_naivigation_agent.py (L129-144)
```python
    # Normalize coordinates from SCALE_FACTOR range to [0, 1]
    if "coordinate" in action:
        coordinates = action["coordinate"]
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}"
            )
        point_x = point_x / SCALE_FACTOR
        point_y = point_y / SCALE_FACTOR
        action["coordinate"] = [point_x, point_y]
```

**File:** src/mai_naivigation_agent.py (L253-267)
```python
            # Convert normalized coordinates back to SCALE_FACTOR range
            if "coordinate" in action_json:
                coordinates = action_json.get("coordinate", [])
                if len(coordinates) == 2:
                    point_x, point_y = coordinates
                elif len(coordinates) == 4:
                    x1, y1, x2, y2 = coordinates
                    point_x = (x1 + x2) / 2
                    point_y = (y1 + y2) / 2
                else:
                    continue
                action_json["coordinate"] = [
                    int(point_x * SCALE_FACTOR),
                    int(point_y * SCALE_FACTOR),
                ]
```
