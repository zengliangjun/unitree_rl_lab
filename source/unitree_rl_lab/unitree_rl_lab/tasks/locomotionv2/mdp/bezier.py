import torch

# ============================================================================
# 贝塞尔曲线工具函数
# ============================================================================

def _parse_control_points(control_points, default_points=None):
    """
    解析多种格式的控制点，返回标准化格式的列表[(x1, y1), (x2, y2), ...]

    支持的格式：
    1. 列表格式: [(0.0, 0.0), (0.2, 0.9), (0.8, 0.2), (1.0, 0.0)]
    2. 字符串格式: "0.0,0.0;0.2,0.9;0.8,0.2;1.0,0.0"
    3. 嵌套列表: [[0.0, 0.0], [0.2, 0.9], [0.8, 0.2], [1.0, 0.0]]

    Args:
        control_points: 控制点输入
        default_points: 默认控制点，当control_points为None时使用

    Returns:
        list[tuple[float, float]]: 标准化控制点列表
    """
    if control_points is None:
        if default_points is None:
            # 默认的三次贝塞尔曲线控制点（快速上升，缓慢下降）
            return [(0.0, 0.0), (0.2, 0.9), (0.8, 0.2), (1.0, 0.0)]
        return default_points

    # 如果是字符串格式
    if isinstance(control_points, str):
        points = []
        for point_str in control_points.split(';'):
            if ',' in point_str:
                x_str, y_str = point_str.split(',', 1)
                points.append((float(x_str.strip()), float(y_str.strip())))
            else:
                raise ValueError(f"Invalid control point format: {point_str}. Expected 'x,y'")
        return points

    # 如果是列表/元组格式
    if isinstance(control_points, (list, tuple)):
        result = []
        for point in control_points:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                result.append((float(point[0]), float(point[1])))
            elif isinstance(point, str) and ',' in point:
                x_str, y_str = point.split(',', 1)
                result.append((float(x_str.strip()), float(y_str.strip())))
            else:
                raise ValueError(f"Invalid control point element: {point}")
        return result

    raise ValueError(f"Unsupported control points format: {type(control_points)}")


def _validate_control_points(control_points):
    """
    验证控制点的有效性

    Args:
        control_points: 控制点列表[(x, y), ...]

    Raises:
        ValueError: 如果控制点无效
    """
    if not control_points:
        raise ValueError("Control points cannot be empty")

    if len(control_points) < 2:
        raise ValueError(f"Need at least 2 control points, got {len(control_points)}")

    # 检查x坐标是否在[0, 1]范围内且单调递增
    prev_x = -float('inf')
    for i, (x, y) in enumerate(control_points):
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"Control point {i}: x coordinate {x} must be in [0, 1]")
        if x <= prev_x:
            raise ValueError(f"Control point {i}: x coordinate {x} must be greater than previous {prev_x}")
        prev_x = x

    # 检查y坐标是否在合理范围内（允许稍微超出[0, 1]）
    for i, (x, y) in enumerate(control_points):
        if not (-0.5 <= y <= 2.0):
            raise ValueError(f"Control point {i}: y coordinate {y} should be in [-0.5, 2.0] for reasonable curves")


def _compute_bezier_curve(t: torch.Tensor, control_points: list[tuple[float, float]]) -> torch.Tensor:
    """
    计算贝塞尔曲线值（通用实现，支持任意阶数）

    使用德卡斯特里奥算法（De Casteljau's algorithm）

    Args:
        t: 参数值，形状为任意，值在[0, 1]之间
        control_points: 控制点列表[(x, y), ...]

    Returns:
        torch.Tensor: 贝塞尔曲线在参数t处的y值
    """
    # 将控制点转换为张量
    n = len(control_points)
    if n == 0:
        return torch.zeros_like(t)

    t_shape = t.shape
    t = t.view(-1)


    # 提取y坐标作为初始控制点值
    points = torch.tensor([cp[1] for cp in control_points], dtype=torch.float32, device=t.device)

    # 扩展维度以支持广播
    points = points.view(1, -1).repeat(t.shape[0], 1)

    # 德卡斯特里奥算法
    for r in range(1, n):
        for i in range(n - r):
            points[:, i] = (1 - t) * points[:, i] + t * points[:, i + 1]

    return points[:, 0].view(*t_shape)


def _bezier_target_height(swing_phase: torch.Tensor, target_height: float,
                         control_points: list[tuple[float, float]] | None = None) -> torch.Tensor:
    """
    根据摆动相位计算贝塞尔曲线目标高度

    Args:
        swing_phase: 摆动相位，值在[0, 1]之间
        target_height: 目标高度缩放因子
        control_points: 贝塞尔曲线控制点

    Returns:
        torch.Tensor: 目标高度
    """
    # 解析和验证控制点
    parsed_points = _parse_control_points(control_points)
    _validate_control_points(parsed_points)

    # 计算贝塞尔曲线值
    bezier_value = _compute_bezier_curve(swing_phase, parsed_points)

    # 应用目标高度缩放
    return bezier_value * target_height


if __name__ == "__main__":

    swing_phase = torch.abs(torch.randn((1, 2), dtype=torch.float32))
    points = [(0.0, 0.0), (0.25, 2), (0.8, 0.4), (1.0, 0.0)]
    for i in range(100):
        swing_phase += (0.02 / 0.6)
        swing_phase %= 1

        height = _bezier_target_height(swing_phase, 1, points)
        print(i, height)

