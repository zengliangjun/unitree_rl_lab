import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 启用交互模式，适配Matlab的动态绘图
plt.ion()

def plot_ankle_mechanism_animation():
    # ===================== 参数定义（与Matlab完全一致）=====================
    l_axis = 0.03685    # A点轴长
    l_axis_C = 0.033    # C点轴长
    l_bar = 0.0225      # 杆长
    l_bar_c = 0.0185    # 杆长
    l_rod1 = 0.2105     # 推杆1长度（注意Matlab中l_rod1和l_rod2与原Python互换）
    l_rod2 = 0.1455     # 推杆2长度
    l_2 = 0.0           # 踝关节长度
    lz = 0.01           # 控制面和十字中心的高度

    # ===================== 初始位置计算（与Matlab完全一致）=====================
    C1_0 = np.array([-l_axis_C, l_bar_c, -lz])
    C2_0 = np.array([-l_axis_C, -l_bar_c, -lz])
    A1_0 = np.array([-l_axis, 0, 0.200])  # Matlab中A1_0是0.200，A2_0是0.135
    A2_0 = np.array([-l_axis, 0, 0.135])
    B1_0 = A1_0 + np.array([0, l_bar, 0])
    B2_0 = A2_0 + np.array([0, -l_bar, 0])
    O1 = np.array([0.0, 0.0, 0.0])
    O2 = np.array([0.0, 0.0, 0.0])

    # ===================== 创建图形（复刻Matlab的figure设置）=====================
    fig = plt.figure(figsize=(12, 8))  # Matlab Position [100,100,1200,800]对应figsize
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    ax.axis('equal')
    ax.view_init(elev=30, azim=-45)    # 对应Matlab view(-45,30)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Ankle Joint Mechanism Animation - Roll and Pitch from -10° to 10°', fontsize=14)
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.05, 0.25)

    # ===================== 动画参数（与Matlab完全一致）=====================
    num_frames = 50
    roll_range = np.linspace(15, 0, num_frames)    # Matlab: linspace(15, 0, num_frames)
    pitch_range = np.linspace(0, 0, num_frames)    # Matlab: linspace(0, 0, num_frames)

    # ===================== 初始化图形对象句柄（复刻Matlab的struct）=====================
    h = {}

    # ===================== 预计算最大/最小theta角度（复刻Matlab逻辑）=====================
    # 最大值（对应Matlab num_frames-1）
    roll = np.deg2rad(roll_range[-2])
    pitch = np.deg2rad(pitch_range[-2])
    C1_rot = compute_rotated_C(roll, pitch, C1_0, l_2)
    C2_rot = compute_rotated_C(roll, pitch, C2_0, l_2)
    theta1, theta2 = compute_theta(C1_rot, C2_rot, A1_0, A2_0, l_rod1, l_rod2, l_bar)
    print(f'最大值θ1 = {np.rad2deg(theta1):.2f} deg, θ2 = {np.rad2deg(theta2):.2f} deg')

    # 最小值（对应Matlab 1）
    roll = np.deg2rad(roll_range[0])
    pitch = np.deg2rad(pitch_range[0])
    C1_rot = compute_rotated_C(roll, pitch, C1_0, l_2)
    C2_rot = compute_rotated_C(roll, pitch, C2_0, l_2)
    theta1, theta2 = compute_theta(C1_rot, C2_rot, A1_0, A2_0, l_rod1, l_rod2, l_bar)
    print(f'最小值θ1 = {np.rad2deg(theta1):.2f} deg, θ2 = {np.rad2deg(theta2):.2f} deg')

    # ===================== 动画循环（复刻Matlab的while(true)+for循环）=====================
    try:
        while True:
            for i in range(num_frames):
                # 清除上一帧（复刻Matlab的delete_plot_objects）
                if i > 0:
                    delete_plot_objects(h)

                # 计算当前角度
                roll = np.deg2rad(roll_range[i])
                pitch = np.deg2rad(pitch_range[i])

                # 计算旋转后的C点
                C1_rot = compute_rotated_C(roll, pitch, C1_0, l_2)
                C2_rot = compute_rotated_C(roll, pitch, C2_0, l_2)

                # 计算theta角度
                theta1, theta2 = compute_theta(C1_rot, C2_rot, A1_0, A2_0, l_rod1, l_rod2, l_bar)

                # 计算旋转后的B点（X轴旋转）
                B1_rot = A1_0 + rotation_x(theta1) @ (B1_0 - A1_0)
                B2_rot = A2_0 + rotation_x(theta2) @ (B2_0 - A2_0)

                # 绘制当前帧机构
                h = plot_mechanism_animation(ax, A1_0, B1_rot, C1_rot, A2_0, B2_rot, C2_rot, O1, O2, h)
                # 绘制旋转轴
                h = add_rotation_axes_animation(ax, O1, O2, roll, pitch, h)

                # 添加角度信息文本（复刻Matlab的text）
                if 'info_text' in h and h['info_text'] is not None:
                    h['info_text'].remove()
                info_str = (f'Roll: {np.rad2deg(roll):.1f}°\nPitch: {np.rad2deg(pitch):.1f}°\n'
                            f'Motor1 θ1: {np.rad2deg(theta1):.2f}°\nMotor2 θ2: {np.rad2deg(theta2):.2f}°')
                h['info_text'] = ax.text(-0.08, -0.08, 0.2, info_str,
                                        fontsize=12, fontweight='bold',
                                        bbox=dict(facecolor='white', alpha=1.0))

                # 刷新图形（复刻Matlab的drawnow）
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                # 暂停控制速度（复刻Matlab的pause(0.05)）
                time.sleep(0.05)

                # 帧结束时删除对象（复刻Matlab的delete_plot_objects）
                delete_plot_objects(h)

    except KeyboardInterrupt:
        # 按Ctrl+C退出循环，避免程序卡死
        plt.close(fig)
        print("动画已停止")

def delete_plot_objects(h):
    """复刻Matlab的delete_plot_objects函数：删除所有图形对象"""
    for key in list(h.keys()):
        if key == 'info_text':
            continue  # info_text单独处理
        try:
            if isinstance(h[key], list):
                for obj in h[key]:
                    obj.remove()
            else:
                h[key].remove()
        except:
            pass
        h[key] = None

def plot_mechanism_animation(ax, A1, B1, C1, A2, B2, C2, O1, O2, h):
    """复刻Matlab的plot_mechanism_animation函数：绘制机构动画"""
    # 绘制点（复刻scatter3）
    h['A1_scatter'] = ax.scatter(A1[0], A1[1], A1[2], s=100, c='r', marker='o', edgecolors='none')
    h['A2_scatter'] = ax.scatter(A2[0], A2[1], A2[2], s=100, c='b', marker='o', edgecolors='none')
    h['B1_scatter'] = ax.scatter(B1[0], B1[1], B1[2], s=80, c='r', marker='s')
    h['B2_scatter'] = ax.scatter(B2[0], B2[1], B2[2], s=80, c='b', marker='s')
    h['C1_scatter'] = ax.scatter(C1[0], C1[1], C1[2], s=80, c='r', marker='^')
    h['C2_scatter'] = ax.scatter(C2[0], C2[1], C2[2], s=80, c='b', marker='^')
    h['O1_scatter'] = ax.scatter(O1[0], O1[1], O1[2], s=100, c='k', marker='o', edgecolors='none')
    h['O2_scatter'] = ax.scatter(O2[0], O2[1], O2[2], s=100, c='m', marker='o', edgecolors='none')

    # 绘制连线（复刻plot3）
    h['AB1_line'] = ax.plot3D([A1[0], B1[0]], [A1[1], B1[1]], [A1[2], B1[2]], 'r-', linewidth=3)[0]
    h['AB2_line'] = ax.plot3D([A2[0], B2[0]], [A2[1], B2[1]], [A2[2], B2[2]], 'b-', linewidth=3)[0]
    h['BC1_line'] = ax.plot3D([B1[0], C1[0]], [B1[1], C1[1]], [B1[2], C1[2]], 'r--', linewidth=3)[0]
    h['BC2_line'] = ax.plot3D([B2[0], C2[0]], [B2[1], C2[1]], [B2[2], C2[2]], 'b--', linewidth=3)[0]
    h['CO1_line'] = ax.plot3D([C1[0], O1[0]], [C1[1], O1[1]], [C1[2], O1[2]], 'r-.', linewidth=1.5)[0]
    h['CO2_line'] = ax.plot3D([C2[0], O1[0]], [C2[1], O1[1]], [C2[2], O1[2]], 'b-.', linewidth=1.5)[0]
    h['O1O2_line'] = ax.plot3D([O1[0], O2[0]], [O1[1], O2[1]], [O1[2], O2[2]], 'k-', linewidth=2)[0]

    return h

def add_rotation_axes_animation(ax, O1, O2, roll, pitch, h):
    """复刻Matlab的add_rotation_axes_animation函数：绘制旋转轴"""
    axis_length = 0.05

    # X轴（Roll轴）- 红色
    h['roll_axis'] = ax.quiver(O2[0], O2[1], O2[2], axis_length, 0, 0,
                              color='r', linewidth=2, length=axis_length, arrow_length_ratio=0.5)

    # Pitch轴 - 绿色
    h['pitch_axis'] = ax.quiver(O1[0], O1[1], O1[2], 0, axis_length, 0,
                               color='g', linewidth=2, length=axis_length, arrow_length_ratio=0.5)

    # Y轴 - 绿色
    h['y_axis'] = ax.quiver(O2[0], O2[1], O2[2], 0, axis_length, 0,
                           color='g', linewidth=2, length=axis_length, arrow_length_ratio=0.5)

    # Z轴 - 蓝色
    h['z_axis'] = ax.quiver(O2[0], O2[1], O2[2], 0, 0, axis_length,
                           color='b', linewidth=2, length=axis_length, arrow_length_ratio=0.5)

    # Roll角度标注
    if 'roll_text' in h and h['roll_text'] is not None:
        h['roll_text'].remove()
    h['roll_text'] = ax.text(O2[0]+axis_length/2, O2[1]-0.01, O2[2],
                            f'Roll: {np.rad2deg(roll):.1f}°',
                            color='r', fontweight='bold',
                            bbox=dict(facecolor='white', alpha=1.0))

    return h

def compute_rotated_C(roll, pitch, C0, l_2):
    """复刻Matlab的compute_rotated_C函数：计算旋转后的C点"""
    # 构造4x4变换矩阵
    T_roll = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])

    T_trans = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -l_2],
        [0, 0, 0, 1]
    ])

    T_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])

    T_trans2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, l_2],
        [0, 0, 0, 1]
    ])

    T_total = T_trans2 @ T_pitch @ T_trans @ T_roll  # 矩阵乘法对应Matlab的*
    C0_home = np.array([C0[0], C0[1], C0[2], 1])     # 齐次坐标
    C_final = T_total @ C0_home
    return C_final[:3]  # 取前3维

def rotation_x(theta):
    """复刻Matlab的rotation_x函数：X轴旋转矩阵"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_y(q_pitch):
    """复刻Matlab的rotation_y函数：Y轴旋转矩阵"""
    return np.array([
        [np.cos(q_pitch), 0, np.sin(q_pitch)],
        [0, 1, 0],
        [-np.sin(q_pitch), 0, np.cos(q_pitch)]
    ])

def compute_theta(C1, C2, A1, A2, l_rod1, l_rod2, l_bar):
    """复刻Matlab的compute_theta函数：计算theta角度"""
    # 计算坐标差
    dy1 = C1[1] - A1[1]
    dz1 = C1[2] - A1[2]
    dy2 = C2[1] - A2[1]
    dz2 = C2[2] - A2[2]

    # 计算模长
    norm_CA1 = np.linalg.norm(C1 - A1)
    norm_CA2 = np.linalg.norm(C2 - A2)

    # 计算c1/c2
    c1 = (l_rod1**2 - l_bar**2 - norm_CA1**2) / (-2 * l_bar)
    c2 = (l_rod2**2 - l_bar**2 - norm_CA2**2) / (2 * l_bar)

    # 计算分母（防止除零）
    denom1 = np.sqrt(dy1**2 + dz1**2)
    denom2 = np.sqrt(dy2**2 + dz2**2)

    if denom1 == 0:
        raise ValueError('分母1为零，无法计算角')
    if denom2 == 0:
        raise ValueError('分母2为零，无法计算角')

    # 钳制参数到[-1,1]
    arg1 = c1 / denom1
    arg2 = c2 / denom2
    arg1 = np.clip(arg1, -1, 1)
    arg2 = np.clip(arg2, -1, 1)

    # 计算phi角度
    phi1 = np.arctan2(dy1, dz1)
    phi2 = np.arctan2(dy2, dz2)

    # 计算asin值并生成候选角度
    asin_val1 = np.arcsin(arg1)
    asin_val2 = np.arcsin(arg2)
    theta1_candidate1 = asin_val1 - phi1
    theta1_candidate2 = (np.pi - asin_val1) - phi1
    theta2_candidate1 = asin_val2 - phi2
    theta2_candidate2 = (np.pi - asin_val2) - phi2

    # 选择有效角度
    theta1 = select_valid_angle(theta1_candidate1, theta1_candidate2)
    theta2 = select_valid_angle(theta2_candidate1, theta2_candidate2)

    return theta1, theta2

def select_valid_angle(angle1, angle2):
    """复刻Matlab的select_valid_angle函数：选择有效角度"""
    # 规范化角度到[-pi, pi]
    angle1_norm = normalize_angle(angle1)
    angle2_norm = normalize_angle(angle2)

    # 检查是否在[-pi/2, pi/2]范围内
    is_valid1 = (-np.pi/2 <= angle1_norm <= np.pi/2)
    is_valid2 = (-np.pi/2 <= angle2_norm <= np.pi/2)

    if is_valid1 and is_valid2:
        # 选择绝对值更小的
        if abs(angle1_norm) <= abs(angle2_norm):
            return angle1_norm
        else:
            return angle2_norm
    elif is_valid1:
        return angle1_norm
    elif is_valid2:
        return angle2_norm
    else:
        # 强制映射到[-pi/2, pi/2]
        selected = normalize_angle(angle1_norm)
        if selected > np.pi/2:
            selected -= np.pi
        elif selected < -np.pi/2:
            selected += np.pi
        return selected

def normalize_angle(angle):
    """复刻Matlab的normalize_angle函数：规范化角度到[-pi, pi]"""
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def adjust_angle(angle):
    """复刻Matlab的adjust_angle函数：映射角度到[-pi/2, pi/2]"""
    if angle > np.pi/2:
        angle -= np.pi
    elif angle < -np.pi/2:
        angle += np.pi
    return angle

def plot_mechanism(ax, A1, B1, C1, A2, B2, C2, O1, O2, label):
    """复刻Matlab的plot_mechanism函数：绘制完整机构（带标注）"""
    # 绘制点
    ax.scatter(A1[0], A1[1], A1[2], s=100, c='r', marker='o', edgecolors='none', label='A1 (Motor)')
    ax.scatter(A2[0], A2[1], A2[2], s=100, c='b', marker='o', edgecolors='none', label='A2 (Motor)')
    ax.scatter(B1[0], B1[1], B1[2], s=80, c='r', marker='s', label='B1 (Link)')
    ax.scatter(B2[0], B2[1], B2[2], s=80, c='b', marker='s', label='B2 (Link)')
    ax.scatter(C1[0], C1[1], C1[2], s=80, c='r', marker='^', label='C1 (Output)')
    ax.scatter(C2[0], C2[1], C2[2], s=80, c='b', marker='^', label='C2 (Output)')
    ax.scatter(O1[0], O1[1], O1[2], s=100, c='k', marker='o', edgecolors='none', label='O1 (Ref)')
    ax.scatter(O2[0], O2[1], O2[2], s=100, c='m', marker='o', edgecolors='none', label='O2 (Ref)')

    # 在点上标注符号
    ax.text(A1[0]+0.002, A1[1], A1[2], 'A1', color='r', fontsize=12, fontweight='bold')
    ax.text(A2[0]+0.002, A2[1], A2[2], 'A2', color='b', fontsize=12, fontweight='bold')
    ax.text(B1[0]+0.002, B1[1], B1[2], 'B1', color='r', fontsize=12, fontweight='bold')
    ax.text(B2[0]+0.002, B2[1], B2[2], 'B2', color='b', fontsize=12, fontweight='bold')
    ax.text(C1[0]+0.002, C1[1], C1[2], 'C1', color='r', fontsize=12, fontweight='bold')
    ax.text(C2[0]+0.002, C2[1], C2[2], 'C2', color='b', fontsize=12, fontweight='bold')
    ax.text(O1[0]+0.002, O1[1], O1[2], 'O1', color='k', fontsize=12, fontweight='bold')
    ax.text(O2[0]+0.002, O2[1], O2[2], 'O2', color='m', fontsize=12, fontweight='bold')

    # 绘制连杆
    ax.plot3D([A1[0], B1[0]], [A1[1], B1[1]], [A1[2], B1[2]], 'r-', linewidth=3, label='A-B Link')
    ax.plot3D([A2[0], B2[0]], [A2[1], B2[1]], [A2[2], B2[2]], 'b-', linewidth=3, label='A-B Link')

    # 绘制推杆
    ax.plot3D([B1[0], C1[0]], [B1[1], C1[1]], [B1[2], C1[2]], 'r--', linewidth=3, label='B-C Rod')
    ax.plot3D([B2[0], C2[0]], [B2[1], C2[1]], [B2[2], C2[2]], 'b--', linewidth=3, label='B-C Rod')

    # 绘制C-O连线
    ax.plot3D([C1[0], O1[0]], [C1[1], O1[1]], [C1[2], O1[2]], 'r-.', linewidth=1.5, label='C-O')
    ax.plot3D([C2[0], O1[0]], [C2[1], O1[1]], [C2[2], O1[2]], 'b-.', linewidth=1.5, label='C-O')
    ax.plot3D([C1[0], O2[0]], [C1[1], O2[1]], [C1[2], O2[2]], 'r-.', linewidth=1.5, label='C-O')
    ax.plot3D([C2[0], O2[0]], [C2[1], O2[1]], [C2[2], O2[2]], 'b-.', linewidth=1.5, label='C-O')

    # 绘制O1-O2参考线
    ax.plot3D([O1[0], O2[0]], [O1[1], O2[1]], [O1[2], O2[2]], 'k-', linewidth=2, label='O1-O2')

def add_rotation_axes(ax, O1, O2, roll, pitch):
    """复刻Matlab的add_rotation_axes函数：绘制旋转轴（带标注）"""
    axis_length = 0.05

    # X轴（Roll轴）
    ax.quiver(O2[0], O2[1], O2[2], axis_length, 0, 0, color='r', linewidth=2,
              length=axis_length, arrow_length_ratio=0.5, label='Roll Axis (X)')

    # Pitch轴
    ax.quiver(O1[0], O1[1], O1[2], 0, axis_length, 0, color='g', linewidth=2,
              length=axis_length, arrow_length_ratio=0.5, label='Pitch Axis')

    # Y轴
    ax.quiver(O2[0], O2[1], O2[2], 0, axis_length, 0, color='g', linewidth=2,
              length=axis_length, arrow_length_ratio=0.5, label='Y Axis')

    # Z轴
    ax.quiver(O2[0], O2[1], O2[2], 0, 0, axis_length, color='b', linewidth=2,
              length=axis_length, arrow_length_ratio=0.5, label='Z Axis')

    # 轴末端标注
    ax.text(O1[0], O1[1]+axis_length+0.005, O1[3] if len(O1)>3 else O1[2], 'pitch',
            color='g', fontsize=12, fontweight='bold')
    ax.text(O2[0]+axis_length+0.005, O2[1], O2[2], 'X',
            color='r', fontsize=12, fontweight='bold')
    ax.text(O2[0], O2[1]+axis_length+0.005, O2[2], 'Y',
            color='g', fontsize=12, fontweight='bold')
    ax.text(O2[0], O2[1], O2[2]+axis_length+0.005, 'Z',
            color='b', fontsize=12, fontweight='bold')

    # Roll角度标注
    ax.text(O2[0]+axis_length/2, O2[1]-0.01, O2[2], f'Roll: {np.rad2deg(roll):.1f}°',
            color='r', fontweight='bold', bbox=dict(facecolor='white', alpha=1.0))

# 主函数调用
if __name__ == '__main__':
    plot_ankle_mechanism_animation()
    plt.ioff()
    plt.show()
