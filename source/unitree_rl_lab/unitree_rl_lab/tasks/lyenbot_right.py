import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 启用交互模式，确保3D绘图实时刷新
plt.ion()

def plot_ankle_mechanism_animation():
    # ===================== 参数定义（与Matlab完全一致）=====================
    l_axis = 0.03685    # A点轴长
    l_axis_C = 0.033    # C点轴长
    l_bar = 0.0225      # 杆长
    l_bar_c = 0.0185    # 杆长
    l_rod1 = 0.1455     # 推杆1长度
    l_rod2 = 0.2105     # 推杆2长度
    l_2 = 0.0           # 踝关节长度
    lz = 0.01           # 控制面和十字中心的高度

    # ===================== 初始位置计算 =====================
    C1_0 = np.array([-l_axis_C, l_bar_c, -lz])
    C2_0 = np.array([-l_axis_C, -l_bar_c, -lz])
    A1_0 = np.array([-l_axis, 0, 0.135])
    A2_0 = np.array([-l_axis, 0, 0.200])
    B1_0 = A1_0 + np.array([0, l_bar, 0])
    B2_0 = A2_0 + np.array([0, -l_bar, 0])
    O1 = np.array([0.0, 0.0, 0.0])
    O2 = np.array([0.0, 0.0, 0.0])

    # ===================== 创建3D图形 =====================
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Ankle Joint Mechanism Animation - Roll and Pitch from -10° to 10°', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.05, 0.25)
    # 调整视角（更易观察）
    ax.view_init(elev=20, azim=-60)

    # ===================== 动画参数 =====================
    num_frames = 50
    # 动态角度范围，方便观察效果
    roll_range = np.linspace(-10, 10, num_frames)
    pitch_range = np.linspace(-10, 10, num_frames)

    # ===================== 预计算最大/最小theta角度 ======================
    # 最大值
    roll = np.deg2rad(roll_range[-1])
    pitch = np.deg2rad(pitch_range[-1])
    C1_rot = compute_rotated_C(roll, pitch, C1_0, l_2)
    C2_rot = compute_rotated_C(roll, pitch, C2_0, l_2)
    theta1, theta2 = compute_theta(C1_rot, C2_rot, A1_0, A2_0, l_rod1, l_rod2, l_bar)
    print(f'最大值θ1 = {np.rad2deg(theta1):.2f} deg, θ2 = {np.rad2deg(theta2):.2f} deg')

    # 最小值
    roll = np.deg2rad(roll_range[0])
    pitch = np.deg2rad(pitch_range[0])
    C1_rot = compute_rotated_C(roll, pitch, C1_0, l_2)
    C2_rot = compute_rotated_C(roll, pitch, C2_0, l_2)
    theta1, theta2 = compute_theta(C1_rot, C2_rot, A1_0, A2_0, l_rod1, l_rod2, l_bar)
    print(f'最小值θ1 = {np.rad2deg(theta1):.2f} deg, θ2 = {np.rad2deg(theta2):.2f} deg')

    # ===================== 初始化图形对象 ======================
    # 存储所有创建的图形对象，用于后续删除
    scatter_objects = []
    line_objects = []
    quiver_objects = []
    text_objects = []

    # ===================== 动画循环 =====================
    try:
        while True:
            for i in range(num_frames):
                # 1. 清空上一帧所有图形元素（修复版本兼容性问题）
                # 删除散点 - 兼容所有Matplotlib版本
                for scatter in scatter_objects:
                    try:
                        scatter.remove()  # 直接删除对象本身
                    except:
                        pass
                scatter_objects = []

                # 删除连线
                for line in line_objects:
                    try:
                        line.remove()
                    except:
                        pass
                line_objects = []

                # 删除箭头
                for quiver in quiver_objects:
                    try:
                        quiver.remove()
                    except:
                        pass
                quiver_objects = []

                # 删除文本
                for text in text_objects:
                    try:
                        text.remove()
                    except:
                        pass
                text_objects = []

                # 2. 计算当前帧参数
                roll = np.deg2rad(roll_range[i])
                pitch = np.deg2rad(pitch_range[i])

                # 3. 计算旋转后的点
                C1_rot = compute_rotated_C(roll, pitch, C1_0, l_2)
                C2_rot = compute_rotated_C(roll, pitch, C2_0, l_2)

                # 4. 计算theta角度
                theta1, theta2 = compute_theta(C1_rot, C2_rot, A1_0, A2_0, l_rod1, l_rod2, l_bar)

                # 5. 计算旋转后的B点
                B1_rot = A1_0 + rotation_x(theta1) @ (B1_0 - A1_0)
                B2_rot = A2_0 + rotation_x(theta2) @ (B2_0 - A2_0)

                # 6. 绘制机构点（散点）
                # A1(红圆)、A2(蓝圆)
                s1 = ax.scatter(A1_0[0], A1_0[1], A1_0[2], s=100, c='r', marker='o', edgecolors='none')
                s2 = ax.scatter(A2_0[0], A2_0[1], A2_0[2], s=100, c='b', marker='o', edgecolors='none')
                # B1(红方)、B2(蓝方)
                s3 = ax.scatter(B1_rot[0], B1_rot[1], B1_rot[2], s=80, c='r', marker='s')
                s4 = ax.scatter(B2_rot[0], B2_rot[1], B2_rot[2], s=80, c='b', marker='s')
                # C1(红三角)、C2(蓝三角)
                s5 = ax.scatter(C1_rot[0], C1_rot[1], C1_rot[2], s=80, c='r', marker='^')
                s6 = ax.scatter(C2_rot[0], C2_rot[1], C2_rot[2], s=80, c='b', marker='^')
                # O1(黑圆)、O2(紫圆)
                s7 = ax.scatter(O1[0], O1[1], O1[2], s=100, c='k', marker='o', edgecolors='none')
                s8 = ax.scatter(O2[0], O2[1], O2[2], s=100, c='m', marker='o', edgecolors='none')

                # 将散点对象加入列表
                scatter_objects.extend([s1, s2, s3, s4, s5, s6, s7, s8])

                # 7. 绘制连杆（连线）
                # A-B 实线
                l1 = ax.plot3D([A1_0[0], B1_rot[0]], [A1_0[1], B1_rot[1]], [A1_0[2], B1_rot[2]], 'r-', linewidth=3)[0]
                l2 = ax.plot3D([A2_0[0], B2_rot[0]], [A2_0[1], B2_rot[1]], [A2_0[2], B2_rot[2]], 'b-', linewidth=3)[0]
                # B-C 虚线
                l3 = ax.plot3D([B1_rot[0], C1_rot[0]], [B1_rot[1], C1_rot[1]], [B1_rot[2], C1_rot[2]], 'r--', linewidth=3)[0]
                l4 = ax.plot3D([B2_rot[0], C2_rot[0]], [B2_rot[1], C2_rot[1]], [B2_rot[2], C2_rot[2]], 'b--', linewidth=3)[0]
                # C-O 点划线
                l5 = ax.plot3D([C1_rot[0], O1[0]], [C1_rot[1], O1[1]], [C1_rot[2], O1[2]], 'r-.', linewidth=1.5)[0]
                l6 = ax.plot3D([C2_rot[0], O1[0]], [C2_rot[1], O1[1]], [C2_rot[2], O1[2]], 'b-.', linewidth=1.5)[0]
                # O1-O2 黑线
                l7 = ax.plot3D([O1[0], O2[0]], [O1[1], O2[1]], [O1[2], O2[2]], 'k-', linewidth=2)[0]

                # 将连线对象加入列表
                line_objects.extend([l1, l2, l3, l4, l5, l6, l7])

                # 8. 绘制旋转轴（箭头）
                axis_length = 0.05
                # X轴（Roll轴）- 红色
                q1 = ax.quiver(O2[0], O2[1], O2[2], axis_length, 0, 0, color='r', linewidth=2,
                              length=axis_length, arrow_length_ratio=0.3, normalize=True)
                # Y轴（Pitch轴）- 绿色
                q2 = ax.quiver(O1[0], O1[1], O1[2], 0, axis_length, 0, color='g', linewidth=2,
                              length=axis_length, arrow_length_ratio=0.3, normalize=True)
                # Z轴 - 蓝色
                q3 = ax.quiver(O2[0], O2[1], O2[2], 0, 0, axis_length, color='b', linewidth=2,
                              length=axis_length, arrow_length_ratio=0.3, normalize=True)

                # 将箭头对象加入列表
                quiver_objects.extend([q1, q2, q3])

                # 9. 添加角度信息文本
                info_str = (f'Roll: {np.rad2deg(roll):.1f}°\nPitch: {np.rad2deg(pitch):.1f}°\n'
                            f'Motor1 θ1: {np.rad2deg(theta1):.2f}°\nMotor2 θ2: {np.rad2deg(theta2):.2f}°')
                t1 = ax.text(-0.08, -0.08, 0.2, info_str, fontsize=10, fontweight='bold',
                              bbox=dict(facecolor='white', alpha=0.8))
                # Roll角度标注
                t2 = ax.text(O2[0]+axis_length/2, O2[1]-0.01, O2[2], f'Roll: {np.rad2deg(roll):.1f}°',
                              color='r', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

                # 将文本对象加入列表
                text_objects.extend([t1, t2])

                # 10. 强制刷新图形
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                time.sleep(0.08)  # 稍慢一点，方便观察

    except KeyboardInterrupt:
        # 按Ctrl+C退出循环
        plt.close(fig)
        print("动画已停止")

# ===================== 核心子函数（保持不变）=====================
def compute_rotated_C(roll, pitch, C0, l_2):
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
    T_total = T_trans2 @ T_pitch @ T_trans @ T_roll
    C0_home = np.array([C0[0], C0[1], C0[2], 1])
    C_final = T_total @ C0_home
    return C_final[:3]

def rotation_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def compute_theta(C1, C2, A1, A2, l_rod1, l_rod2, l_bar):
    dy1 = C1[1] - A1[1]
    dz1 = C1[2] - A1[2]
    dy2 = C2[1] - A2[1]
    dz2 = C2[2] - A2[2]

    norm_CA1 = np.linalg.norm(C1 - A1)
    norm_CA2 = np.linalg.norm(C2 - A2)

    c1 = (l_rod1**2 - l_bar**2 - norm_CA1**2) / (-2 * l_bar)
    c2 = (l_rod2**2 - l_bar**2 - norm_CA2**2) / (2 * l_bar)

    denom1 = np.sqrt(dy1**2 + dz1**2)
    denom2 = np.sqrt(dy2**2 + dz2**2)
    if denom1 == 0:
        raise ValueError('分母1为零，无法计算角')
    if denom2 == 0:
        raise ValueError('分母2为零，无法计算角')

    arg1 = c1 / denom1
    arg2 = c2 / denom2
    arg1 = np.clip(arg1, -1, 1)
    arg2 = np.clip(arg2, -1, 1)

    phi1 = np.arctan2(dy1, dz1)
    phi2 = np.arctan2(dy2, dz2)

    asin_val1 = np.arcsin(arg1)
    asin_val2 = np.arcsin(arg2)
    theta1_candidate1 = asin_val1 - phi1
    theta1_candidate2 = (np.pi - asin_val1) - phi1
    theta2_candidate1 = asin_val2 - phi2
    theta2_candidate2 = (np.pi - asin_val2) - phi2

    theta1 = select_valid_angle(theta1_candidate1, theta1_candidate2)
    theta2 = select_valid_angle(theta2_candidate1, theta2_candidate2)

    return theta1, theta2

def select_valid_angle(angle1, angle2):
    angle1_norm = normalize_angle(angle1)
    angle2_norm = normalize_angle(angle2)
    is_valid1 = (-np.pi/2 <= angle1_norm <= np.pi/2)
    is_valid2 = (-np.pi/2 <= angle2_norm <= np.pi/2)

    if is_valid1 and is_valid2:
        if abs(angle1_norm) <= abs(angle2_norm):
            return angle1_norm
        else:
            return angle2_norm
    elif is_valid1:
        return angle1_norm
    elif is_valid2:
        return angle2_norm
    else:
        selected = normalize_angle(angle1_norm)
        if selected > np.pi/2:
            selected -= np.pi
        elif selected < -np.pi/2:
            selected += np.pi
        return selected

def normalize_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

# 主函数调用
if __name__ == '__main__':
    plot_ankle_mechanism_animation()
    plt.ioff()  # 关闭交互模式
    plt.show()
