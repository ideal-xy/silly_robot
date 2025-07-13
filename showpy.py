import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import heapq

#设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以便结果可重现
np.random.seed(int(time.time()))

M = 22  # 地图高度
N = 18  # 地图宽度

# 初始化固定障碍矩阵（墙、课桌、讲台等）
fixed_obstacle = np.ones((M, N))  # 1表示可通行 ✅
fixed_obstacle[[0, M-1], :] = np.inf  # 墙不可以通行 ❌
fixed_obstacle[:, [0, N-1]] = np.inf
for i in range(5, 21, 2):  
    for j in range(3, 15, 4):
        fixed_obstacle[i, j:j+2] = np.inf  # 课桌
        
fixed_obstacle[6:10, 2:3] = np.inf  # 讲台
fixed_obstacle[3, 0] = 1  # 门是可通行的

# 随机选择扫地机器人的初始位置
valid_positions = [(i, j) for i in range(1, M-1) for j in range(1, N-1) 
                  if fixed_obstacle[i, j] != np.inf and (i, j) != (3, 0)]
rx, ry = valid_positions[np.random.randint(len(valid_positions))]
robot_pos = (rx, ry)

# 初始化多个人的位置
num_people = 3
human_positions = []
human_paths = []
human_directions_list = []

for _ in range(num_people):
    valid_human_pos = [
    pos for pos in valid_positions 
    if (fixed_obstacle[pos] != np.inf  # 新增障碍物检查
        and pos != robot_pos
        and pos != (3, 0)
        and pos not in human_positions)
]
    hx, hy = valid_human_pos[np.random.randint(len(valid_human_pos))]
    human_positions.append((hx, hy))
    human_paths.append([(hx, hy)])
    human_directions_list.append(np.random.randint(4))

# A*算法辅助函数
def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def a_star(start, goal, cost_map):
    # 初始化优先队列
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # 记录路径和代价
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # 重构路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        # 遍历所有相邻单元格
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # 检查是否为有效位置
            if (0 <= neighbor[0] < M and 0 <= neighbor[1] < N and 
                cost_map[neighbor[0], neighbor[1]] != np.inf):
                
                tentative_g = g_score[current] + 1  # 移动代价为1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    
    # 如果没有找到路径
    return []

# 初始化颜色
R = np.ones((M, N))
R[[0, M-1], :] = 0
R[:, [0, N-1]] = 0
R[3, 0] = 0
for i in range(5, 21, 2):  
    for j in range(3, 15, 4):
        R[i, j:j+2] = 0
R[2:3, 6:10] = 0

G = R.copy()
B = R.copy()
G[3, 0] = 1

# 标记扫地机器人位置
R[rx, ry] = 1
G[rx, ry] = 0
B[rx, ry] = 0

# 标记人的位置
for pos in human_positions:
    R[pos] = 0
    G[pos] = 0
    B[pos] = 1

# 路径规划
T = 300  # 总帧数
img_array = np.empty((M, N, 3, T))
effective_frames = T

# 定义人的移动方向
human_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上

# 初始化动态障碍矩阵（初始包含固定障碍和人的位置）
dynamic_obstacle = fixed_obstacle.copy()
for pos in human_positions:
    dynamic_obstacle[pos] = np.inf

# 记录扫地机器人路径
robot_path = [robot_pos]
robot_visited = set([robot_pos])

# 记录性能指标
total_path_length = 0
obstacle_avoidance_count = 0
reach_goal_time = None

# 创建日志文件
with open("path_log.txt", "w") as log_file:
    log_file.write("Robot Path Log:\n")

# 主循环
for t in range(T):
    if robot_pos == (3, 0):
        effective_frames = t
        reach_goal_time = t
        break
    
    # 更新多个人的位置
    for p_idx in range(num_people):
        hx, hy = human_positions[p_idx]
        current_direction = human_directions_list[p_idx]
        
        # 人有70%概率继续当前方向，30%概率随机改变方向
        if np.random.random() < 0.3:
            current_direction = np.random.randint(4)
        
        dhx, dhy = human_directions[current_direction]
        new_hx, new_hy = hx + dhx, hy + dhy
        
        # 检查新位置是否有效
        valid_move = False
        if (1 <= new_hx < M-1 and 1 <= new_hy < N-1 and 
            fixed_obstacle[new_hx, new_hy] != np.inf and
            (new_hx, new_hy) != robot_pos and 
            (new_hx, new_hy) != (3, 0) and
            (new_hx, new_hy) not in human_positions):
            valid_move = True
        
        # 如果新位置无效，尝试其他方向
        if not valid_move:
            possible_directions = []
            for dir_idx, (dx, dy) in enumerate(human_directions):
                nx, ny = hx + dx, hy + dy
                if (1 <= nx < M-1 and 1 <= ny < N-1 and 
                    fixed_obstacle[nx, ny] != np.inf and
                    (nx, ny) != robot_pos and 
                    (nx, ny) != (3, 0) and
                    (nx, ny) not in human_positions):
                    possible_directions.append(dir_idx)
            
            if possible_directions:
                current_direction = possible_directions[np.random.randint(len(possible_directions))]
                dhx, dhy = human_directions[current_direction]
                new_hx, new_hy = hx + dhx, hy + dhy
                valid_move = True
        
        # 如果找到了有效移动，更新人的位置
        if valid_move:
            # 更新动态障碍矩阵
            dynamic_obstacle[hx, hy] = fixed_obstacle[hx, hy]  # 恢复旧位置
            dynamic_obstacle[new_hx, new_hy] = np.inf          # 设置新位置为障碍
            
            # 更新位置
            hx, hy = new_hx, new_hy
            human_positions[p_idx] = (hx, hy)
            human_paths[p_idx].append((hx, hy))
            human_directions_list[p_idx] = current_direction
    
    # 使用A*算法重新规划路径
    start = robot_pos
    goal = (3, 0)
    
    # 创建临时障碍地图（固定障碍+动态障碍）
    temp_obstacle = dynamic_obstacle.copy()
    
    # 规划路径
    path = a_star(start, goal, temp_obstacle)
    
    # 如果有有效路径，沿着路径移动一步
    if path and robot_pos != goal:
        next_pos = path[1] if len(path) > 1 else path[0]
        robot_pos = next_pos
        robot_path.append(robot_pos)
        robot_visited.add(robot_pos)
    
    # 更新颜色
    R = np.ones((M, N))
    R[[0, M-1], :] = 0
    R[:, [0, N-1]] = 0
    R[3, 0] = 0
    for i in range(5, 21, 2):  
        for j in range(3, 15, 4):
            R[i, j:j+2] = 0
    R[2:3, 6:10] = 0

    G = R.copy()
    B = R.copy()
    G[3, 0] = 1
    
    # 标记扫地机器人路径（浅红色）
    for pos in robot_path:
        if fixed_obstacle[pos] != np.inf and pos not in human_positions and pos != (3, 0):
            R[pos] = 1
            G[pos] = 0.5
            B[pos] = 0.5
    
    # # 标记人的路径（浅蓝色）
    # for path in human_paths:
    #     for pos in path:
    #         if fixed_obstacle[pos] != np.inf and pos != robot_pos and pos != (3, 0):
    #             R[pos] = 0.5
    #             G[pos] = 0.5
    #             B[pos] = 1
    
    # 标记扫地机器人当前位置（亮红色）
    R[robot_pos] = 1
    G[robot_pos] = 0
    B[robot_pos] = 0
    
    # 标记人的当前位置（亮蓝色）
    for pos in human_positions:
        R[pos] = 0
        G[pos] = 0
        B[pos] = 1
    
    # 标记门（绿色）
    R[3, 0] = 0
    G[3, 0] = 1
    B[3, 0] = 0
    
    # 保存帧
    img_array[:, :, :, t] = np.stack((R, G, B), axis=-1)
    
    # 记录性能指标
    total_path_length += 1
    if len(robot_path) > 1 and robot_path[-1] != robot_path[-2]:
        obstacle_avoidance_count += 1
    
    
    with open("path_log.txt", "a") as log_file:
        log_file.write(f"Step {t+1}: Robot at {robot_pos}, Humans at {human_positions}\n")

# 生成动画
fig, ax = plt.subplots(figsize=(12, 10))
img = ax.imshow(img_array[:, :, :, 0])

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Cleaning robot'),
    Patch(facecolor='blue', label='human'),
    Patch(facecolor='pink', label='robot trace'),
    Patch(facecolor='lightblue', label='human trace'),
    Patch(facecolor='green', label='door'),
    Patch(facecolor='black', label='barrier')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

ax.set_title('Cleaning Robot\'s Random Walk', fontsize=16)
ax.set_xticks([])
ax.set_yticks([])

def update(t):
    img.set_data(img_array[:, :, :, tx])
    ax.set_title(f'Cleaning Robot\'s Random Walk (Steps: {t+1})', fontsize=16)
    return img

ani = animation.FuncAnimation(
    fig,
    update,
    frames=effective_frames,
    interval=200
)
# 直接显示动画
plt.tight_layout()
plt.show()

# 保存动画
save_path = r"homework.gif"
ani.save(save_path, writer='pillow', fps=5)

# 显示性能指标
print(f"动画已保存至: {save_path}")
print(f"总步数: {effective_frames}")
print(f"机器人起始位置: ({rx}, {ry})")
print(f"人的起始位置: {human_positions}")
print(f"总路径长度: {total_path_length}")
print(f"避障次数: {obstacle_avoidance_count}")
if reach_goal_time is not None:
    print(f"到达目标时间: {reach_goal_time}")