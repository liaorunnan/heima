import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义 x 轴范围（注意：log(x) 的定义域是 x > 0）
x = np.linspace(0.01, 10, 400)

# 计算 y = log(x)，这里使用自然对数 ln(x)
y_log = np.log(x)

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(x, y_log, label=r'$y = \log(x)$ (自然对数)', color='blue', linewidth=2)

# 添加一些参考线和标注
plt.axhline(0, color='black', linestyle='--', alpha=0.3, label='y=0')
plt.axvline(1, color='red', linestyle='--', alpha=0.3, label='x=1')

# 标注关键点
plt.plot(1, 0, 'ro', markersize=8, label=f'点(1, 0): log(1)=0')
plt.plot(np.e, 1, 'go', markersize=8, label=f'点(e, 1): log(e)=1')

# 设置坐标轴标签和标题
plt.xlabel('x')
plt.ylabel('y')
plt.title('对数函数 $y = \log(x)$ 的图像')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()

print("对数函数 y = log(x) 的特点：")
print("- 定义域：x > 0")
print("- 经过点 (1, 0)")
print("- 当 x 增大时，y 缓慢增大")
print("- 当 x 接近 0 时，y 趋向负无穷")
print("- 函数是单调递增的凹函数")