# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:10:48 2025

@author: WatabeT
"""

import numpy as np
from statsmodels.stats.power import FTestAnovaPower

# 必要なパラメータを入力
mean_control = 100  # コントロール群の平均
# mean_treatment = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61]  # 各薬剤群の平均
mean_treatment = [100, 50, 101, 101, 101, 101,101, 101, 101, 101]  # 各薬剤群の平均
# mean_treatment = [50]  # 各薬剤群の平均
std_dev = 100  # 標準偏差（すべての群で同じと仮定）
alpha = 0.05  # 有意水準
power_target = 0.8  # 検出力目標（80%）

# 効果サイズ（eta-squared）を計算
group_means = [mean_control] + mean_treatment
overall_mean = np.mean(group_means)
ss_between = sum(len(group_means) * (mean - overall_mean) ** 2 for mean in group_means)
effect_size = ss_between / (ss_between + std_dev ** 2)

# 必要なサンプルサイズを計算
anova_power = FTestAnovaPower()
num_groups = len(group_means)
sample_size = anova_power.solve_power(effect_size=effect_size, alpha=alpha, power=power_target, k_groups=num_groups)

print(f"目標の検出力を達成するために必要なサンプルサイズ: 各群 {np.ceil(sample_size):.0f} 件")