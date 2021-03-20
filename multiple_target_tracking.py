# -*- coding: UTF-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

from get_ellipse import get_ellipse
from kalman_filter import KalmanFilter4D
from obj import Object

def control_target(tar, ti, sigma_ax, sigma_ay):
    # 功能：控制目标随机移动
    # 输入：tar <class 'numpy.ndarray'> (4, 1)
    # 输出：tar <class 'numpy.ndarray'> (4, 1)
    
    v = np.zeros((2, 1))
    v[0, 0] = sigma_ax * float(np.random.randn(1))
    v[1, 0] = sigma_ay * float(np.random.randn(1))
    
    f = np.array([[1, ti, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, ti],
                  [0, 0, 0, 1],])
    g = np.array([[0.5 * ti ** 2, 0],
                  [ti, 0],
                  [0, 0.5 * ti ** 2],
                  [0, ti],])
    tar_new = f @ tar + g @ v
    
    return tar_new

def observe(tar, arange, sigma_ox, sigma_oy):
    # 功能：量测目标
    # 输入：tar <class 'numpy.ndarray'> (4, 1)
    #      arange <class 'float'> 量测范围
    # 输出：x <class 'float'>
    #      y <class 'float'>
    #      flag <class 'bool'> 是否为有效量测
    
    h = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],])
    
    flag = False
    if tar[0, 0] ** 2 + tar[2, 0] ** 2 <= arange ** 2:
        w = np.zeros((2, 1))
        w[0, 0] = sigma_ox * float(np.random.randn(1))
        w[1, 0] = sigma_oy * float(np.random.randn(1))
        
        z = h @ tar
        x = z[0, 0] + w[0, 0]
        y = z[1, 0] + w[1, 0]
        flag = True
    else:
        x, y = None, None
        flag = False
    
    return x, y, flag

ni = 500               # 仿真迭代次数
ti = 0.1               # 时间间隔
nt = 30                # 目标数量
gate_threshold = 4000  # 跟踪门阈值
blind_update_limit = 5 # 中断更新的次数限制

xtrue = [0, 0]         # 自车位置
arange = 75            # 量测范围

sigma_ax = 1             # 过程噪声标准差
sigma_ay = 1             # 过程噪声标准差
sigma_ox = 0.1           # 量测噪声标准差
sigma_oy = 0.1           # 量测噪声标准差

# Create a drawing window.
fig = plt.figure(1)

filename = 'result.txt'
with open(filename, 'w') as fob:
    fob.seek(0)
    fob.truncate()

# 初始化跟踪列表
objs = []
number = 0

# 初始化临时跟踪列表
objs_temp = []

# 初始化目标位置和速度
targets = np.zeros((4 * nt, ni))
for t in range(nt):
    targets[4 * t + 0, 0] = 40 * np.random.randn(1)
    targets[4 * t + 1, 0] = 1 * np.random.randn(1)
    targets[4 * t + 2, 0] = 1500 * np.random.rand(1) + 100
    targets[4 * t + 3, 0] = 5 * np.random.randn(1) - 20

for i in range(1, ni):
    print()
    print("\nIteration:", i)
    
    # 控制目标随机移动
    for j in range(nt):
        tar = np.array([[targets[4 * j + 0, i - 1]],
                        [targets[4 * j + 1, i - 1]],
                        [targets[4 * j + 2, i - 1]],
                        [targets[4 * j + 3, i - 1]]])
        tar = control_target(tar, ti, sigma_ax, sigma_ay)
        targets[4 * j + 0, i] = tar[0, 0]
        targets[4 * j + 1, i] = tar[1, 0]
        targets[4 * j + 2, i] = tar[2, 0]
        targets[4 * j + 3, i] = tar[3, 0]
    
    # 初始化量测列表
    objs_observed = []
    
    # 获取量测
    for j in range(nt):
        tar = np.array([[targets[4 * j + 0, i]],
                        [targets[4 * j + 1, i]],
                        [targets[4 * j + 2, i]],
                        [targets[4 * j + 3, i]]])
        x, y, flag = observe(tar, arange, sigma_ox, sigma_oy)
        if flag:
            obj = Object()
            obj.xref = x
            obj.yref = y
            objs_observed.append(obj)
    objs_observed_copy = objs_observed.copy()
    
    # 数据关联与跟踪
    num = len(objs)
    for j in range(num):
        flag = False
        idx = 0
        ddm = float('inf')
        
        n = len(objs_observed)
        for k in range(n):
            zx = objs_observed[k].xref
            zy = objs_observed[k].yref
            dd = objs[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < gate_threshold:
                idx = k
                ddm = dd
                flag = True
        
        if flag:
            zx = objs_observed[idx].xref
            zy = objs_observed[idx].yref
            objs[j].tracker.predict()
            objs[j].tracker.update(zx, zy)
            objs[j].tracker_blind_update = 0
            objs_observed.pop(idx)
        else:
            objs[j].tracker.predict()
            objs[j].tracker_blind_update += 1
    
    # 删除长时间未跟踪的目标
    objs_remained = []
    num = len(objs)
    for j in range(num):
        if objs[j].tracker_blind_update <= blind_update_limit:
            objs_remained.append(objs[j])
    objs = objs_remained
    
    # 增广跟踪列表
    num = len(objs_temp)
    for j in range(num):
        flag = False
        idx = 0
        ddm = float('inf')
        
        n = len(objs_observed)
        for k in range(n):
            zx = objs_observed[k].xref
            zy = objs_observed[k].yref
            dd = objs_temp[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < gate_threshold:
                idx = k
                ddm = dd
                flag = True
        
        if flag:
            zx = objs_observed[idx].xref
            zy = objs_observed[idx].yref
            x = objs_temp[j].tracker.xx[0, 0]
            y = objs_temp[j].tracker.xx[2, 0]
            
            objs_temp[j].tracker.xx[0, 0] = zx
            objs_temp[j].tracker.xx[1, 0] = (zx - x) / objs_temp[j].tracker.ti
            objs_temp[j].tracker.xx[2, 0] = zy
            objs_temp[j].tracker.xx[3, 0] = (zy - y) / objs_temp[j].tracker.ti
            
            objs_observed.pop(idx)
            number += 1
            objs_temp[j].number = number
            objs.append(objs_temp[j])
    
    # 增广临时跟踪列表
    objs_temp = objs_observed
    num = len(objs_temp)
    for j in range(num):
        objs_temp[j].tracker = KalmanFilter4D(ti, objs_temp[j].xref, objs_temp[j].vx, objs_temp[j].yref, objs_temp[j].vy, sigma_ax, sigma_ay, sigma_ox, sigma_oy)
    
    # 终端输出跟踪列表
    num = len(objs)
    for j in range(num):
        print()
        print('ID')
        print(objs[j].number)
        print('xx')
        print(objs[j].tracker.xx)
        print('pp')
        print(objs[j].tracker.pp)
    
    # 以txt保存结果
    filename = 'result.txt'
    num = len(objs)
    for j in range(num):
        with open(filename, 'a') as fob:
            fob.write('frame:%d id:%d x:%.3f vx:%.3f y:%.3f vy:%.3f' % (i, objs[j].number, objs[j].tracker.xx[0, 0], objs[j].tracker.xx[1, 0], objs[j].tracker.xx[2, 0], objs[j].tracker.xx[3, 0]))
            fob.write('\n')
    
    # Draw dynamically.
    plt.clf()
    
    # Draw ego vehicle.
    plt.scatter(xtrue[0], xtrue[1], c='blue', edgecolor='none', s=100)
    
    # Draw range of observation.
    theta = np.linspace(0, 2 * np.pi, 1000)
    arange_x = arange * np.cos(theta)
    arange_y = arange * np.sin(theta)
    plt.plot(arange_x, arange_y, '--', c='black', linewidth=1)
    
    # Draw real targets.
    for j in range(nt):
        x = targets[4 * j, i]
        vx = targets[4 * j + 1, i]
        y = targets[4 * j + 2, i]
        vy = targets[4 * j + 3, i]
        if math.fabs(x) < 100 and math.fabs(y) < 100:
            plt.scatter(x, y, c='black', edgecolor='none', s=25)
            text_real_ve = "Velocity: (" + str(round(vx, 1)) + "," + str(round(vy, 1)) + ")"
            plt.text(x + 2, y - 10, text_real_ve, color='lightgrey', fontsize=12)
    
    # Draw observation.
    num = len(objs_observed_copy)
    for j in range(num):
        x = objs_observed_copy[j].xref
        y = objs_observed_copy[j].yref
        ob_x = [xtrue[0], x]
        ob_y = [xtrue[1], y]
        plt.plot(ob_x, ob_y, c='black', linewidth=1)
    
    # Draw estimated targets and association gate.
    num = len(objs)
    for j in range(num):
        x = objs[j].tracker.xx[0, 0]
        vx = objs[j].tracker.xx[1, 0]
        y = objs[j].tracker.xx[2, 0]
        vy = objs[j].tracker.xx[3, 0]
        plt.scatter(x, y, c='red', edgecolor='none', s=25)
        
        a, b = objs[j].tracker.compute_association_gate(gate_threshold)
        xs, ys = get_ellipse(x, y, a, b, 0)
        plt.plot(xs, ys, '--', c='red', linewidth=1)
        
        text_id = "ID: " + str(objs[j].number)
        text_lo = "Location: (" + str(round(x, 1)) + "," + str(round(y, 1)) + ")"
        text_ve = "Velocity: (" + str(round(vx, 1)) + "," + str(round(vy, 1)) + ")"
        plt.text(x + 2, y + 8, text_id, fontsize=12)
        plt.text(x + 2, y + 2, text_lo, fontsize=12)
        plt.text(x + 2, y - 4, text_ve, fontsize=12)
    
    # Draw legend.
    plt.scatter(46, 90, c='black', edgecolor='none', s=25)
    plt.text(50, 88, "real targets", fontsize=12)
    plt.scatter(46, 80, c='red', edgecolor='none', s=25)
    plt.text(50, 78, "estimated targets", fontsize=12)
    
    # Close the drawing window after showing for a while.
    plt.axis([-100, 100, -100, 100])
    plt.xlabel('Meter', fontsize=14)
    plt.ylabel('Meter', fontsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(0.02)
    
    if i == ni - 1:
        print("\nSimulation process finished!")
        break
