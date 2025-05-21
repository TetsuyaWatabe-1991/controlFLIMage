# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:37:03 2024

@author: yasudalab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:15:34 2024

@author: watabetetsuya
"""

import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def total_distance(points, order):
    n = len(order)
    total = 0
    for i in range(n - 1):
        total += calculate_distance(points[order[i]], points[order[i + 1]])
    total += calculate_distance(points[order[-1]], points[order[0]])  # 最後の点から最初の点への距離を加算
    return total

def nearest_neighbor(points):
    #not the best, but very good enough.
    n = len(points)
    visited = [False] * n
    order = [0]
    current_index = 0
    visited[current_index] = True

    for _ in range(n - 1):
        next_index = None
        min_distance = float('inf')
        current_point = points[current_index]

        for i in range(n):
            if not visited[i]:
                dist = calculate_distance(current_point, points[i])
                if dist < min_distance:
                    min_distance = dist
                    next_index = i
                    
        order.append(next_index)
        visited[next_index] = True
        current_index = next_index

    return order


def for_test():
    import time
    import random
    import matplotlib.pyplot as plt

    coordinates = [
        *[ [0, random.randint(0, 100), random.randint(0, 100)] for _ in range(100)]
        ]
    
    if True:
        coordinates = []
        for x in range(0,12):
            for y in range(0,8):
                coordinates.append([0,x,y])

    plt.plot([li[1] for li in coordinates]+[coordinates[0][1]],
             [li[2] for li in coordinates]+[coordinates[0][2]],
             ".-")
    plt.show()
    
    raw_distance = total_distance(coordinates, [i for i in range(len(coordinates))])
    print("raw_distance: ", raw_distance)
    start = time.time()
    order = nearest_neighbor(coordinates)
    total_distance(coordinates, order)
    end = time.time()
    print("elaplsed time: ", end-start)
    
    plt.plot([coordinates[i][1] for i in order]+[coordinates[0][1]],
             [coordinates[i][2] for i in order]+[coordinates[0][2]],
             ".-")
    plt.show()
    
    calculated_distance = total_distance(coordinates, order)
    print("calculated_distance: ", calculated_distance)
    
    
    
if __name__ == "__main__":
    for_test()
