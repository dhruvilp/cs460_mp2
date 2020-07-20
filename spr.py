"""
    Name: Dhruvil Patel
    NetId: dhp68
    Project: MP2
"""
import sys
import numpy as np
from heapq import *
from collections import defaultdict
import visualize
import matplotlib.pyplot as plt

def getPolygon(vertex, polygons):
    for i in range(0, len(polygons)):
        if vertex in polygons[i]:
            return i
    return -1

def is_intersecting(p1, p2, l1, l2):
    x2, y2 = p2[0], p2[1]
    x1, y1 = p1[0], p1[1]
    x3, y3 = l1[0], l1[1]
    x4, y4 = l2[0], l2[1]
    A = np.array([[y2 - y1, x1 - x2], [y4 - y3, x3 - x4]])
    b = np.array([[y2 * x1 - x2 * y1], [y4 * x3 - x4 * y3]])
    if np.linalg.det(A) == 0:
        return False
    else:
        inter = np.dot(np.linalg.inv(A), b)
        x = inter[0, 0]
        y = inter[1, 0]
        if (x - x1) * (x - x2) + (y - y1) * (y - y2) > -1e-10 or (x - x3) * (x - x4) + (y - y3) * (y - y4) > 1e-10:
            return False
    return True

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def line_func(p, p1, p2):
    return (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

def isBitangentingVertices(v1, v2, p1):
    last_sign = 0
    for v in p1:
        now_sign = np.sign(line_func(v, v1, v2))
        if now_sign == 0:
            continue
        if last_sign != 0 and now_sign != last_sign:
            return False
        last_sign = now_sign
    return True


def isBitangentingRoadMap(v1, v2, p2):
    id1 = getPolygon(v1, p2)
    id2 = getPolygon(v2, p2)
    for p in p2:
        for i in range(0, len(p)):
            pi = p[i]
            pj = p[(i + 1) % len(p)]
            if is_intersecting(v1, v2, pi, pj):
                return False
    if id1 == id2 and id1 != -1:
        if not isBitangentingVertices(v1, v2, p2[id1]):
            return False
    return True


'''
Report reflexive vertices
'''
def findReflexiveVertices(polygons):
    vertices = []
    for polygon in polygons:
        n = len(polygon)
        for i in range(0, n):
            p0 = polygon[i]
            p1 = polygon[(i - 1) % n]
            p2 = polygon[(i + 1) % n]
            if isBitangentingVertices(p0, p1, polygon) or isBitangentingVertices(p0, p2, polygon):
                vertices.append(p0)
    return vertices


'''
Compute the roadmap graph
'''
def computeSPRoadmap(polygons, reflexVertices):
    vertexMap = dict()
    adjacencyListMap = defaultdict(list)
    for i in range(0, len(reflexVertices)):
        vertexMap[i + 1] = reflexVertices[i]
    for i in range(0, len(reflexVertices)):
        ai = []
        for j in range(0, len(reflexVertices)):
            vi = reflexVertices[i]
            vj = reflexVertices[j]
            if i == j:
                continue
            else:
                pi = polygons[getPolygon(vi, polygons)]
                pj = polygons[getPolygon(vj, polygons)]
                if pi == pj and (i == ((j + 1) % len(pi)) or i == ((j - 1) % len(pj))):
                    ai.append([j + 1, get_distance(vi, vj)])
                else:
                    if isBitangentingRoadMap(vi, vj, polygons):
                        ai.append([j + 1, get_distance(vi, vj)])
        adjacencyListMap[i + 1] = ai.copy()
    return vertexMap, adjacencyListMap


'''
Perform uniform cost search 
'''
def uniformCostSearch(adjListMap, start, goal):
    global dis
    h = []
    mapping = dict()
    pathmap = dict()
    mapping[start] = 0
    pathmap[start] = [start]
    heappush(h, 0)
    while len(h) != 0:
        dis = heappop(h)
        new_dict = {v: k for k, v in mapping.items()}
        id = new_dict.get(dis)
        if id == goal:
            break
        for adj in adjListMap[id]:
            d = adj[1] + dis
            if adj[0] in mapping and d < mapping[adj[0]]:
                i = h.index(mapping[adj[0]])
                h[i] = d
                heapify(h)
                mapping[adj[0]] = d
                tmp = pathmap[id].copy()
                tmp.append(adj[0])
                pathmap[adj[0]] = tmp
            else:
                if adj[0] not in mapping:
                    heappush(h, d)
                    mapping[adj[0]] = d
                    tmp = pathmap[id].copy()
                    tmp.append(adj[0])
                    pathmap[adj[0]] = tmp

    pathLength = dis
    path = pathmap[goal]
    return path, pathLength


'''
Agument roadmap to include start and goal
'''
def updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2):
    updatedALMap = dict()
    startLabel = 0
    goalLabel = -1
    updatedALMap[startLabel] = []
    updatedALMap[goalLabel] = []
    for i in range(0, len(adjListMap)):
        updatedALMap[i + 1] = adjListMap[i + 1].copy()
        vi = vertexMap[i + 1]
        if isBitangentingRoadMap(vi, [x1, y1], polygons):
            updatedALMap[i + 1].append([startLabel, get_distance([x1, y1], vi)])
            updatedALMap[startLabel].append([i + 1, get_distance([x1, y1], vi)])
        if isBitangentingRoadMap(vi, [x2, y2], polygons):
            updatedALMap[i + 1].append([goalLabel, get_distance([x2, y2], vi)])
            updatedALMap[goalLabel].append([i + 1, get_distance([x2, y2], vi)])
    vertexMap[startLabel] = [x1, y1]
    vertexMap[goalLabel] = [x2, y2]
    return startLabel, goalLabel, updatedALMap


if __name__ == "__main__":
    # Retrive file name for input data
    if len(sys.argv) < 6:
        print("Five arguments required: python spr.py [env-file] [x1] [y1] [x2] [y2]")
        exit()

    filename = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

    # Read data and parse polygons
    lines = [line.rstrip('\n') for line in open(filename)]
    polygons = []
    for line in range(0, len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(0, len(xys)):
            polygon.append([float(i) for i in xys[p].split(',')])
        polygons.append(polygon)

    # Print out the data
    print("Pologonal obstacles:")
    for p in range(0, len(polygons)):
        print(str(polygons[p]))
    print("")

    # Compute reflex vertices
    reflexVertices = findReflexiveVertices(polygons)
    print("Reflexive vertices:")
    print(str(reflexVertices))
    print("")

    # Compute the roadmap
    vertexMap, adjListMap = computeSPRoadmap(polygons, reflexVertices)
    print("Vertex map:")
    print(str(vertexMap))
    print("")
    print("Base roadmap:")
    print(dict(adjListMap))
    print("")

    # Update roadmap
    start, goal, updatedALMap = updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2)
    print("Updated roadmap:")
    print(dict(updatedALMap))
    print("")

    # Search for a solution
    path, length = uniformCostSearch(updatedALMap, start, goal)
    print("Final path:")
    print(str(path))
    print("Final path length:" + str(length))

# **********************************************************
#                   Visualization Bonus
# **********************************************************

    for i in path:
        print(vertexMap[i])
    fig, ax = visualize.setupPlot()
    for v in findReflexiveVertices(polygons):
        plt.plot(v[0], v[1], marker='D')
    visualize.drawPolygons(polygons, fig, ax)
    for i in vertexMap:
        for adj in updatedALMap[i]:
            j = adj[0]
            if i < j:
                plt.plot([vertexMap[i][0], vertexMap[j][0]], [vertexMap[i][1], vertexMap[j][1]], linestyle=':',
                         color='grey', marker='o', markersize='8', markerfacecolor='black')
    x = [vertexMap[i][0] for i in path]
    y = [vertexMap[i][1] for i in path]
    plt.plot(x, y, linewidth=4, marker='o', markersize='8', markerfacecolor='yellow', color='blue')
    ax.plot()
    plt.show()
