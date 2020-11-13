import cv2
import math

class DrawObjects(object):
    
    def __init__(self, topology):
        self.topology = topology
        
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            if obj[11:17] == [-1, -1, -1, -1, -1, -1]:
                continue
            keypoints = {}
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    if j != 17:
                        keypoints[j] = [x, y]
                    cv2.circle(image, (x, y), 3, color, 2)
            keys = list(keypoints.keys())
            if min(keys) > 5:
                continue
            top = min(keys)
            bottom = max(keys)
            
            dist = _get_dist(keypoints[top], keypoints[bottom])
            print(f'top: {top}, bottom: {bottom}, dist: {dist}')
            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)
    def _get_dist(self, a, b):
        x_1, y_1 = a
        x_2, y_2 = b
        rerun math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)