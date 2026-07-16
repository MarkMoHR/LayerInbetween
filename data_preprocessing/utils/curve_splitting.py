import math


def compute_midpoint(p1, p2):
    # return (p1 + p2) / 2.0
    return [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]


class ControlVector():
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.compute_length()

    def compute_length(self):
        self.length = math.dist(self.start, self.end)

    def get_length(self):
        return self.length


class CubicBezierSegment():
    def __init__(self, segment, depth):
        self.stroke = segment
        self.depth = depth
        self.start_p, self.ctrl_p1, self.ctrl_p2, self.end_p = segment[0], segment[1], segment[2], segment[3]
        self.compute_distance()
        self.compute_vectors()

        self.split_curves = []
        self.split_check = False
        if depth <= 1:
            self.check_split()

    def compute_distance(self):
        self.end_distance = math.dist(self.start_p, self.end_p)

    def compute_vectors(self):
        v1 = ControlVector(self.start_p, self.ctrl_p1)
        v2 = ControlVector(self.end_p, self.ctrl_p2)
        self.vectors = [v1, v2]

    def check_split(self):
        end_distance = self.end_distance
        v1, v2 = self.vectors[0], self.vectors[1]
        vl1, vl2 = v1.get_length(), v2.get_length()
        if vl1 >= end_distance or vl2 >= end_distance:
            self.split_check = True
            self.split_curves = self.split()

    def split(self):
        A = self.start_p
        B = self.ctrl_p1
        C = self.ctrl_p2
        D = self.end_p
        E = compute_midpoint(A, B)
        F = compute_midpoint(B, C)
        G = compute_midpoint(C, D)
        H = compute_midpoint(E, F)
        I = compute_midpoint(F, G)
        K = compute_midpoint(H, I)
        c1 = [A, E, H, K]  # (4, 2)
        c2 = [K, I, G, D]  # (4, 2)
        curve_1 = CubicBezierSegment(c1, self.depth + 1)
        curve_2 = CubicBezierSegment(c2, self.depth + 1)
        return [curve_1.get_split_curves(), curve_2.get_split_curves()]

    def get_split_curves(self):
        if self.split_check:
            return self.split_curves
        else:
            return [self]

    def get_segment(self):
        return self.stroke


def fetch_nested_items(input):
    rst_items = []
    if isinstance(input, list):
        for item in input:
            rst_items += fetch_nested_items(item)
    else:
        rst_items.append(input)
    return rst_items


class CubicBezierChain():
    def __init__(self, path):
        # path: a curve with several strokes; list of (4, 2)
        self.BezierChainSplitStatus = []
        self.BezierSplitChain = []
        self.BezierSplitChainNested = []

        for each_path in path:  # single stroke
            stroke = CubicBezierSegment(each_path, depth=1)
            self.BezierChainSplitStatus.append(stroke.split_check)
            split_curves_nested = stroke.get_split_curves()
            split_curves = fetch_nested_items(split_curves_nested)
            self.BezierSplitChain += split_curves
            self.BezierSplitChainNested.append(split_curves)

    def export_chain(self):
        chain = [each.get_segment() for each in self.BezierSplitChain]
        return chain

    def export_chain_nested(self):
        chain = []
        for each in self.BezierSplitChainNested:
            chain_sub = [each_sub.get_segment() for each_sub in each]
            chain.append(chain_sub)
        return chain

    def get_split_statue(self):
        return self.BezierChainSplitStatus


def curve_splitting(curve_path):
    # curve_path: list of (4, 2)
    path = CubicBezierChain(curve_path)
    new_path = path.export_chain()  # list of (4, 2)
    new_path_nested = path.export_chain_nested()
    split_status = path.get_split_statue()
    return new_path, split_status, new_path_nested