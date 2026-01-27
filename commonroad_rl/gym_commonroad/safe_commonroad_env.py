import math
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Optional, Union, Dict

from scipy.interpolate import interp1d

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely import union_all
from gymnasium import spaces
import copy

from commonroad_rl.gym_commonroad.action.action import Action, ContinuousAction
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.state import State
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad.scenario.obstacle import Obstacle
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams
from commonroad_clcs.util import compute_orientation_from_polyline
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_clcs.pycrccosy import CartesianProjectionDomainError
from commonroad.geometry.shape import Rectangle
def traveled_distance(curve: np.ndarray, target):
    """
        Get the distance from the start of the given point along the curve used for
         lane propagation asin to know where the closest object in the next lane is.
    """
    i = np.argmin(np.linalg.norm(curve - target, axis=1))  # position of the point in the list
    d = 0.0
    for k in range(i):  # sum distances from curve[0] to curve[i]
        d += math.hypot(curve[k][0] - curve[k + 1][0], curve[k][1] - curve[k + 1][1])
    return d

def kappa(laneCenterPoints):
    xs = laneCenterPoints[:, 0]
    ys = laneCenterPoints[:, 1]
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # from https://www.storyofmathematics.com/curvature-formula/
    # Curvature formula k = |x'y'' − y'x''| / (x'^2 + y'^2)^(3/2)
    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom < 1e-12] = 1e-12
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    return float(np.mean(curvature))

def compute_kappa_dot_dot_helper(theta, pos, v, a_lat_max, kap, kappa_dot,ct, center_points,ncp,nct):
    """
        theta       ego orientation
        pos         ego position
        v           ego velocity
        a_lat_max   ego max laterial acceleration
        kap         ego kappa
        kappa_dot   ego yaw rate
        ct              commonroad clcs with the current lane center as refrence
        center_points   current lane center points
        ncp             next lane center points
        nct             commonroad clcs with the next lane center as refrence
    """
    try:
        s, d = ct.convert_to_curvilinear_coords(pos[0][0], pos[0][1])
    except CartesianProjectionDomainError:
        ccp = center_points[np.linalg.norm(center_points - pos, axis=1).argmin()]
        s, _ = ct.convert_to_curvilinear_coords(ccp[0], ccp[1])
        d = math.dist(pos[0], ccp)
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    la = max(5.0, v)
    local_center = extract_segment(ct, pos, center_points, s, la, ncp, nct)
    e_theta = wrap_to_pi(theta - float(compute_orientation_from_polyline(local_center).mean()))
    kappa_ref = kappa(local_center) - (0.8 * d) - (1.5 * e_theta)
    kappa_max = a_lat_max / (v ** 2)
    kappa_ref = np.clip(kappa_ref, - kappa_max, kappa_max)
    kappa_ddot = 4.0 * (kappa_ref - kap) - 2.0 * kappa_dot
    return float(np.clip(kappa_ddot / 20.0, -1.0, 1.0))

def extract_segment(ct : CurvilinearCoordinateSystem, pos, center_points, s, lookahead, nxt_cps,nct : CurvilinearCoordinateSystem):
    closest_centerpoint = np.linalg.norm(center_points - pos, axis=1).argmin()
    remain = traveled_distance(center_points[::-1], center_points[closest_centerpoint])
    if lookahead - 1e-4 < remain < lookahead + 1e-4:
        if closest_centerpoint == center_points.shape[0] - 1:   closest_centerpoint -= 2
        return center_points[closest_centerpoint:]
    elif remain > lookahead:
        far_pos = np.linalg.norm(center_points - np.array(ct.convert_to_cartesian_coords(s+lookahead,0)),axis=1).argmin()
        return center_points[closest_centerpoint:far_pos + 1]
    if nxt_cps is None:
        if closest_centerpoint == center_points.shape[0] - 1:   closest_centerpoint -= 2
        return center_points[closest_centerpoint:]
    lookahead_in_nxt = lookahead - remain
    if lookahead_in_nxt <0.1 or lookahead_in_nxt>traveled_distance(nxt_cps,nxt_cps[-1]):
        far_pos = len(center_points)
    else:
        far_pos = np.linalg.norm(nxt_cps - np.array(nct.convert_to_cartesian_coords(lookahead_in_nxt, 0)),axis=1).argmin()
    return np.vstack((center_points[closest_centerpoint:], nxt_cps[:far_pos + 1]))

class SafetyVerifier:

    def __init__(self, scenario: Scenario, prop_ego, precomputed_lane_polygons, dense_lanes, first_lane):
        self.l_id = None
        self.r_id = None
        self.scenario = scenario
        self.prop_ego = prop_ego
        self.safe_set : List[Tuple[List[Tuple[int,int,float,Polygon]],Lanelet]] = []
        self.precomputed_lane_polygons: Dict[int, Tuple[CurvilinearCoordinateSystem,
                    np.ndarray, np.ndarray, np.ndarray]] = precomputed_lane_polygons
        self.time_step = -1
        self.ego_lanelet = None
        self.dense_lanes : Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = dense_lanes
        self.first_lane = first_lane
        self.succsesors_to_add = []
        self.succsesors_to_addl = []
        self.succsesors_to_addr = []

    def get_reachable_lanes(self) -> List[Lanelet]:
        lanes = [self.ego_lanelet]
        self.succsesors_to_add = []
        self.succsesors_to_addl = []
        self.succsesors_to_addr = []
        def add_successors(lane,needed,k):
            lanes = []
            for l_id in lane.successor:
                s = self.scenario.lanelet_network.find_lanelet_by_id(l_id)
                l,c,r = self.dense_lanes[l_id]
                succ_len = traveled_distance(c,c[-1])
                lanes.append(s)
                if k == 0:  self.succsesors_to_add.append(l_id)
                elif k == 1: self.succsesors_to_addl.append(l_id)
                elif k == 2: self.succsesors_to_addr.append(l_id)
                if succ_len < needed:
                    lanes.extend(add_successors(s, needed - succ_len,0))
            return lanes
        for l_id in self.ego_lanelet.successor:
            lane = self.scenario.lanelet_network.find_lanelet_by_id(l_id)
            _, c, _ = self.dense_lanes[l_id]
            succ_len = traveled_distance(c,c[-1])
            if succ_len < self.prop_ego["ego_length"] + 1:
                lanes.extend(add_successors(lane, self.prop_ego["ego_length"] + 1 - succ_len, 0))
            lanes.append(lane)
        if self.ego_lanelet.adj_left_same_direction:
            ll = self.scenario.lanelet_network.find_lanelet_by_id(self.ego_lanelet.adj_left)
            lanes.append(ll)
            for l_id in ll.successor:
                lane = self.scenario.lanelet_network.find_lanelet_by_id(l_id)
                _, c, _ = self.dense_lanes[l_id]
                succ_len = traveled_distance(c, c[-1])
                if succ_len < self.prop_ego["ego_length"] + 1:
                    lanes.extend(add_successors(lane, self.prop_ego["ego_length"] + 1 - succ_len, 1))
                lanes.append(lane)
        if self.ego_lanelet.adj_right_same_direction:
            rl = self.scenario.lanelet_network.find_lanelet_by_id(self.ego_lanelet.adj_right)
            lanes.append(rl)
            for l_id in rl.successor:
                lane = self.scenario.lanelet_network.find_lanelet_by_id(l_id)
                _, c, _ = self.dense_lanes[l_id]
                succ_len = traveled_distance(c, c[-1])
                if succ_len < self.prop_ego["ego_length"] + 1:
                    lanes.extend(add_successors(lane, self.prop_ego["ego_length"] + 1 - succ_len, 2))
                lanes.append(lane)
        return lanes

    @staticmethod
    def get_lane_side_obs_intersection(x, y, orientation, length, width, curve: np.ndarray):
        def local_rotate(px, py):
            s, c = math.sin(-orientation), math.cos(-orientation)
            dx, dy = px - x, py - y
            return dx * c - dy * s, dx * s + dy * c
        local = np.array([local_rotate(px, py) for px, py in curve])
        l, w = length / 2, width / 2
        rect = np.array([ [-l, -w], [l, -w],[l, -w], [l, w],
                [l, w], [-l, w],[-l, w], [-l, -w]   ])
        def segment_intersect(p1, p2, q1, q2):
            def ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
            return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
        hits = []
        for i in range(len(local) - 1):
            p1, p2 = local[i], local[i + 1]
            for j in range(0, len(rect), 2):
                q1, q2 = rect[j], rect[j + 1]
                if segment_intersect(p1, p2, q1, q2):
                    hits.append(i)
                    break
        if not hits:    return None
        return hits[:2]

    def obs_start_end_index(self, obs : Obstacle, l_id) -> list[int]:
        obs_state = obs.state_at_time(self.time_step)
        shape = obs.occupancy_at_time(self.time_step).shape
        left,center,right = self.dense_lanes[l_id]
        l = shape.length if isinstance(shape,Rectangle) else shape.radius
        w = shape.width if isinstance(shape,Rectangle) else shape.radius
        o = obs_state.orientation if isinstance(shape,Rectangle) else 0.0
        c_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1], o, l, w, center)
        l_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1], o, l, w, left)
        r_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1], o, l, w, right)
        pts = []
        for k in (c_pts, l_pts, r_pts):
            if k is not None:   pts.extend(k)
        if not pts: return []
        start = min(pts)
        end = max(pts)
        if start == end:    return [start]
        else:   return [start, end]

    def get_end_collision_free_area(self, lane : Lanelet, center, pt : list[int], preceding_v,depth = 0,max_depth = 3)\
            -> (np.ndarray,Lanelet,float,float,float):
        if depth > max_depth:
            return center[pt[1]:], lane, preceding_v, 0.0, 0.0
        successors = lane.successor
        if len(successors) == 0:
            return center[pt[1] : len(center)], lane, preceding_v, self.prop_ego["v_max"] , 0
        def get_closest_obstacle_lane_velocity_distance(ls: Lanelet):
            lso = ls.get_obstacles(self.scenario.obstacles, self.time_step)
            cp = self.dense_lanes[ls.lanelet_id][1]
            if len(lso) == 0:
                k = kappa(cp)
                if k == 0:  r_min = math.inf
                else:   r_min = 1.0 / k
                v_crit = np.sqrt(r_min * self.prop_ego["a_lat_max"])
                return min(v_crit,self.prop_ego["v_max"]), traveled_distance(cp, cp[-1]), False
            else:
                lso = self.sort_obstacles_in_lane(ls.lanelet_id, lso)
                lobs_state = lso[0].state_at_time(self.time_step)
                pts = self.obs_start_end_index(lso[0],ls.lanelet_id)
                if not pts:
                    s_centers = self.precomputed_lane_polygons[ls.lanelet_id][1]
                    p = np.asarray(lobs_state.position).reshape(1, 2)
                    closest_centerpoint = np.linalg.norm(cp - p, axis=1).argmin()
                    s = s_centers[closest_centerpoint]
                    return lobs_state.velocity, s, True
                return lobs_state.velocity, traveled_distance(cp,cp[pts[0]]), True
        s_v_d = []
        for successor in successors:
            v_i, d_i, found = get_closest_obstacle_lane_velocity_distance(self.scenario.lanelet_network.find_lanelet_by_id(successor))
            if not found:
                next_lane = self.scenario.lanelet_network.find_lanelet_by_id(successor)
                _, _, _, rec_v, rec_d = self.get_end_collision_free_area(
                    next_lane, self.dense_lanes[successor][1], [0, 0], v_i, depth + 1, max_depth)
                total_dist = d_i + rec_d
                min_v = min(v_i, rec_v)
                s_v_d.append((successor, min_v, total_dist))
            else:
                s_v_d.append((successor, v_i, d_i))
        min_ttc = math.inf
        closest_car = s_v_d[0][0]
        for lane_id, v_i, d_i in s_v_d:
            relative_speed = self.prop_ego["v_max"] - v_i
            if relative_speed <= 0: continue
            ttc = d_i / relative_speed
            if ttc < min_ttc:
                min_ttc = ttc
                closest_car = lane_id
        v,d = 0, 0
        for lane_id, v_i, d_i in s_v_d:
            if lane_id == closest_car:
                d = d_i
                v = v_i
        return center[pt[1] : len(center)], lane, preceding_v, v , d

    def get_lane_collision_free_areas(self, lane : Lanelet) -> List[Tuple[np.ndarray,Lanelet,float,float,float]]:
        C : List[Tuple[np.ndarray,Lanelet,float,float,float]] = []
        center = self.dense_lanes[lane.lanelet_id][1]
        obs = lane.get_obstacles(self.scenario.obstacles, self.time_step)
        if len(obs) == 0:   # empty lane with no vehicle entering or exiting it
            return [self.get_end_collision_free_area(lane, center, [0,0], 0)]
        i = 0
        obs = self.sort_obstacles_in_lane(lane.lanelet_id, obs)
        obs_state = obs[i].state_at_time(self.time_step)
        pts = self.obs_start_end_index(obs[i], lane.lanelet_id)
        def solve_0_pts(os, q):
            s_centers = self.precomputed_lane_polygons[lane.lanelet_id][1]
            p = np.asarray(os.position).reshape(1, 2)
            closest_centerpoint = np.linalg.norm(self.dense_lanes[lane.lanelet_id][1] - p, axis=1).argmin()
            s = s_centers[closest_centerpoint]
            l = obs[q].obstacle_shape.length if isinstance(obs[q].obstacle_shape, Rectangle) else obs[q].obstacle_shape.radius
            start = s + l/2
            end = s - l/2
            if start == end: return np.array(np.linalg.norm(s_centers - start).argmin())
            else: return np.array([np.linalg.norm(s_centers - start).argmin(),
                   np.linalg.norm(s_centers - end).argmin()])
        if len(pts) == 0:   pts = solve_0_pts(obs_state, 0)
        preceding_v = obs_state.velocity
        if len(pts) == 1:
            pt = pts[0]
        else:
            cps = center[0 : pts[0] + 1]
            C.append((cps,lane, 0.0, preceding_v, 0.0)) # add first collision free area
            pt = pts[1]
        for i in range(len(obs) - 2):
            obs_state = obs[i + 1].state_at_time(self.time_step)
            pts = self.obs_start_end_index(obs[i + 1], lane.lanelet_id)
            if len(pts) == 0:   pts = solve_0_pts(obs_state, i+1)
            cps = center[pt : pts[0] + 1]
            C.append((cps, lane, preceding_v, obs_state.velocity, 0.0))  # add middle collision free areas
            if len(pts) == 2:
                pt = pts[1]
            preceding_v = obs_state.velocity
        if len(pts) == 2 and ((lane.lanelet_id == self.ego_lanelet.lanelet_id) or
                              (self.l_id is not None and self.l_id == lane.lanelet_id) or
                              (self.r_id is not None and self.r_id == lane.lanelet_id)):
            # add last collision free area only for the ego and adjacent lanes
            C.append(self.get_end_collision_free_area(lane, center, pts, preceding_v))
        if len(C) == 0:
            if traveled_distance(center,center[pts[0]]) < self.prop_ego["ego_length"]:
                C.append(self.get_end_collision_free_area(lane, center, [0,pts[0]],preceding_v))
            else:
                C.append((center[0 : pts[0]+1],lane, 0.0, preceding_v, 0.0))
        return C

    def sort_obstacles_in_lane(self, l_id : int ,obs : List[Obstacle], offset = 0) -> List[Obstacle]:
        obs_with_center : List[Tuple[Obstacle, float]] = []
        ct= self.precomputed_lane_polygons[l_id][0]
        for ob in obs:
            pos = (ob.state_at_time(self.time_step + offset).position[0], ob.state_at_time(self.time_step + offset).position[1])
            try:
                pos_inlane = ct.convert_to_curvilinear_coords(*pos)
            except CartesianProjectionDomainError:
                cp = ct.reference_path_original()
                if math.dist(pos,cp[0]) > math.dist(pos,cp[1]):
                    pos_inlane = ct.convert_to_curvilinear_coords(*cp[1])
                else:
                    pos_inlane = ct.convert_to_curvilinear_coords(*cp[0])
            obs_with_center.append((ob, pos_inlane[0]))
        obs_with_center.sort(key=lambda k: k[1])
        return list(obs for obs, obs_center in obs_with_center)

    def build_safe_area(self,start,end,l_id, ego_state):
        lb,c,rb = self.dense_lanes[l_id]
        valid_road_polygons = []
        lane = self.scenario.lanelet_network.find_lanelet_by_id(l_id)
        if start == 0:
            if l_id == self.first_lane: # sometimes the ego starts from outside the map
                p = ego_state.position
                W, L = self.prop_ego["ego_width"], self.prop_ego["ego_length"]
                rect = Polygon([(-L / 2, -W), (-L / 2, W), (L / 2, W), (L / 2, -W)]).buffer(.2)
                rect = rotate(rect, ego_state.orientation * 180 / math.pi, origin=(0, 0), use_radians=False)
                rect = translate(rect, xoff=p[0], yoff=p[1])
                valid_road_polygons.append(rect)
            for p_id in lane.predecessor:
                pl, pc, pr = self.dense_lanes[p_id]
                valid_road_polygons.append(Polygon(pl.tolist() + pr.tolist()[::-1]).buffer(.2))
                p_len = traveled_distance(pc,pc[-1])
                if p_len < self.prop_ego["ego_length"]:
                    for pp_id in self.scenario.lanelet_network.find_lanelet_by_id(p_id).predecessor:
                        pl, pc, pr = self.dense_lanes[pp_id]
                        valid_road_polygons.append(Polygon(pl.tolist() + pr.tolist()[::-1]).buffer(.2))
        poly = Polygon(lb + rb[::-1]).buffer(.2)
        valid_road_polygons.append(poly)
        lane_polygon = union_all(valid_road_polygons)
        lane_polygon.buffer(-0.2)
        return lane_polygon

    def safeDistanceSetForSection(self, xi:float, yi:float, v_i:float, xj:float, yj:float, v_j:float,cp, l_id, distance_to_add, ego_state) \
            -> List[Tuple[float,float,float,Polygon]] :
        """
            This function takes the position and velocity of two Obstacles and the collision
            free area between them (as a part of the lane i.e. left-,center- and right points)
            and returns all the possible (v,poly) that ego vehicle to be in with:
               - v -> Velocities of the ego vehicle
               - p -> Shapely Polygon
        """
        if traveled_distance(cp,cp[-1]) < self.prop_ego["ego_length"]: return []
        ct, s_centers, _, _ = self.precomputed_lane_polygons[l_id]
        txi, _ = ct.convert_to_curvilinear_coords(xi, yi)
        txj, _ = ct.convert_to_curvilinear_coords(xj, yj)
        txj += distance_to_add
        a_lat_max, a_lon_max, w, l, delta_react = (self.prop_ego["a_lat_max"], self.prop_ego["a_lon_max"],
                self.prop_ego["ego_width"], self.prop_ego["ego_length"], self.prop_ego["delta_react"])
        k = kappa(cp)
        if k == 0:  r_min = math.inf
        else:   r_min = 1.0 / k
        v_crit = min(np.sqrt(r_min * a_lat_max),self.prop_ego["v_max"])
        # s >= s_i + Δ_safe(v, i)
        # s <= s_j - Δ_safe(v, j)
        vs = np.linspace(0, v_crit, 500)
        safe_states = []
        # a_lon(v) = a_lon_max * sqrt( 1 - (v^2 / v_crit^2)^2 )
        def a_lon(v, a_lon_max, v_crit ):
            return a_lon_max * np.sqrt(max(0.01, 1 - (v ** 2 / v_crit ** 2) ** 2))
        # z(v,j) = v^2/(2 a_ego(v)) - v_j^2/(2 a_j_max) + delta_react*v
        def zeta_preceding(v, v_j, a_lon_max, v_crit, delta_react):
            return (v ** 2) / (2 * abs(a_lon(v,a_lon_max,v_crit))) - (v_j ** 2) / (2 * abs(a_lon_max)) + delta_react * v
        # z(v,i) = v_i^2/(2|a_i_max|) - v^2/(2|a_ego(v)|) + delta_react*v_i
        def zeta_succeeding(v, v_i, a_lon_max, v_crit, delta_react):
            return (v_i ** 2) / (2 * abs(a_lon_max)) - (v ** 2) / (2 * abs(a_lon(v,a_lon_max,v_crit))) + delta_react * v_i
        for v in vs:
            s_min_final = max(txi + zeta_succeeding(v,v_i,a_lon_max,v_crit,delta_react), 0)
            s_max_final = txj - zeta_preceding(v,v_j,a_lon_max,v_crit,delta_react)
            if s_min_final < s_max_final:
                start = np.argmin(np.abs(s_centers - s_min_final))
                end = np.argmin(np.abs(s_centers - s_max_final))
                if start == end: continue
                safe_states.append((start,end, v, self.build_safe_area(start,end,l_id,ego_state)))
        return safe_states

    def expand_area_in_succ(self, S):
        ll_id = None if not self.ego_lanelet.adj_left_same_direction else self.ego_lanelet.adj_left
        rl_id = None if not self.ego_lanelet.adj_right_same_direction else self.ego_lanelet.adj_right
        def comulate_polys(lanes):
            polys = []
            for k,lane in S:
                if lane.lanelet_id in lanes:
                    spolys = []
                    for s,e,_,p in k:
                        if s == 0: spolys.append(p)
                    if not spolys: continue
                    polys.append(union_all(spolys).buffer(0.2))
            if not polys: return None
            return union_all(polys).buffer(0.2)
        for i in range(len(S)):
            ko,laneo = S[i]
            if laneo.lanelet_id == self.ego_lanelet.lanelet_id: polys = comulate_polys(self.succsesors_to_add)
            elif ll_id and laneo.lanelet_id == ll_id: polys = comulate_polys(self.succsesors_to_addl)
            elif rl_id and laneo.lanelet_id == rl_id: polys = comulate_polys(self.succsesors_to_addr)
            else: continue
            if polys is None: continue
            nk = []
            for s,e,v,p in ko:
                if e >= len(self.dense_lanes[laneo.lanelet_id][1]) - 1:
                    p = p.buffer(0.2).union(polys)
                    nk.append((s,e,v,p.buffer(-0.2)))
                else:
                    nk.append((s, e, v, p))
            S[i] = (nk,laneo)
        return S

    def union_safe_set(self, ll: Lanelet, safe_set_list_left : List[Tuple[int,int,float,Polygon]]
                            , rl : Lanelet, safe_set_list_right :List[Tuple[int,int,float,Polygon]]):
        """
            Does a left only union of the safe sets of two adjacent same direction lanes
            returns a List of the two new safe sets
        """
        ct,ls, _, lrp = self.precomputed_lane_polygons[ll.lanelet_id]
        nls = []
        nrs = []
        for s in safe_set_list_left:
            c1, c2, lv, _ = s
            l_start = int(ls[c1])
            l_end = int(ls[c2])
            for c in safe_set_list_right:
                c1, c2, rv, _ = c
                _, rcp, rrp = self.dense_lanes[rl.lanelet_id]
                r_start, _ = ct.convert_to_curvilinear_coords(rcp[c1][0],rcp[c1][1])
                r_end, _ = ct.convert_to_curvilinear_coords(rcp[c2][0],rcp[c2][1])
                start = max(l_start, r_start)
                end = min(l_end, r_end)
                if start < end:
                    llp = self.dense_lanes[ll.lanelet_id][1].tolist()
                    flrp = []
                    for p in lrp:
                        flrp.append(ct.convert_to_cartesian_coords(p[0], p[1] - 4))
                    rct,_,rlp,_ = self.precomputed_lane_polygons[rl.lanelet_id]
                    frlp = []
                    for p in rlp:
                        frlp.append(ct.convert_to_cartesian_coords(p[0], p[1] + 4))
                    nls.append((start, end, lv, Polygon(llp + flrp[::-1]).buffer(0)))
                    nrs.append((start, end, rv, Polygon(frlp + rrp.tolist()[::-1]).buffer(0)))
        return [(nls,ll), (nrs,rl)]

    def safeDistanceSet(self, ego_lanelet : Lanelet, in_or_entering_intersection,ego_state):
        """
            Constructs the list of the admissible velocities and locations for the ego Vehicle.
                We must first construct Collision free areas as:
                    - Centerpoints between the two Obstacles,
                    - Lane
                    - Preceding and succeeding Obstacles velocities
                    - Distance between the obstacle with the smallest ttc in all next lanes and the current lane end.
                Returns Safe states as a List of Tuples each :
                    - List of Tuples each:
                        - First and Last Indices
                        - Velocity
                        - Shapely Polygon of the safe area
                    - Lane
        """
        self.ego_lanelet = ego_lanelet
        self.l_id = self.ego_lanelet.adj_left if self.ego_lanelet.adj_left_same_direction else None
        self.r_id = self.ego_lanelet.adj_right if self.ego_lanelet.adj_right_same_direction else None
        self.time_step += 1
        S : List[Tuple[List[Tuple[float,float,float,Polygon]],Lanelet]] = []
        C = []
        for lane in self.get_reachable_lanes():
            C.extend(self.get_lane_collision_free_areas(lane))
        for c in C:
            cp, l, vi, vj, d = c
            if len(cp) < 3 : continue
            S.append((self.safeDistanceSetForSection(cp[0][0],cp[0][1],vi,cp[-1][0],cp[-1][1],vj,cp,l.lanelet_id,d,ego_state),l))
        #   For lane change we must have parts where the safe bounds don't exist,
        #   we do this by expanding the bounds into the adj lane when two safe states area are next to each other.
        #   We only do unions to the left side i.e. left lane with ego and ego with right lane.
        S = self.expand_area_in_succ(S)
        es = []
        for s in S:
            k, lane = s
            if lane == self.ego_lanelet:
                es.extend(k)
        if not in_or_entering_intersection:
            if self.l_id:
                ll : Lanelet = self.scenario.lanelet_network.find_lanelet_by_id(self.ego_lanelet.adj_left)
                ls = []
                for s in S:
                    k, lane = s
                    if lane.lanelet_id == ll:
                        ls.extend(k)
                S.extend(self.union_safe_set(ll,ls,self.ego_lanelet,es))
            if self.ego_lanelet.adj_right_same_direction:
                rl : Lanelet = self.scenario.lanelet_network.find_lanelet_by_id(self.ego_lanelet.adj_right)
                rs = []
                for s in S:
                    k, lane = s
                    if lane.lanelet_id == rl:
                        rs.extend(k)
                S.extend(self.union_safe_set(self.ego_lanelet,es,rl,rs))
        self.safe_set = S

    def compute_kappa_dot_dot(self, l_id, nxt_id, state):
        center_points = self.dense_lanes[l_id][1]
        v = max(state.velocity,0.1)
        kap = state.slip_angle
        kappa_dot = state.yaw_rate
        pos = state.position.reshape(1, 2)
        theta = state.orientation
        return compute_kappa_dot_dot_helper(theta, pos, v,self.prop_ego["a_lat_max"],kap,kappa_dot,
                        self.precomputed_lane_polygons[l_id][0], center_points,
                        self.dense_lanes[nxt_id][1] if nxt_id != 0 else None,
                        self.precomputed_lane_polygons[nxt_id][0] if nxt_id != 0 else None)

    def find_safe_jerk_dot(self, ego_action, kappa_ddot, l_id, nxt_id):
        """
            Binary search for the min and max jerk_dot for given kappa_dot_dot.
            Using the binary search made it has constant complexity of 18 iterations for each 36 checks in total
        """
        low, high = -0.8, 0.8
        while high - low > 1e-5:
            mid = (low + high) / 2
            copy_action : ContinuousAction = copy.deepcopy(ego_action)
            if self.safe_action_check(mid, kappa_ddot, copy_action, 0, l_id, nxt_id):
                high = mid
            else:   low = mid
        safe_min = high
        low, high = -0.8, 0.8
        while high - low > 1e-5:
            mid = (low + high) / 2
            copy_action : ContinuousAction = copy.deepcopy(ego_action)
            if self.safe_action_check(mid, kappa_ddot, copy_action, 0, l_id, nxt_id):
                low = mid
            else:   high = mid
        safe_max = low
        return safe_min, safe_max

    def find_feisable_jerk_dot(self, ego_action, kappa_ddot, l_id = 0, nxt_id = 0, k = 0):
        for i in range(11):
            current_val = (0.05 * ((i + 1) // 2)) * (1 if i % 2 != 0 else -1)
            if not (-0.8 <= current_val <= 0.8):    continue
            copy_action: ContinuousAction = copy.deepcopy(ego_action)
            if self.safe_action_check(current_val, kappa_ddot, copy_action, k, l_id, nxt_id):
                return current_val
        return -2

    def check_feisable_jerk_dot(self, ego_action, kappa_ddot, l_id = 0, nxt_id = 0, k = 0):
        for i in range(11):
            current_val = (0.05 * ((i + 1) // 2)) * (1 if i % 2 != 0 else -1)
            if not (-0.8 <= current_val <= 0.8):    continue
            copy_action: ContinuousAction = copy.deepcopy(ego_action)
            if self.safe_action_check(current_val, kappa_ddot, copy_action, k, l_id, nxt_id):
                return True
        return False

    """def safe_action_check(self, jd, kdd, ego_action : Action, q = 0, l_id = 0, nxt_id = 0):
        if q == 4:
            ego_action.step((np.array(jd,kdd)))
            new_vehicle_state = ego_action.vehicle.state
            p = new_vehicle_state.position
            nv = new_vehicle_state.velocity
            W, L = self.prop_ego["ego_width"], self.prop_ego["ego_length"]
            rect = Polygon([(-L / 2, -W / 2), (-L / 2, W / 2), (L / 2, W / 2), (L / 2, -W / 2)])
            rect = rotate(rect, new_vehicle_state.orientation * 180 / math.pi, origin=(0, 0), use_radians=False)
            rect = translate(rect, xoff=p[0], yoff=p[1])
            try:
                curr_l = self.scenario.lanelet_network.find_lanelet_by_position(p)
            except:
                curr_l = None
            if curr_l is not None:
                for s in self.safe_set:
                    k, lane = s
                    if lane.lanelet_id == curr_l:
                        for start, end, v, poly in k:
                            if v - .1 > nv: break
                            if start == end or not (v - .1 <= nv <= v + .1): continue
                            if poly.contains(rect): return True
            else:
                for l in self.get_reachable_lanes():
                    for s in self.safe_set:
                        k, lane = s
                        if lane.lanelet_id == l.lanelet_id:
                            for start, end, v, poly in k:
                                if start == end or not (v - 1 <= nv <= v + 1) : continue
                                if poly.contains(rect): return True
            return False
        q += 1
        ego_action.step(np.array([jd,kdd]))
        new_vehicle_state = ego_action.vehicle.state
        kdd = self.compute_kappa_dot_dot(l_id, nxt_id, new_vehicle_state)
        kappa_dot_dots = np.linspace(kdd - 0.05, kdd + 0.05, 3)
        for kdd in kappa_dot_dots:
            if self.check_feisable_jerk_dot(ego_action, kdd, l_id, nxt_id, q):  return True
        return False"""

    def safe_action_check(self, jd, kdd, ego_action : Action, q = 0, l_id = 0, nxt_id = 0):
        if q == 2:            return True
        q += 1
        ego_action.step(np.array([jd,kdd]))
        new_vehicle_state = ego_action.vehicle.state
        p = new_vehicle_state.position
        nv = new_vehicle_state.velocity
        W, L = self.prop_ego["ego_width"], self.prop_ego["ego_length"]
        rect = Polygon([(-L / 2, -W / 2), (-L / 2, W / 2), (L / 2, W / 2), (L / 2, -W / 2)])
        rect = rotate(rect, new_vehicle_state.orientation * 180 / math.pi, origin=(0, 0), use_radians=False)
        rect = translate(rect, xoff=p[0], yoff=p[1])
        hl = L / 2.0
        hw = W / 2.0
        c = math.cos(new_vehicle_state.orientation)
        s = math.sin(new_vehicle_state.orientation)
        local = [[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]]
        corners = []
        for k in local:
            r = [k[0] * c - k[1] * s, k[0] * s + k[1] * c];
            corners.append(np.array([p[0] + r[0], p[1] + r[1]]))
        try:
            curr_l = self.scenario.lanelet_network.find_lanelet_by_position(corners)
            curr_l = [item for sublist in curr_l for item in sublist]
        except:
            curr_l = None
        for s in self.safe_set:
            k, lane = s
            if curr_l is not None and lane.lanelet_id not in curr_l: continue
            for start, end, v, poly in k:
                if v - .1 > nv: break
                if start == end or not (v - .1 <= nv <= v + .1) : continue
                if poly.contains(rect):
                    kdd = self.compute_kappa_dot_dot(l_id,nxt_id,new_vehicle_state)
                    if kdd > 0.8 or kdd < -0.8: return False
                    if self.check_feisable_jerk_dot(ego_action, kdd, l_id, nxt_id, q):   return True
                    #kappa_dot_dots = np.linspace(kdd - 0.05, kdd + 0.05, 3)
                    #for kdd in kappa_dot_dots:
                    #    if self.check_feisable_jerk_dot(ego_action,kdd,l_id,nxt_id,q):   return True
        return False

class SafetyLayer(CommonroadEnv):

    def __init__(self, meta_scenario_path=PATH_PARAMS["meta_scenario"],
                 train_reset_config_path=PATH_PARAMS["train_reset_config"],
                 test_reset_config_path=PATH_PARAMS["test_reset_config"],
                 visualization_path=PATH_PARAMS["visualization"], logging_path=None, test_env=False, play=False,
                 config_file=PATH_PARAMS["configs"]["commonroad-v1"], logging_mode=1, **kwargs) -> None:
        kwargs["flatten_observation"] = False
        super().__init__(meta_scenario_path, train_reset_config_path, test_reset_config_path, visualization_path,
                         logging_path, test_env, play, config_file, logging_mode, **kwargs)
        self.observation = None
        self.past_ids = []
        self.prop_ego = {"ego_length" : 4.5, "ego_width" : 1.6 , "a_lat_max" : 9.0, "a_lon_max" : 11.5, "delta_react" : 0.5}
        self.time_step = 0
        self.last_relative_heading = 0
        self.prop_ego["v_max"] = 45
        self.final_priority = -1
        self.intersection_lanes : List[Lanelet] = []
        self.conflict_lanes : defaultdict[int, List[Tuple[Lanelet, bool]]] = defaultdict(list)
        self.pre_intersection_lanes = None
        self.precomputed_lane_polygons : Dict[int, Tuple[CurvilinearCoordinateSystem, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.dense_lanes : Dict[int,Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.safety_verifier : SafetyVerifier = None
        self.in_or_entering_intersection = False
        self.new_low = np.concatenate([self.observation_collector.observation_space.low.astype(np.float32),
                                        np.full(35, -1.0, dtype=np.float32)])
        self.new_high = np.concatenate([self.observation_collector.observation_space.high.astype(np.float32),
                                        np.full(35, 1.0, dtype=np.float32)])
        self.l_id = 0
        self.nxt_id = 0
        self.kappa_dds = []
        self.type = 0

    def pack_observation(self, observation_dict):
        def pack_orig():
            observation_vector = np.zeros(self.observation_collector.observation_space.shape)
            index = 0
            for k in observation_dict.keys():
                if k in ["safe_actions", "final_priority", "distance_to_lane_end"] : continue
                size = np.prod(self.observation_dict[k].shape)
                observation_vector[index: index + size] = observation_dict[k].flat
                index += size
            return observation_vector
        base = pack_orig()
        sa = self.observation["safe_actions"]
        vec = np.zeros(sa.size + 2 + base.size, dtype=np.float64)
        idx = 0
        vec[idx:idx + base.size] = base
        idx += base.size
        vec[idx] = float(self.observation["distance_to_lane_end"])
        idx += 1
        vec[idx:idx + sa.size] = sa
        idx += sa.size
        vec[idx] = float(self.observation["final_priority"])
        return vec

    def apply_safety(self, observation,terminated):
        if self.in_or_entering_intersection:
            actions = self.intersection_safety()
        else:
            self.pre_intersection_lanes = None
            self.final_priority = -1
            actions = self.lane_safety()
        if actions.size > 33:   actions = actions[:33]
        elif actions.size < 33:   actions = np.pad(actions, (0, 33 - actions.size), mode='constant', constant_values=0)
        self.observation["safe_actions"] = actions
        self.observation["final_priority"] = np.array([self.final_priority], dtype=object)
        observation_vector = self.pack_observation(observation)
        #if terminated:
        #    print(self.termination_reason)
        #    for k in self.observation.keys():   print(k, " : ", self.observation[k])
        return observation_vector

    def get_distance_to_lane_end(self):
        _, center_points, _ = self.dense_lanes[self.observation_collector.ego_lanelet.lanelet_id]
        ego_pos = np.asarray(self.observation_collector._ego_state.position).reshape(1 ,2)
        closest_centerpoint = center_points[np.linalg.norm(center_points - ego_pos, axis=1).argmin()]
        self.observation["distance_to_lane_end"] = traveled_distance(center_points[::-1],closest_centerpoint)

    def reset(self, seed=None, options: Optional[dict] = None, benchmark_id=None, scenario: Scenario = None,
              planning_problem: PlanningProblem = None) -> np.ndarray:
        self.past_ids = []
        initial_observation, info = super().reset(seed, options, benchmark_id, scenario, planning_problem)
        self.past_ids.append(self.observation_collector.ego_lanelet.lanelet_id)
        self.l_id = self.observation_collector.ego_lanelet.lanelet_id
        self.nxt_id = 0
        self.time_step = 0
        self.time_step = 0
        self.compute_lane_sides_and_conflict()
        self.observation = initial_observation.copy()
        self.get_distance_to_lane_end()
        self.safety_verifier = SafetyVerifier(self.scenario, self.prop_ego, self.precomputed_lane_polygons,
                                              self.dense_lanes, self.observation_collector.ego_lanelet.lanelet_id)
        self.in_or_entering_intersection = self.intersection_check()
        self.safety_verifier.safeDistanceSet(self.observation_collector.ego_lanelet,self.in_or_entering_intersection,self.observation_collector._ego_state)
        self.pre_intersection_lanes = None
        observation_vector = self.apply_safety(initial_observation,False)
        return observation_vector, info

    def compute_lane_sides_and_conflict(self):
        """
            Converts all lanes borders to Curvilinear coordinates and compute the relative conflict lanes.
        """
        def is_right(a: np.ndarray, b: np.ndarray) -> bool:
            def segment_intersection(p1, p2, q1, q2):
                A = np.array([p2 - p1, q1 - q2]).T
                b_vec = q1 - p1
                try:  # [p2-p1 q1−q2] [t u]^T = q1−p1
                    t, u = np.linalg.solve(A, b_vec)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        return True
                except np.linalg.LinAlgError:
                    return False
                return False
            for i in range(len(a) - 1):
                for j in range(len(b) - 1):
                    if segment_intersection(a[i], a[i + 1], b[j], b[j + 1]):
                        ta = a[i + 1] - a[i]
                        ta /= np.linalg.norm(ta)
                        tb = b[j + 1] - b[j]
                        tb /= np.linalg.norm(tb)
                        cross = np.float64(ta[0] * tb[1] - ta[1] * tb[0])
                        return cross < 0  # True if coming from right
            return False
        self.precomputed_lane_polygons.clear()
        self.dense_lanes.clear()
        self.conflict_lanes.clear()
        def densify_lane_and_normalize(center, left, right, ds=0.1):
            def arclength_parametrize(poly):
                d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
                d[d < 1e-6] = 0.0
                s = np.insert(np.cumsum(d), 0, 0.0)
                s += np.arange(len(s)) * 1e-9  # enforce strict monotonicity
                return s, poly
            s_c, c = arclength_parametrize(center)
            s_l, l = arclength_parametrize(left)
            s_r, r = arclength_parametrize(right)
            s_new = np.arange(0, s_c[-1], ds)
            def interp_curve(s_old, curve):
                from scipy.interpolate import CubicSpline
                fx = CubicSpline(s_old, curve[:, 0], bc_type="natural")
                fy = CubicSpline(s_old, curve[:, 1], bc_type="natural")
                return np.column_stack((fx(s_new), fy(s_new)))
            center_new = interp_curve(s_c, c)
            left_new = interp_curve(s_l, l)
            right_new = interp_curve(s_r, r)
            t = np.diff(center_new, axis=0)
            n = np.column_stack((-t[:, 1], t[:, 0]))
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            for i in range(len(n)):
                left_new[i] = center_new[i] + np.dot(left_new[i] - center_new[i], n[i]) * n[i]
                right_new[i] = center_new[i] + np.dot(right_new[i] - center_new[i], n[i]) * n[i]
            return center_new, left_new, right_new
        for l in self.scenario.lanelet_network.lanelets:
            center_dense, left_dense, right_dense = densify_lane_and_normalize(l.center_vertices,l.left_vertices,l.right_vertices)
            ct = CurvilinearCoordinateSystem(center_dense, CLCSParams())
            left, right = [], []
            to_remove = []
            for i in range(len(left_dense)):
                try:
                    left.append(np.array(ct.convert_to_curvilinear_coords(left_dense[i][0], left_dense[i][1])))
                    right.append(np.array(ct.convert_to_curvilinear_coords(right_dense[i][0], right_dense[i][1])))
                except CartesianProjectionDomainError:
                    to_remove.append(i)
            center_dense = np.delete(center_dense, to_remove, axis=0)
            right_dense = np.delete(right_dense, to_remove, axis=0)
            left_dense = np.delete(left_dense, to_remove, axis=0)
            self.dense_lanes[l.lanelet_id] = (left_dense, center_dense, right_dense)
            left = np.array(left)
            right = np.array(right)
            s_centers = np.array([])
            for x, y in center_dense:
                s, d = ct.convert_to_curvilinear_coords(x, y)
                s_centers = np.append(s_centers, s)
            self.precomputed_lane_polygons[l.lanelet_id] = (ct, s_centers, left, right)
        for l in self.scenario.lanelet_network.lanelets:
            for k in self.scenario.lanelet_network.lanelets:
                if l.lanelet_id == k.lanelet_id or (l.predecessor and k.predecessor and l.predecessor == k.predecessor) \
                    or k.lanelet_id == l.adj_left or k.lanelet_id == l.adj_right :
                    continue
                if l.polygon.shapely_object.intersects(k.polygon.shapely_object):
                    self.conflict_lanes[l.lanelet_id].append((k,
                            is_right(self.dense_lanes[l.lanelet_id][1], self.dense_lanes[k.lanelet_id][1])))

    def check_safety(self,action,action_copy):
        if self.l_id == 0: self.l_id = self.observation_collector.ego_lanelet.lanelet_id
        fall_back_kkd = self.compute_kappa_dot_dot(self.l_id, 0)
        fall_back_jd =  -0.1 if self.observation["v_ego"] > 2 else 0.1
        if action[0] > 0.8 or action[1] > 0.8 or action[0] < -0.8 or action[1] < -0.8:
            reward_for_safe_action = 0
            a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action,fall_back_kkd,self.l_id,self.nxt_id,0)
            if a == -2: a = fall_back_jd
            action = np.array([a,fall_back_kkd])
        else:
            if self.safety_verifier.safe_action_check(action[0],action[1], action_copy,0,self.l_id,self.nxt_id):
                reward_for_safe_action = 1
            else:
                a =  self.safety_verifier.find_feisable_jerk_dot(self.ego_action,action[1],self.l_id,self.nxt_id,0)
                if a != -2:
                    action[0] = a
                    reward_for_safe_action = 0.5
                else:
                    reward_for_safe_action = 0
                    actions : np.ndarray = self.observation["safe_actions"]
                    if np.all(actions == 0):
                        action = np.array([fall_back_jd,fall_back_kkd])
                    else:
                        if self.type == 1:
                            a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action,actions[3],self.l_id,self.nxt_id,0)
                            if a != -2: action = np.array([a,actions[3]])
                            else :
                                a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action,actions[0],self.l_id,self.nxt_id,0)
                                if a != -2: action = np.array([a,actions[0]])
                                else:
                                    a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action, actions[6], self.l_id, self.nxt_id,0)
                                    if a != -2: action = np.array([a, actions[6]])
                                    else:
                                        a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action, 0, self.l_id, self.nxt_id,0)
                                        if a != -2: action = np.array([a,0])
                        elif self.type == 2:
                            fcl_input = self.compute_kappa_dot_dot(self.l_id,self.nxt_id)
                            h = action[1]
                            a = -2
                            while a == -2 and fcl_input < 0.8:
                                a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action,fcl_input,self.l_id,self.nxt_id,0)
                                fcl_input += 0.5
                            if a == -2: a = action[0]
                            else:   h = fcl_input
                            action = np.array([a,h])
                        elif self.type == 3:
                            fcl_input = self.compute_kappa_dot_dot(self.l_id,self.nxt_id)
                            h = action[1]
                            a = -2
                            while a == -2 and fcl_input > -0.8:
                                a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action,fcl_input,self.l_id,self.nxt_id,0)
                                fcl_input -= 0.5
                            if a == -2: a = action[0]
                            else:   h = fcl_input
                            action = np.array([a,h])
                        else:
                            fcl_input = self.compute_kappa_dot_dot(self.l_id,self.nxt_id)
                            h = action[1]
                            a = -2
                            for i in range(11):
                                fcl_input = fcl_input + ((0.05 * ((i + 1) // 2)) * 1 if i % 2 != 0 else -1)
                                if not (-0.8 <= fcl_input <= 0.8):    continue
                                a = self.safety_verifier.find_feisable_jerk_dot(self.ego_action,fcl_input,self.l_id,self.nxt_id,0)
                                if a != -2: break
                            if a == -2: a = action[0]
                            else:   h = fcl_input
                            action = np.array([a,h])
        return reward_for_safe_action, action

    def step(self, action: Union[np.ndarray, State]) -> Tuple[np.ndarray, float, bool, dict]:
        in_conflict = self.observation_collector.conflict_zone.check_in_conflict_region(self.observation_collector._ego_vehicle)
        in_intersection = True if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys() else False
        action_copy = copy.deepcopy(self.ego_action)
        reward_for_safe_action, action = self.check_safety(action,action_copy)
        observation, reward, terminated, truncated, info = super().step(action)
        if self.observation_collector.ego_lanelet.lanelet_id not in self.past_ids:
            self.past_ids.append(self.observation_collector.ego_lanelet.lanelet_id)
        if reward_for_safe_action:
            if self.in_or_entering_intersection:
                reward += self.safe_reward(action, in_intersection, in_conflict)
        else:   reward -= 80
        self.observation = observation
        self.get_distance_to_lane_end()
        self.time_step += 1
        self.in_or_entering_intersection = self.intersection_check()
        self.safety_verifier.safeDistanceSet(self.observation_collector.ego_lanelet,self.in_or_entering_intersection,self.observation_collector._ego_state)
        observation_vector = self.apply_safety(observation, terminated)
        return observation_vector, reward, terminated, truncated, info

    def safe_reward(self, action, in_intersection, in_conflict):
        in_conflict_after = self.observation_collector.conflict_zone.check_in_conflict_region(self.observation_collector._ego_vehicle)
        reward_for_exiting_conflict_zone = 0
        penalty_for_slowing_down_in_conflict_zone = 0
        priority_non_compliance = 0
        entering_occupied_conflict_zone = 0
        not_slowing_occupied_conflict_zone = 0
        slowing_in_conflict_zone = 0
        if not in_intersection :
            if self.final_priority == 1:
                if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys():
                        priority_non_compliance = 1
                else:
                    if action[1] > 0:
                        priority_non_compliance = 0.5
        if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys():
            if not in_conflict_after:
                if in_conflict:
                    reward_for_exiting_conflict_zone = 1
                else: # in intersection not in conflict area is not entering conflict area
                    cl = self.conflict_lanes[self.observation_collector.ego_lanelet.lanelet_id]
                    for l,_ in cl:
                        if l.get_obstacles(self.scenario.obstacles):
                            if action[1] > 0:
                                not_slowing_occupied_conflict_zone = 1
                            break
            else: # entered conflict area
                if not in_conflict:
                    cl = self.conflict_lanes[self.observation_collector.ego_lanelet.lanelet_id]
                    for l,_ in cl:
                        if l.get_obstacles(self.scenario.obstacles):
                            entering_occupied_conflict_zone = 1
                            break
                else:
                    if action[1] < 0.1:
                        slowing_in_conflict_zone = 1
        return (60  *  reward_for_exiting_conflict_zone +
                -20  *  penalty_for_slowing_down_in_conflict_zone +
                -20  *  priority_non_compliance +
                -50  *  entering_occupied_conflict_zone +
                -10  *  not_slowing_occupied_conflict_zone +
                -10  *  slowing_in_conflict_zone)

    def intersection_check(self):
        nearest, farthest = self.observation["ego_distance_intersection"]
        if farthest < 0 or nearest <= 0 <= farthest:    return False
        return (self.observation["v_ego"]**2 / (2 * 5) < self.observation["distance_to_lane_end"] or
                self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys())

    def compute_kappa_dot_dot(self, l_id, nxt_id):
        center_points = self.dense_lanes[l_id][1]
        v = max(self.observation["v_ego"][0], 0.1)
        kap = self.observation["slip_angle"][0]
        kappa_dot = self.observation["global_turn_rate"][0]
        pos = self.observation_collector._ego_state.position.reshape(1, 2)
        theta = self.observation_collector._ego_state.orientation
        return compute_kappa_dot_dot_helper(theta, pos, v, self.prop_ego["a_lat_max"], kap, kappa_dot,
                                            self.precomputed_lane_polygons[l_id][0], center_points,
                                            self.dense_lanes[nxt_id][1] if nxt_id != 0 else None,
                                            self.precomputed_lane_polygons[nxt_id][0] if nxt_id != 0 else None)

    def neighbor_check(self, curr, other):
        """
            0 -> not a neighbor
            1 -> left neighbor
            2 -> right neighbor
        """
        curr_l = self.scenario.lanelet_network.find_lanelet_by_id(curr)
        if curr_l.adj_right_same_direction and curr_l.adj_right == other:   return True
        if curr_l.adj_left_same_direction and curr_l.adj_left == other: return True
        return False

    def wrong_lane_choice_fall_back(self,route_ids):
        last_index = route_ids.index(self.past_ids[len(self.past_ids) - 2])
        if self.neighbor_check(self.observation_collector.ego_lanelet.lanelet_id, route_ids[last_index]):
            l_id, nxt_id = route_ids[last_index], route_ids[last_index + 1] if len(route_ids) > last_index + 1 else 0
            fcl_input = self.compute_kappa_dot_dot(l_id, nxt_id)  # swerved into a neighbor
        else:  # got into a wrong successor
            l_id, nxt_id = route_ids[last_index + 1], route_ids[last_index + 2] if len(route_ids) > last_index + 2 else 0
            fcl_input = self.compute_kappa_dot_dot(l_id, nxt_id)
        self.type = 1
        kappa_dot_dots = np.linspace(fcl_input - 0.05, fcl_input + 0.05, 3)  # return to the route
        return l_id, nxt_id, kappa_dot_dots

    def lane_safety(self):
        """
            Returns an array of safe actions each as a tuple of (kdd,(jd_min,dj_max)).
            This does a max of 396 checks to build the Safe action set
        """
        At_safe_l = []
        route_ids = self.observation_collector.navigator.route.list_ids_lanelets
        if self.observation_collector.ego_lanelet.lanelet_id not in route_ids:
            if self.past_ids[len(self.past_ids)-2] in route_ids:
                if route_ids.index(self.past_ids[len(self.past_ids) - 2]) == len(route_ids) - 1:
                    return np.array([])  # simulation is done no next route
                l_id, nxt_id, kappa_dot_dots = self.wrong_lane_choice_fall_back(route_ids)
            else:   return np.array([]) # lost
        else :
            idx = route_ids.index(self.observation_collector.ego_lanelet_id)
            l_id, nxt_id = self.observation_collector.ego_lanelet.lanelet_id,route_ids[idx + 1] if len(route_ids) > idx + 1 else 0
            fcl_input = self.compute_kappa_dot_dot(l_id, nxt_id)
            if self.observation_collector.ego_lanelet.adj_left_same_direction:
                if self.observation_collector.ego_lanelet.adj_right_same_direction:
                    self.type = 4
                    kappa_dot_dots = np.linspace(-0.8, 0.8, 11) # left, current and right
                else:
                    self.type = 2
                    kappa_dot_dots = np.linspace(fcl_input - 0.1, 0.8, 11) # left and current
            elif self.observation_collector.ego_lanelet.adj_right_same_direction:
                self.type = 3
                kappa_dot_dots = np.linspace(-0.8, fcl_input + 0.1, 11) # current and right
            else:
                self.type = 1
                kappa_dot_dots = np.linspace(fcl_input-0.05, fcl_input+0.05, 3) # only current lane
        for kdd in kappa_dot_dots:
            safe_min, safe_max = self.safety_verifier.find_safe_jerk_dot(self.ego_action, kdd,l_id,nxt_id)
            if safe_min <= safe_max:    At_safe_l.extend([kdd, safe_min, safe_max])
            else: At_safe_l.extend([kdd,0,0])
            #At_safe_l.extend([kdd])
        self.l_id, self.nxt_id = l_id, nxt_id
        return np.array(At_safe_l, dtype=object)

    def priority_condition(self, lane : Lanelet,obstacle : Obstacle, D_m=30.0, T_a=3.0):
        """
            d_c_i : float Distance of vehicle i to the conflict area (meters)
            v_i : float Velocity of vehicle i (m/s)
            D_m : float Monitoring range (default: 30 m)
            T_a : float Arrival time threshold (default: 3 s)
        Returns:
            bool    True if vehicle i has priority relevance
        """
        pos = obstacle.state_at_time(self.time_step).position
        dist = 100000
        curr = None
        c : np.ndarray = self.dense_lanes[lane.lanelet_id][1]
        for cp in c:
            if math.dist(pos, cp) < dist:
                dist = math.dist(pos, cp)
                curr = cp
        d_c_i = traveled_distance(c[::-1],curr)
        return (d_c_i <= D_m) or (d_c_i / obstacle.state_at_time(self.time_step).velocity <= T_a)

    def priority(self, lane : Lanelet, pre_intersection_lanes):
        """
            Updates priority with:
                1 --> yield
                0 --> priority
        """
        def dir_priority(incoming_lanes):
            intersection_lanes_obs = []
            for l in incoming_lanes:
                obs = self.scenario.lanelet_network.find_lanelet_by_id(l).get_obstacles(self.scenario.obstacles, self.time_step)
                obs = self.safety_verifier.sort_obstacles_in_lane(l, obs)
                if not obs: continue
                intersection_lanes_obs.append((self.scenario.lanelet_network.find_lanelet_by_id(l), obs[-1]))
            for k in intersection_lanes_obs:
                if self.priority_condition(*k):
                    return True
            return False
        self.final_priority = 1 if dir_priority(pre_intersection_lanes[lane.lanelet_id]) else 0

    def get_per_intersection_lanes_update_priority(self, lane : Lanelet):
        pre_intersection_lanes: defaultdict[int, list[int]] = defaultdict(list)
        for k,v in self.conflict_lanes.items():
            if k == lane.lanelet_id:
                for conflict_lane,right in v:
                    if right:
                        for l in conflict_lane.predecessor:
                            if l not in pre_intersection_lanes[k]:
                                pre_intersection_lanes[k].append(l)
        self.priority(lane, pre_intersection_lanes)
        return pre_intersection_lanes

    def intersection_safety(self):
        """
            Returns an array of safe actions each as a tuple of (kdd,(jd_min,jd_max))
        """
        At_safe_in : List[float] = []
        route_ids = self.observation_collector.navigator.route.list_ids_lanelets
        if self.observation_collector.ego_lanelet.lanelet_id not in route_ids:
            if self.past_ids[len(self.past_ids) - 2] in route_ids:
                if route_ids.index(self.past_ids[len(self.past_ids) - 2]) == len(route_ids) - 1:
                    return np.array([])  # simulation is done no next route
                l_id, nxt_id, kappa_dot_dots = self.wrong_lane_choice_fall_back(route_ids)
            else:   return np.array([]) # lost
        else:
            l_id = self.observation_collector.ego_lanelet.lanelet_id
            curr_index = route_ids.index(l_id)
            if curr_index == len(route_ids) - 1:
                return np.array([]) # simulation is done no next route
            nxt_id = route_ids[curr_index + 1]
            nxt_lane = self.scenario.lanelet_network.find_lanelet_by_id(nxt_id)
            if not self.pre_intersection_lanes:
                self.pre_intersection_lanes = self.get_per_intersection_lanes_update_priority(nxt_lane)
            else:
                self.priority(nxt_lane, self.pre_intersection_lanes)
            if self.prop_ego["ego_length"] / 2 >= self.observation["distance_to_lane_end"]:
                fcl_input = self.compute_kappa_dot_dot(nxt_lane.lanelet_id,route_ids[curr_index + 2] if len(route_ids) > curr_index + 2 else 0)
                self.final_priority = 1
            else:
                if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys():
                    self.final_priority = 1
                fcl_input = self.compute_kappa_dot_dot(self.observation_collector.ego_lanelet.lanelet_id,route_ids[curr_index + 1] if len(route_ids) > curr_index + 1 else 0)
            self.type = 1
            kappa_dot_dots = np.linspace(fcl_input - 0.05, fcl_input + 0.05, 3)  # only current lane
        for kdd in kappa_dot_dots:
            safe_min, safe_max = self.safety_verifier.find_safe_jerk_dot(self.ego_action, kdd, l_id, nxt_id)
            if safe_min <= safe_max:    At_safe_in.extend([kdd, safe_min, safe_max])
            else:   At_safe_in.extend([kdd , 0, 0])
            #At_safe_in.extend([kdd])
        self.l_id, self.nxt_id = l_id, nxt_id
        return np.array(At_safe_in, dtype=object)

    @property
    def observation_space(self):
        return spaces.Box(low=self.new_low, high=self.new_high, dtype=self.observation_collector.observation_space.dtype)
