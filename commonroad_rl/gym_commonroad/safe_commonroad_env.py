import math
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from scipy.interpolate import interp1d

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import shapely
from gymnasium import spaces

from commonroad_rl.gym_commonroad.action.action import Action
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.state import State
from commonroad_rl.gym_commonroad.action import ContinuousVehicle
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad.scenario.obstacle import Obstacle
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.utils.stanley_controller_piecewise import StanleyController
from commonroad_clcs.pycrccosy import CartesianProjectionDomainError

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

class SafetyVerifier:

    def __init__(self, scenario: Scenario, prop_ego, precomputed_lane_polygons, dense_lanes):
        self.l_id = None
        self.r_id = None
        self.scenario = scenario
        self.prop_ego = prop_ego
        self.in_or_entering_intersection = False
        self.safe_set : List[Tuple[List[Tuple[int,int,float,float,float]],Lanelet]] = []
        self.precomputed_lane_polygons: Dict[int, Tuple[CurvilinearCoordinateSystem,
                    np.ndarray, np.ndarray, np.ndarray]] = precomputed_lane_polygons
        self.time_step = -1
        self.lane_width = 5
        self.ego_lanelet = None
        self.dense_lanes : Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = dense_lanes

    def get_reachable_lanes(self) -> List[Lanelet]:
        lanes = [self.ego_lanelet]
        for l_id in self.ego_lanelet.successor:
            lanes.append(self.scenario.lanelet_network.find_lanelet_by_id(l_id))
        if not self.in_or_entering_intersection:
            if self.ego_lanelet.adj_left_same_direction:
                lanes.append(self.scenario.lanelet_network.find_lanelet_by_id(self.ego_lanelet.adj_left))
            if self.ego_lanelet.adj_right_same_direction:
                lanes.append(self.scenario.lanelet_network.find_lanelet_by_id(self.ego_lanelet.adj_right))
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

    @staticmethod
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

    def obs_start_end_index(self, obs : Obstacle, l_id) -> list[int]:
        obs_state = obs.state_at_time(self.time_step)
        shape = obs.occupancy_at_time(self.time_step).shape
        left,center,right = self.dense_lanes[l_id]
        c_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1], obs_state.orientation,
                                                      shape.length, shape.width, center)
        l_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1],
                                                      obs_state.orientation,shape.length, shape.width, left)
        r_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1],
                                                     obs_state.orientation,shape.length, shape.width, right)
        pts = []
        for k in (c_pts, l_pts, r_pts):
            if k is not None:   pts.extend(k)
        if not pts: return []
        start = min(pts)
        end = max(pts)
        if start == end:    return [start]
        else:   return [start, end]

    def get_end_collision_free_area(self, lane : Lanelet, center, pt : list[int], preceding_v,depth = 0,max_depth = 3):
        if depth > max_depth:
            return center[pt[1]:], lane, preceding_v, 0.0, 0.0
        successors = lane.successor
        if len(successors) == 0:
            return center[pt[1] : len(center)], lane, preceding_v, self.prop_ego["v_max"] , 0
        def get_closest_obstacle_lane_velocity_distance(ls: Lanelet):
            lso = ls.get_obstacles(self.scenario.obstacles, self.time_step)
            if len(lso) == 0:
                r_min = 1.0 / self.kappa(self.dense_lanes[ls.lanelet_id][1])
                v_crit = np.sqrt(r_min * self.prop_ego["a_lat_max"])
                return min(v_crit,self.prop_ego["v_max"]), traveled_distance(self.dense_lanes[ls.lanelet_id][1], self.dense_lanes[ls.lanelet_id][1][-1]), False
            else:
                lso = self.sort_obstacles_in_lane(ls.lanelet_id, lso)
                lobs_state = lso[0].state_at_time(self.time_step)
                pts = self.obs_start_end_index(lso[0],ls.lanelet_id)
                if not pts:
                    s_centers = self.precomputed_lane_polygons[ls.lanelet_id][1]
                    p = np.asarray(lobs_state.position).reshape(1, 2)
                    closest_centerpoint = np.linalg.norm(self.dense_lanes[ls.lanelet_id][1] - p, axis=1).argmin()
                    s = s_centers[closest_centerpoint]
                    return lobs_state.velocity, s, True
                return lobs_state.velocity, traveled_distance(self.dense_lanes[ls.lanelet_id][1],
                                                              self.dense_lanes[ls.lanelet_id][1][pts[0]]), True
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

    def get_lane_collision_free_areas(self, lane : Lanelet):
        #print("get_lane_collision_free_areas")
        C = []
        center = self.dense_lanes[lane.lanelet_id][1]
        obs = lane.get_obstacles(self.scenario.obstacles, self.time_step)
        # empty lane with no vehicle entering or exiting it
        if len(obs) == 0:
            #print("Real Empty lane")
            return [self.get_end_collision_free_area(lane, center, [0,0], 0)]
        i = 0
        obs = self.sort_obstacles_in_lane(lane.lanelet_id, obs)
        obs_state = obs[i].state_at_time(self.time_step)
        pts = self.obs_start_end_index(obs[i], lane.lanelet_id)
        if len(pts) == 0:
            s_centers = self.precomputed_lane_polygons[lane.lanelet_id][1]
            p = np.asarray(obs_state.position).reshape(1, 2)
            closest_centerpoint = np.linalg.norm(self.dense_lanes[lane.lanelet_id][1] - p, axis=1).argmin()
            s = s_centers[closest_centerpoint]
            start = s + obs[0].obstacle_shape.length/2
            end = s - obs[0].obstacle_shape.length/2
            if start == end: pts = np.array(np.linalg.norm(s_centers - start).argmin())
            else: pts = np.array([np.linalg.norm(s_centers - start).argmin(),
                   np.linalg.norm(s_centers - end).argmin()])
        preceding_v = obs_state.velocity
        if len(pts) == 1:
            pt = pts[0]
            #print("first obs points 1")
        else:
            #print("first obs points 2")
            cps = center[0 : pts[0] + 1]
            C.append((cps,lane, 0.0, preceding_v, 0.0)) # add first collision free area
            pt = pts[1]
        for i in range(len(obs) - 2):
            obs_state = obs[i + 1].state_at_time(self.time_step)
            pts = self.obs_start_end_index(obs[i + 1], lane.lanelet_id)
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

    def sort_obstacles_in_lane(self, l_id : int ,obs : List[Obstacle]) -> List[Obstacle]:
        obs_with_center : List[Tuple[Obstacle, float]] = []
        ct= self.precomputed_lane_polygons[l_id][0]
        for ob in obs:
            pos = (ob.state_at_time(self.time_step).position[0], ob.state_at_time(self.time_step).position[1])
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

    def safeDistanceSetForSection(self, xi:float, yi:float, v_i:float, xj:float, yj:float, v_j:float,cp, l_id, distance_to_add) \
            -> List[Tuple[float,float,float,float,float]] :
        """
            This function takes the position and velocity of two Obstacles and the collision
            free area between them (as a part of the lane i.e. left-,center- and right points)
            and returns all the possible (s,v,d) states of an ego vehicle to be in this area
            without collision, with:
               - s -> All the center points in the lane that the vehicle can be on
               - v -> Velocities of the ego vehicle for each center
               - d -> The Area to leave on edges for safe bounds in the lane
        """
        #print(xi , "   ",yi, "   ",v_i, "   ",xj, "   ",yj, "   ",v_j)
        if traveled_distance(cp,cp[-1]) < self.prop_ego["ego_length"]:
            return []
        ct, s_centers, _, _ = self.precomputed_lane_polygons[l_id]
        txi, _ = ct.convert_to_curvilinear_coords(xi, yi)
        txj, _ = ct.convert_to_curvilinear_coords(xj, yj)
        txj += distance_to_add
        a_lat_max, a_lon_max, w, l, delta_react = (self.prop_ego["a_lat_max"], self.prop_ego["a_lon_max"],
                self.prop_ego["ego_width"], self.prop_ego["ego_length"], self.prop_ego["delta_react"])
        k = self.kappa(cp)
        if k == 0:  r_min = math.inf
        else:   r_min = 1.0 / k
        v_crit = min(np.sqrt(r_min * a_lat_max),self.prop_ego["v_max"])
        # s >= s_i + Δ_safe(v, i)
        # s <= s_j - Δ_safe(v, j)
        vs = np.linspace(0, v_crit, 50)
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
                # Changed from the original formula d_lim = r_min - sqrt(r_min^2 - l_front^2 + 0.5w),
                # as the car shape will be handled in the action testing part with Shapely, replaced it
                # with a lookahead distance equal to the velocity as we do this for each timestamp
                #d_lim = r_min - np.sqrt(max(r_min ** 2 - (v/2) ** 2, 0))
                #shrink_per_side = max(0, 2.5 - d_lim)
                #lane_half_width = self.lane_width / 2
                #shrink_per_side = np.clip(shrink_per_side, 0, lane_half_width - 0.1)
                safe_states.append((min(start,end), max(start,end), v, 0, 0))
        return safe_states

    def union_safe_set(self, ll: Lanelet, safe_set_list_left : List[Tuple[int,int,float,float,float]]
                            , rl : Lanelet, safe_set_list_right :List[Tuple[int,int,float,float,float]]):
        ct,ls, _, _ = self.precomputed_lane_polygons[ll.lanelet_id]
        nls = []
        nrs = []
        for s in safe_set_list_left:
            c1, c2, lv, dl, _ = s
            l_start = int(ls[c1])
            l_end = int(ls[c2])
            for c in safe_set_list_right:
                c1, c2, rv, _, dr = c
                _, rcp, _ = self.dense_lanes[rl.lanelet_id]
                r_start, _ = ct.convert_to_curvilinear_coords(rcp[c1][0],rcp[c1][1])
                r_end, _ = ct.convert_to_curvilinear_coords(rcp[c2][0],rcp[c2][1])
                start = max(l_start, r_start)
                end = min(l_end, r_end)
                if start < end:
                    nls.append((start, end, lv, dl, -4))
                    nrs.append((start, end, rv, -4, dr))
        return [(nls,ll), (nrs,rl)]

    def safeDistanceSet(self, ego_lanelet : Lanelet, in_or_entering_intersection):
        """
            Constructs the list of the admissible velocities and locations for the ego Vehicle.
                We must first construct Collision free areas as:
                    - Centerpoints between the two Obstacles,
                    - Lane
                    - Preceding and succeeding Obstacles velocities
                    - Distance between the obstacle with the smallest ttc in all next lanes and the current lane end.
                Returns Safe states as a List of Tuples each :
                    - List of Tuples each:
                        - First and Last Centerpoints to be between with this Velocity
                        - Velocity
                        - Left Safety bound based on section Curvature
                        - Right Safety bound based on section Curvature
                    - Lane
        """
        self.ego_lanelet = ego_lanelet
        self.l_id = self.ego_lanelet.adj_left if self.ego_lanelet.adj_left_same_direction else None
        self.r_id = self.ego_lanelet.adj_right if self.ego_lanelet.adj_right_same_direction else None
        self.time_step += 1
        self.in_or_entering_intersection = in_or_entering_intersection
        S : List[Tuple[List[Tuple[float,float,float,float,float]],Lanelet]] = []
        C = []
        #print("lanes to check")
        #for l in self.get_reachable_lanes():
            #print(l.lanelet_id)
        for lane in self.get_reachable_lanes():
            C.extend(self.get_lane_collision_free_areas(lane))
        for c in C:
            cp, l, vi, vj, d = c
            S.append((self.safeDistanceSetForSection(cp[0][0],cp[0][1],vi,cp[-1][0],cp[-1][1],vj,cp,l.lanelet_id,d),l))
        #   For lane change we must have parts where the safe bounds don't exist,
        #   we do this by expanding the bounds into the adj lane when two safe states area are next to each other.
        #   We only do unions to the left side i.e. left lane with ego and ego with right lane.
        es = []
        for s in S:
            k, lane = s
            if lane == self.ego_lanelet:
                es.extend(k)
        if not self.in_or_entering_intersection:
            if self.l_id:
                #print("unioning")
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
        """print("Printing Safe sets : ")
        print("--------------------------------------------------------------------------------------------------------------")
        print(S)
        print("--------------------------------------------------------------------------------------------------------------")"""

    def safe_action_check(self, jd, kdd, ego_action : Action):
        curr_vehicle: ContinuousVehicle = ego_action.vehicle
        new_vehicle_state = curr_vehicle.get_new_state(ego_action.rescale_action(np.array([jd, kdd])), ego_action.action_base)
        p = new_vehicle_state.position
        nv = new_vehicle_state.velocity
        W, L = self.prop_ego["ego_width"], self.prop_ego["ego_length"]
        rect = Polygon([(-L / 2, -W / 2), (-L / 2, W / 2), (L / 2, W / 2), (L / 2, -W / 2)])
        rect = rotate(rect, new_vehicle_state.orientation * 180 / math.pi, origin=(0, 0), use_radians=False)
        rect = translate(rect, xoff=p[0], yoff=p[1])
        for l in self.get_reachable_lanes():
            ct, _, lp, rp = self.precomputed_lane_polygons[l.lanelet_id]
            for s in self.safe_set:
                k, lane = s
                if lane.lanelet_id == l.lanelet_id:
                    for start, end, v, dl, dr in k:
                        if start == end or not (v - 1 <= nv <= v + 1) : continue
                        def in_safe_space(left_points : np.ndarray, right_points: np.ndarray):
                            left_bound = left_points[start: end + 1]
                            right_bound = right_points[start: end + 1]
                            lb, rb = [], []
                            i = 0
                            while i < len(left_bound):
                                try :
                                    lb.append(ct.convert_to_cartesian_coords(left_bound[i][0], left_bound[i][1] - dl + 0.1))
                                    rb.append(ct.convert_to_cartesian_coords(right_bound[i][0], right_bound[i][1] - dr - 0.1))
                                except CartesianProjectionDomainError:
                                    print("Cartesian projection domain error")
                                    pass
                                i+=1
                            valid_road_polygons = [Polygon(lb + rb[::-1]).buffer(0)]
                            if end == len(left_bound):
                                for s_id in lane.successor:
                                    valid_road_polygons.append(self.scenario.lanelet_network.find_lanelet_by_id(s_id).polygon.shapely_object.buffer(0))
                            elif start == 0:
                                for p_id in lane.predecessor:
                                    valid_road_polygons.append(self.scenario.lanelet_network.find_lanelet_by_id(p_id).polygon.shapely_object.buffer(0))
                            lane_polygon = shapely.union_all(valid_road_polygons)
                            return lane_polygon.contains(rect)
                        if in_safe_space(lp, rp):   return True
        return False


class SafetyLayer(CommonroadEnv):

    def __init__(self, meta_scenario_path=PATH_PARAMS["meta_scenario"],
                 train_reset_config_path=PATH_PARAMS["train_reset_config"],
                 test_reset_config_path=PATH_PARAMS["test_reset_config"],
                 visualization_path=PATH_PARAMS["visualization"], logging_path=None, test_env=False, play=False,
                 config_file=PATH_PARAMS["configs"]["commonroad-v1"], logging_mode=1, **kwargs) -> None:
        super().__init__(meta_scenario_path, train_reset_config_path, test_reset_config_path, visualization_path,
                         logging_path, test_env, play, config_file, logging_mode, **kwargs)
        #print("init")
        self.observation = None
        self.past_ids = []
        self.prop_ego = {"ego_length" : 3, "ego_width" : 1.5 , "a_lat_max" : 9.0, "a_lon_max" : 11.5, "delta_react" : 0.5}
        self.time_step = 0
        self.lane_width = 5
        self.last_relative_heading = 0
        self.prop_ego["v_max"] = 45
        self.final_priority = -1
        self.intersection_lanes : List[Lanelet] = []
        self.conflict_lanes : defaultdict[int, List[Tuple[Lanelet, bool]]] = defaultdict(list)
        self.stanley_controller = StanleyController(3,1,0.05,0.3, 3.14, 2.9)
        self.pre_intersection_lanes = None
        self.precomputed_lane_polygons = {}
        self.dense_lanes : Dict[int,Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.safety_verifier = None
        self.in_or_entering_intersection = False
        self.new_low = np.concatenate([self.observation_collector.observation_space.low.astype(np.float32),
                                        np.full(35, -1.0, dtype=np.float32)])
        self.new_high = np.concatenate([self.observation_collector.observation_space.high.astype(np.float32),
                                        np.full(35, 1.0, dtype=np.float32)])

    def pack_observation(self, observation_dict):
        def pack_orig():
            observation_vector = np.zeros(self.observation_collector.observation_space.shape)
            index = 0
            for k in observation_dict.keys():
                if k in ["safe_actions", "final_priority", "distance_to_lane_end"] : continue
                size = np.prod(self.observation_dict[k].shape)
                try:
                    observation_vector[index: index + size] = observation_dict[k].flat
                except ValueError as s:
                    print(size , "   ", index, "   " , k)
                    print(s)
                    exit(1)
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

    def reset(self, seed=None, options: Optional[dict] = None, benchmark_id=None, scenario: Scenario = None,
              planning_problem: PlanningProblem = None) -> np.ndarray:
        self.past_ids = []
        initial_observation, info = super().reset(seed, options, benchmark_id, scenario, planning_problem)
        print(self.observation_collector._scenario.scenario_id)
        self.past_ids.append(self.observation_collector.ego_lanelet.lanelet_id)
        self.time_step = 0
        self.compute_lane_sides_and_conflict()
        _, center_points, _ = self.dense_lanes[self.observation_collector.ego_lanelet.lanelet_id]
        ego_pos = np.asarray(self.observation_collector._ego_state.position).reshape(1 ,2)
        closest_centerpoint = center_points[np.linalg.norm(center_points - ego_pos, axis=1).argmin()]
        self.safety_verifier = SafetyVerifier(self.scenario, self.prop_ego, self.precomputed_lane_polygons, self.dense_lanes)
        self.observation = initial_observation.copy()
        self.observation["distance_to_lane_end"] = np.array([traveled_distance(center_points[::-1],closest_centerpoint)], dtype= object)
        self.in_or_entering_intersection = self.intersection_check()
        self.safety_verifier.safeDistanceSet(self.observation_collector.ego_lanelet,self.in_or_entering_intersection)
        self.pre_intersection_lanes = None
        if self.in_or_entering_intersection:
            actions = self.intersection_safety()
        else:
            self.pre_intersection_lanes = None
            self.final_priority = -1
            actions = self.lane_safety()
        #print(actions)
        if actions.size > 33:   actions = actions[:33]
        elif actions.size < 33:   actions = np.pad(actions, (0, 33 - actions.size), mode='constant', constant_values=0)
        self.observation["safe_actions"] = actions
        self.observation["final_priority"] = np.array([self.final_priority], dtype=object)
        # for k in self.observation.keys():   print(k, " : ", self.observation[k])
        observation_vector = self.pack_observation(initial_observation)
        """import matplotlib.pyplot as plt
        from commonroad.visualization.mp_renderer import MPRenderer
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        rnd.draw_params.time_begin = self.time_step
        self.observation_collector._scenario.draw(rnd)
        self.planning_problem.draw(rnd)
        rnd.render()
        plt.show(block=False)"""
        #print("")
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
                fx = interp1d(s_old, curve[:, 0], kind="linear", fill_value="extrapolate")
                fy = interp1d(s_old, curve[:, 1], kind="linear", fill_value="extrapolate")
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
            if type(center_dense) is not np.ndarray:
                print("Center dense is not a numpy array")
                print(l.center_vertices)
                print(l.left_vertices)
                print(l.right_vertices)
                print(to_remove)
            self.dense_lanes[l.lanelet_id] = (left_dense, center_dense, right_dense)
            left = np.array(left)
            right = np.array(right)
            s_centers = np.array([])
            for x, y in center_dense:
                try:
                    s, d = ct.convert_to_curvilinear_coords(x, y)
                    s_centers = np.append(s_centers, s)
                except CartesianProjectionDomainError:
                    print("error in converting")
                    pass
            self.precomputed_lane_polygons[l.lanelet_id] = (ct, s_centers, left, right)
        for l in self.scenario.lanelet_network.lanelets:
            for k in self.scenario.lanelet_network.lanelets:
                if l.lanelet_id == k.lanelet_id or (l.predecessor and k.predecessor and l.predecessor == k.predecessor) \
                    or k.lanelet_id == l.adj_left or k.lanelet_id == l.adj_right :
                    continue
                if l.polygon.shapely_object.intersects(k.polygon.shapely_object):
                    self.conflict_lanes[l.lanelet_id].append((k,
                            is_right(self.dense_lanes[l.lanelet_id][1], self.dense_lanes[k.lanelet_id][1])))

    def step(self, action: Union[np.ndarray, State]) -> Tuple[np.ndarray, float, bool, dict]:
        reward_for_safe_action = 0
        in_conflict = self.observation_collector.conflict_zone.check_in_conflict_region(self.observation_collector._ego_vehicle)
        in_intersection = True if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys() else False
        if self.safety_verifier.safe_action_check(action[1],action[0], self.ego_action):
            print("safe action")
            reward_for_safe_action = 1
        else:
            a,b =  self.find_safe_jerk_dot(action[0])
            if a<=b:
                action[1] = min(0,b) if self.observation["v_ego"] > 0 else max(0,a)
                reward_for_safe_action = 0.5
                print("half safe action")
            else:
                print("unsafe action : ", action)
                kdd = self.compute_kappa_dot_dot(self.dense_lanes[self.observation_collector.ego_lanelet.lanelet_id][1])
                a,b = self.find_safe_jerk_dot(action[0])
                if a <= b:
                    action[0] = kdd
                    if in_conflict: action[1] = b
                    else:   action[1] = b if self.observation["a_ego"] > 0 else a
                else:
                    print("new action : ", action)
        observation, reward, terminated, truncated, info = super().step(action)
        if self.observation_collector.ego_lanelet.lanelet_id not in self.past_ids:
            self.past_ids.append(self.observation_collector.ego_lanelet.lanelet_id)
        if reward_for_safe_action:
            if self.in_or_entering_intersection:
                reward += self.safe_reward(action, in_intersection, in_conflict)
        else:   reward -= 800
        _, center_points, _ = self.dense_lanes[self.observation_collector.ego_lanelet.lanelet_id]
        ego_pos = np.asarray(self.observation_collector._ego_state.position).reshape(1, 2)
        closest_centerpoint = center_points[np.linalg.norm(center_points - ego_pos, axis=1).argmin()]
        self.observation = observation
        self.observation["distance_to_lane_end"] = np.array([traveled_distance(center_points[::-1],closest_centerpoint)], dtype= object)
        self.time_step += 1
        self.in_or_entering_intersection = self.intersection_check()
        self.safety_verifier.safeDistanceSet(self.observation_collector.ego_lanelet,self.in_or_entering_intersection)
        if self.in_or_entering_intersection:
            actions =self.intersection_safety()
        else:
            self.pre_intersection_lanes = None
            self.final_priority = -1
            actions = self.lane_safety()
        #for k in observation.keys():    print(k , " : ", observation[k])
        #print("actions : ", actions)
        if actions.size > 33:   actions = actions[:33]
        elif actions.size < 33:   actions = np.pad(actions, (0, 33 - actions.size), mode='constant', constant_values=0)
        self.observation["safe_actions"] = actions
        self.observation["final_priority"] = np.array([self.final_priority], dtype= object)
        #for k in self.observation.keys():   print(k, " : ", self.observation[k])
        observation_vector = self.pack_observation(observation)
        if terminated:
            #print(info)
            print(self.termination_reason)
        #print(self.observation_collector._ego_state.position)
        """import matplotlib.pyplot as plt
        from commonroad.visualization.mp_renderer import MPRenderer
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        rnd.draw_params.time_begin = self.time_step
        self.observation_collector._scenario.draw(rnd)
        self.planning_problem.draw(rnd)
        rnd.render()
        plt.show(block=False)"""
        print(actions)
        print("")
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
        return (600  *  reward_for_exiting_conflict_zone +
                -200  *  penalty_for_slowing_down_in_conflict_zone +
                -200  *  priority_non_compliance +
                -500  *  entering_occupied_conflict_zone +
                -100  *  not_slowing_occupied_conflict_zone +
                -100  *  slowing_in_conflict_zone)

    def intersection_check(self):
        a_max = 5
        nearest, farthest = self.observation["ego_distance_intersection"]
        if farthest < 0 or nearest <= 0 <= farthest:
            #print("intersection check nearest farthest")
            return False
        if (self.observation["v_ego"]**2 / (2 * a_max) < self.observation["distance_to_lane_end"] or
                self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys()):
            #print("intersection check True")
            #if self.observation["v_ego"]**2 / (2 * a_max) < self.observation["distance_to_lane_end"]:
                #print("near lane end")
            #else:
                #print("in intersection")
            return True
        #print("intersection check False")
        return False

    def compute_kappa_dot_dot(self, center_points):
        v = max(self.observation["v_ego"], 0.1)
        kappa_ref = self.safety_verifier.kappa(center_points)
        kappa_max = self.prop_ego["a_lat_max"] / (v ** 2)
        kappa_ref = np.clip(kappa_ref, -kappa_max, kappa_max)
        kappa = self.observation["slip_angle"]
        kappa_dot = self.observation["global_turn_rate"]
        """omega_n = 1.5 / dt
        zeta = 0.7

        kp = omega_n ** 2
        kd = 2 * zeta * omega_n
        kp = 4
        kd = 2"""
        kappa_ddot = 4.0 * (kappa_ref - kappa) - 2.0 * kappa_dot
        return float(np.clip(kappa_ddot / 20, -1.0, 1.0))

    def find_safe_jerk_dot(self, kappa_ddot):
        """
            Binary search for the min and max jerk_dot for given kappa_dot_dot.
            Using the binary search made it has constant complexity of 18 iterations for each 36 checks in total
        """
        low, high = -1, 1
        while high - low > 1e-5:
            mid = (low + high) / 2
            if self.safety_verifier.safe_action_check(mid, kappa_ddot, self.ego_action):   high = mid
            else:   low = mid
        safe_min = high
        low, high = -1, 1
        while high - low > 1e-5:
            mid = (low + high) / 2
            if self.safety_verifier.safe_action_check(mid, kappa_ddot, self.ego_action):   low = mid
            else:   high = mid
        safe_max = low
        return safe_min, safe_max

    def neighbor_check(self, curr, other):
        """
            0 -> not a neighbor
            1 -> left neighbor
            2 -> right neighbor
        """
        curr_l = self.scenario.lanelet_network.find_lanelet_by_id(curr)
        if curr_l.adj_right_same_direction:
            if curr_l.adj_right == other:
                return True
        if curr_l.adj_left_same_direction:
            if curr_l.adj_left == other:
                return True
        return False

    def lane_safety(self):
        """
            Returns an array of safe actions each as a tuple of (kdd,(jd_min,dj_max)).
            This does a max of 396 checks to build the Safe action set
        """
        At_safe_l = []
        route_ids = self.observation_collector.navigator.route.list_ids_lanelets
        if self.observation_collector.ego_lanelet.lanelet_id not in route_ids:
            #print(route_ids)
            #print("past lanes : ", self.past_ids)
            if self.past_ids[len(self.past_ids)-2] in route_ids:
                last_index = route_ids.index(self.past_ids[len(self.past_ids) - 2])
                if last_index == len(route_ids) - 1:
                    #print("returned empty list")
                    return np.array([])  # simulation is done no next route
                if self.neighbor_check(self.observation_collector.ego_lanelet.lanelet_id, route_ids[last_index]) :
                    fcl_input = self.compute_kappa_dot_dot(self.dense_lanes[route_ids[last_index]][1]) # swerved into a neighbor
                else: # got into a wrong successor
                    fcl_input = self.compute_kappa_dot_dot(self.dense_lanes[route_ids[last_index + 1]][1])
                kappa_dot_dots = np.linspace(fcl_input - 0.05, fcl_input + 0.05, 3)  # return to the route
            else: # lost
                #print("Lost")
                return np.array([])
        else :
            fcl_input = self.compute_kappa_dot_dot(self.dense_lanes[self.observation_collector.ego_lanelet.lanelet_id][1])
            if self.observation_collector.ego_lanelet.adj_left_same_direction:
                if self.observation_collector.ego_lanelet.adj_right_same_direction:
                    kappa_dot_dots = np.linspace(-0.8, 0.8, 11) # left, current and right
                else:
                    kappa_dot_dots = np.linspace(fcl_input - 0.1, 0.8, 11) # left and current
            elif self.observation_collector.ego_lanelet.adj_right_same_direction:
                kappa_dot_dots = np.linspace(-0.8, fcl_input + 0.1, 11) # current and right
            else:
                kappa_dot_dots = np.linspace(fcl_input-0.05, fcl_input+0.05, 3) # only current lane
        for kdd in kappa_dot_dots:
            safe_min, safe_max = self.find_safe_jerk_dot(kdd)
            if safe_min <= safe_max:    At_safe_l.extend([kdd, safe_min, safe_max])
            else: At_safe_l.extend([kdd,0,0])
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
            #print(route_ids)
            #print("past lanes : ", self.past_ids)
            if self.past_ids[len(self.past_ids) - 2] in route_ids:
                last_index = route_ids.index(self.past_ids[len(self.past_ids) - 2])
                if last_index == len(route_ids) - 1:
                    #print("returned empty list")
                    return np.array([])  # simulation is done no next route
                if self.neighbor_check(self.observation_collector.ego_lanelet.lanelet_id, route_ids[last_index]):
                    fcl_input = self.compute_kappa_dot_dot(
                        self.dense_lanes[route_ids[last_index]][1])  # swerved into a neighbor
                else:  # got into a wrong successor
                    fcl_input = self.compute_kappa_dot_dot(self.dense_lanes[route_ids[last_index + 1]][1])
            else:  # lost
                #print("Lost")
                return np.array([])
        else:
            curr_index = route_ids.index(self.observation_collector.ego_lanelet.lanelet_id)
            if curr_index == len(route_ids) - 1:
                #print("returned empty list")
                return np.array([]) # simulation is done no next route
            nxt_id = route_ids[curr_index + 1]
            nxt_lane = self.scenario.lanelet_network.find_lanelet_by_id(nxt_id)
            if not self.pre_intersection_lanes:
                self.pre_intersection_lanes = self.get_per_intersection_lanes_update_priority(nxt_lane)
            else:
                self.priority(nxt_lane, self.pre_intersection_lanes)
            if self.prop_ego["ego_length"] / 2 >= self.observation["distance_to_lane_end"]:
                fcl_input = self.compute_kappa_dot_dot(self.dense_lanes[nxt_lane.lanelet_id][1])
                self.final_priority = 1
            else:
                if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys():
                    self.final_priority = 1
                fcl_input = self.compute_kappa_dot_dot(self.dense_lanes[self.observation_collector.ego_lanelet.lanelet_id][1])
        kappa_dot_dots = np.linspace(fcl_input - 0.05, fcl_input + 0.05, 3)  # only current lane
        for kdd in kappa_dot_dots:
            safe_min, safe_max = self.find_safe_jerk_dot(kdd)
            if safe_min <= safe_max:
                At_safe_in.extend([kdd, safe_min, safe_max])
            else:
                At_safe_in.extend([kdd , 0, 0])
        return np.array(At_safe_in, dtype=object)

    @property
    def observation_space(self):
        return spaces.Box(low=self.new_low, high=self.new_high, dtype=self.observation_collector.observation_space.dtype)

