import math
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Optional, Union, Dict

from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate, translate

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
from commonroad.geometry.polyline_util import resample_polyline_with_distance
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

    def __init__(self, scenario: Scenario, prop_ego, precomputed_lane_polygons):
        self.scenario = scenario
        self.v_max = 45
        self.prop_ego = prop_ego
        self.in_or_entering_intersection = False
        self.safe_set : List[Tuple[List[Tuple[np.ndarray,np.ndarray,float,float,float]],Lanelet]] = []
        self.precomputed_lane_polygons : Dict[CurvilinearCoordinateSystem ,np.ndarray ,np.ndarray] = precomputed_lane_polygons
        self.time_step = -1
        self.lane_width = 5
        self.ego_lanelet = None

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
    def get_lane_side_obs_intersection(x, y, orientation, length, width, curve : np.ndarray):
        def rotate(px, py, cx, cy, angle):
            s, c = math.sin(-angle), math.cos(-angle)
            dx, dy = px - cx, py - cy
            return dx * c - dy * s, dx * s + dy * c
        def inside(px, py):
            return (-l <= px <= l) and (-w <= py <= w)
        local = np.array(rotate(curve[0],curve[1],x, y, orientation))
        l = length / 2
        w = width / 2
        p = None
        eps = 1e-9
        def on_edge(px, py):
            if abs(px + l) <= eps and -w - eps <= py <= w + eps:  # left
                return True
            if abs(px - l) <= eps and -w - eps <= py <= w + eps:  # right
                return True
            if abs(py + w) <= eps and -l - eps <= px <= l + eps:  # bottom
                return True
            if abs(py - w) <= eps and -l - eps <= px <= l + eps:  # top
                return True
            return False
        for i, (lx, ly) in enumerate(local):
            if on_edge(lx, ly):
                if p is None:
                    p = i
                else:
                    result = [p, i]
                    return result  # if two curve points are on the edges
        for i in range(len(local) - 1):
            p1_inside = inside(*local[i])
            if p1_inside == inside(*local[i + 1]):  # either both are out or both are in
                continue
            outside_idx = i if not p1_inside else i + 1
            if p is None:
                p = outside_idx
            else:
                result = [p, outside_idx]
                return result
        return [p] if p else None

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

    def obs_start_end_index(self, obs : Obstacle, left, center, right):
        obs_state = obs.state_at_time(self.time_step)
        shape = obs.occupancy_at_time(self.time_step).shape
        c_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1], obs_state.orientation,
                                                      shape.length, shape.width, center)
        l_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1],
                                                      obs_state.orientation,shape.length, shape.width, left)
        r_pts = self.get_lane_side_obs_intersection(obs_state.position[0], obs_state.position[1],
                                                     obs_state.orientation,shape.length, shape.width, right)
        pts = []
        for k in (c_pts, l_pts, r_pts):
            if k is not None:
                pts.extend(k)
        if not pts:
            return None
        start = min(pts)
        end = max(pts)
        if start == end:
            return [start]
        else:
            return [start, end]

    def get_end_collision_free_area(self, lane : Lanelet, center, pt, preceding_v):
        successors = lane.successor
        if len(successors) == 0:
            return center[pt : len(center)], lane, preceding_v, self.v_max , 0
        def get_closest_obstacle_lane_velocity_distance(ls: Lanelet):
            lso = ls.get_obstacles(self.scenario.obstacles, self.time_step)
            if len(lso) == 0:
                r_min = 1.0 / self.kappa(ls.center_vertices)
                v_crit = np.sqrt(r_min * self.prop_ego["a_lat_max"])
                return v_crit, traveled_distance(ls.center_vertices, ls.center_vertices[-1])
            else:
                lso = self.sort_obstacles_in_lane(ls.lanelet_id, lso)
                lobs_state = lso[0].state_at_time(self.time_step)
                lshape = lso[0].occupancy_at_time(self.time_step).shape
                pts = self.get_lane_side_obs_intersection(lobs_state.position[0], lobs_state.position[1],
                                                          lobs_state.orientation, lshape.length, lshape.width,
                                                          ls.center_vertices)
                return lobs_state.velocity, traveled_distance(ls.center_vertices, ls.center_vertices[pts[0]])
        if len(successors) == 1:
            v,d = get_closest_obstacle_lane_velocity_distance(self.scenario.lanelet_network.find_lanelet_by_id(successors[0]))
        else:
            s_v_d = []
            for successor in successors:
                s_v_d.append((successor, get_closest_obstacle_lane_velocity_distance(self.scenario.lanelet_network.find_lanelet_by_id(successor))))
            min_ttc = math.inf
            closest_car = s_v_d[0][0]
            for lane in s_v_d:
                d_i = lane[2]
                v_i = lane[1]
                relative_speed = self.v_max - v_i
                ttc = d_i / relative_speed
                if ttc < min_ttc:
                    min_ttc = ttc
                    closest_car = lane[0]
            v,d = 0, 0
            for lane in s_v_d:
                if lane[0] == closest_car:
                    d = lane[2]
                    v = lane[1]
        return center[pt : len(center)], lane, preceding_v, v , d

    def get_lane_collision_free_areas(self, lane : Lanelet):
        C = []
        center = lane.center_vertices
        left = lane.left_vertices
        right = lane.right_vertices
        obs = lane.get_obstacles(self.scenario.obstacles, self.time_step)
        l_id = None
        r_id = None
        if lane.adj_left_same_direction:
            l_id = lane.adj_left
        if lane.adj_right_same_direction:
            r_id = lane.adj_right
        # empty lane with no vehicle entering or exiting it
        if len(obs) == 0:
            print("Empty lane")
            return [(center, lane, 0, self.v_max, 0)]
        i = 0
        obs = self.sort_obstacles_in_lane(lane.lanelet_id, obs)

        obs_state = obs[i].state_at_time(self.time_step)
        pts = self.obs_start_end_index(obs[i], left, center, right)
        if pts is None:
            i += 1
            print("obs: ", obs)
            print(obs[0])
            print(lane.left_vertices)
            print(lane.center_vertices)
            print(lane.right_vertices)
            if len(obs) == 1:
                print("Empty lane")
                return [(center, lane, 0, self.v_max, 0)]
            obs_state = obs[i].state_at_time(self.time_step)
            pts = self.obs_start_end_index(obs[i], left, center, right)
        preceding_v = obs_state.velocity
        if len(pts) == 1:   pt = pts[0]
        else:
            cps = center[0 : pts[0] + 1]
            C.append((cps,lane, 0, preceding_v, 0)) # add first collision free area
            pt = pts[1]
        for i in range(len(obs) - 2):
            obs_state = obs[i + 1].state_at_time(self.time_step)
            pts = self.obs_start_end_index(obs[i + 1], left, center, right)
            cps = center[pt : pts[0] + 1]
            C.append((cps, lane, preceding_v, obs_state.velocity, 0))  # add middle collision free areas
            if len(pts) == 2:
                pt = pts[1]
            preceding_v = obs_state.velocity
        if len(pts) == 2 and ((lane.lanelet_id == self.ego_lanelet.lanelet_id) |
                              (l_id and l_id == lane.lanelet_id) | (r_id and r_id == lane.lanelet_id)):
            # add last collision free area only for the ego and adjacent lanes
            C.append(self.get_end_collision_free_area(lane, center, pt, preceding_v))
        return C

    def sort_obstacles_in_lane(self, l_id : int ,obs : List[Obstacle]) -> List[Obstacle]:
        obs_with_center : List[Tuple[Obstacle, float]] = []
        ct, _, _ = self.precomputed_lane_polygons[l_id]
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

    def safeDistanceSetForSection(self, xi, yi, v_i, xj, yj, v_j,cp, l_id, distance_to_add) \
            -> List[Tuple[np.ndarray,np.ndarray,float,float,float]] :
        """
            This function takes the position and velocity of two Obstacles and the collision
            free area between them (as a part of the lane i.e. left-,center- and right points)
            and returns all the possible (s,v,d) states of an ego vehicle to be in this area
            without collision, with:
               - s -> All the center points in the lane that the vehicle can be on
               - v -> Velocities of the ego vehicle for each center
               - d -> The Area to leave on edges for safe bounds in the lane
        """
        ct,_,_ = self.precomputed_lane_polygons[l_id]
        tc = []
        for c in cp:    tc.append(ct.convert_to_curvilinear_coords(*c))
        txi, _ = ct.convert_to_curvilinear_coords(xi, yi)
        txj, _ = ct.convert_to_curvilinear_coords(xj, yj)
        txj += distance_to_add
        a_lat_max, a_lon_max, w, l, delta_react = (self.prop_ego["a_lat_max"], self.prop_ego["a_lon_max"],
                self.prop_ego["ego_width"], self.prop_ego["ego_length"], self.prop_ego["delta_react"])
        k = self.kappa(cp)
        if k == 0:
            r_min = math.inf
        else:
            r_min = 1.0 / k
        v_crit = np.sqrt(r_min * a_lat_max)
        # s >= s_i + Δ_safe(v, i)
        # s <= s_j - Δ_safe(v, j)
        vs = np.linspace(0, min(v_crit, self.v_max), 50)
        s_centers = np.array([s for (s, d) in tc]) # d is always 0
        safe_states = []
        # a_lon(v) = a_lon_max * sqrt( 1 - (v^2 / v_crit^2)^2 )
        def a_lon(v, a_lon_max, v_crit ):
            return a_lon_max * np.sqrt(max(0, 1 - (v ** 2 / v_crit ** 2) ** 2))
        # z(v,j) = v^2/(2 a_ego(v)) - v_j^2/(2 a_j_max) + delta_react*v
        def zeta_preceding(v, v_j, a_lon_max, v_crit, delta_react):
            return (v ** 2) / (2 * abs(a_lon(v,a_lon_max,v_crit))) - (v_j ** 2) / (2 * abs(a_lon_max)) + delta_react * v
        # z(v,i) = v_i^2/(2|a_i_max|) - v^2/(2|a_ego(v)|) + delta_react*v_i
        def zeta_succeeding(v, v_i, a_lon_max, v_crit, delta_react):
            return (v_i ** 2) / (2 * abs(a_lon_max)) - (v ** 2) / (2 * abs(a_lon(v,a_lon_max,v_crit))) + delta_react * v_i
        for v in vs:
            s_min_final = max(txi, max(zeta_succeeding(v,v_i,a_lon_max,v_crit,delta_react), 0))
            s_max_final = min(txj, max(zeta_preceding(v,v_j,a_lon_max,v_crit,delta_react), 0))
            if s_min_final <= s_max_final:
                start = np.argmin(np.abs(s_centers - s_min_final))
                end = np.argmin(np.abs(s_centers - s_max_final))
                # Changed from the original formula d_lim = r_min - sqrt(r_min^2 - l_front^2 + 0.5w),
                # as the car shape will be handled in the action testing part with Shapely, replaced it
                # with a lookahead distance equal to the velocity as we do this for each timestamp
                d_lim = r_min - np.sqrt(max(r_min ** 2 - v/2 ** 2, 0))
                shrink_per_side = max(0, 2.5 - d_lim)
                safe_states.append((cp[start], cp[end], v, shrink_per_side, shrink_per_side))
        return safe_states

    def union_safe_set(self, ll: Lanelet, safe_set_list_left, rl : Lanelet, safe_set_list_right):
        (ct, _, _) = self.precomputed_lane_polygons[ll.lanelet_id]
        nls = []
        nrs = []
        for s in safe_set_list_left:
            c1, c2, lv, dl, _ = s
            l_start = ct.convert_to_curvilinear_coords(c1[0], c1[1])
            l_end = ct.convert_to_curvilinear_coords(c2[0], c2[1])
            for c in safe_set_list_right:
                c1, c2, rv, _, dr = c
                r_start = ct.convert_to_curvilinear_coords(c1[0], c1[1])
                r_end = ct.convert_to_curvilinear_coords(c2[0], c2[1])
                start = max(l_start, r_start)
                end = min(l_end, r_end)
                if start <= end:
                    nls.append((ct.convert_to_cartesian_coords(start,0), ct.convert_to_cartesian_coords(end,0), lv, dl, -4))
                    nrs.append((ct.convert_to_cartesian_coords
                            (start, self.lane_width), ct.convert_to_cartesian_coords(end,self.lane_width), rv, -4, dr))

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
        self.time_step += 1
        self.in_or_entering_intersection = in_or_entering_intersection
        S : List[Tuple[List[Tuple[np.ndarray,np.ndarray,float,float,float]],Lanelet]] = []
        C = []
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
            if self.ego_lanelet.adj_left_same_direction:
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

    def safe_action_check(self, a, sv, ego_action):
        curr_vehicle: ContinuousVehicle = ego_action.vehicle
        new_vehicle_state = curr_vehicle.get_new_state(ego_action.rescale_action(np.array([a, sv])), ego_action.action_base)
        p = new_vehicle_state.position
        nv = new_vehicle_state.velocity
        W, L = self.prop_ego["ego_width"], self.prop_ego["ego_length"]
        rect = Polygon([(-L / 2, -W / 2), (-L / 2, W / 2), (L / 2, W / 2), (L / 2, -W / 2)])
        rect = rotate(rect, new_vehicle_state.orientation * 180 / math.pi, origin=(0, 0), use_radians=False)
        rect = translate(rect, xoff=p[0], yoff=p[1])
        for l in self.get_reachable_lanes():
            ct,lp,rp = self.precomputed_lane_polygons[l.lanelet_id]
            for s in self.safe_set:
                k, lane = s
                if lane.lanelet_id == l.lanelet_id:
                    for start, end, v, dl, dr in k:
                        if not v - 0.1 <= nv <= v + 0.1:
                            continue
                        def in_safe_space(left_points : np.ndarray, right_points: np.ndarray):
                            left_bound = left_points[start: end + 1]
                            right_bound = right_points[start: end + 1]
                            for i in range(len(left_bound)):
                                left_bound[i] = np.array(ct.convert_to_cartesian_coords(left_bound[i][0], left_bound[i][1] - dl))
                                right_bound[i] = np.array(ct.convert_to_cartesian_coords(right_bound[i][0], right_bound[i][1] + dr))
                            lane_polygon = Polygon(left_bound + right_bound[::-1])
                            if lane_polygon.contains(rect): return True
                            # allow intersections with the end and start of the lane for lane change
                            return not(LineString(left_bound).intersects(rect), LineString(right_bound).intersects(rect))
                        if in_safe_space(lp, rp):
                            return True
        return False


class SafetyLayer(CommonroadEnv):

    def __init__(self, meta_scenario_path=PATH_PARAMS["meta_scenario"],
                 train_reset_config_path=PATH_PARAMS["train_reset_config"],
                 test_reset_config_path=PATH_PARAMS["test_reset_config"],
                 visualization_path=PATH_PARAMS["visualization"], logging_path=None, test_env=False, play=False,
                 config_file=PATH_PARAMS["configs"]["commonroad-v1"], logging_mode=1, **kwargs) -> None:
        super().__init__(meta_scenario_path, train_reset_config_path, test_reset_config_path, visualization_path,
                         logging_path, test_env, play, config_file, logging_mode, **kwargs)
        self.observation = None
        self.prop_ego = {"ego_length" : 4.5, "ego_width" : 1.61 , "a_lat_max" : 9.0, "a_lon_max" : 11.5, "delta_react" : 0.5}
        self.time_step = 0
        self.lane_width = 5
        self.last_relative_heading = 0
        self.v_max = 45
        self.final_priority = -1
        self.intersection_lanes : List[Lanelet] = []
        self.conflict_lanes : defaultdict[int, List[Tuple[Lanelet, bool]]] = defaultdict(list)
        self.stanley_controller = StanleyController(3,1,0.05,0.3, 0.5, 2.9)
        self.pre_intersection_lanes = None
        self.precomputed_lane_polygons = {}
        self.safety_verifier = None
        self.in_or_entering_intersection = False

    def reset(self, seed=None, options: Optional[dict] = None, benchmark_id=None, scenario: Scenario = None,
              planning_problem: PlanningProblem = None) -> np.ndarray:
        initial_observation, info = super().reset(seed, options, benchmark_id, scenario, planning_problem)
        center_points = self.observation_collector.ego_lanelet.center_vertices.reshape(-1, 2)
        closest_centerpoint = center_points[
            np.linalg.norm(center_points - self.observation_collector._ego_state.position, axis=1).argmin()]
        self.safety_verifier = SafetyVerifier(self.scenario,self.prop_ego,self.precomputed_lane_polygons)
        initial_observation["distance_to_lane_end"] = traveled_distance(center_points[::-1], closest_centerpoint)
        self.observation = initial_observation
        self.time_step = 0
        self.compute_lane_sides_and_conflict()
        self.in_or_entering_intersection = self.intersection_check()
        self.safety_verifier.safeDistanceSet(self.observation_collector.ego_lanelet,self.in_or_entering_intersection)
        self.pre_intersection_lanes = None
        if self.in_or_entering_intersection:
            actions = self.intersection_safety()
        else:
            self.pre_intersection_lanes = None
            self.final_priority = -1
            actions = self.lane_safety()
        initial_observation["safe_actions"] = actions
        initial_observation["final_priority"] = np.array([self.final_priority], dtype=object)
        observation_vector = np.zeros(self.observation_space.shape)
        index = 0
        for k in initial_observation.keys():
            size = np.prod(initial_observation[k].shape)
            observation_vector[index: index + size] = initial_observation[k].flat
            index += size
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
        self.conflict_lanes.clear()
        for l in self.scenario.lanelet_network.lanelets:
            for k in self.scenario.lanelet_network.lanelets:
                if l == k or (l.predecessor and k.predecessor and l.predecessor == k.predecessor):
                    continue
                if l.polygon.shapely_object.intersects(k.polygon.shapely_object):
                    self.conflict_lanes[l.lanelet_id].append((k, is_right(l.center_vertices, k.center_vertices)))
        for l in self.scenario.lanelet_network.lanelets:
            def extend_centerline_to_include_points(center, left_pts, right_pts):
                vec = center[0] - center[1]
                points = np.vstack([left_pts[0], right_pts[0]])
                distances = np.dot((points - center[0]), vec / np.linalg.norm(vec))
                ext_len = max(0, -min(distances))
                ext = center[0] + vec / np.linalg.norm(vec) * ext_len
                center = np.vstack([ext, center])

                vec = center[-1] - center[-2]
                points = np.vstack([left_pts[-1], right_pts[-1]])
                distances = np.dot((points - center[-1]), vec / np.linalg.norm(vec))
                ext_len = max(0, max(distances))
                ext = center[-1] + vec / np.linalg.norm(vec) * ext_len
                center = np.vstack([center, ext])
                return center
            center_dense = resample_polyline_with_distance(extend_centerline_to_include_points
                                (l.center_vertices,l.left_vertices,l.right_vertices), 0.1)
            if center_dense.size < 6: continue
            ct = CurvilinearCoordinateSystem(center_dense, CLCSParams())
            x,y = 0,0
            left = np.array([])
            right = np.array([])
            try:
                for x,y in l.left_vertices:
                    left = np.append(left,ct.convert_to_curvilinear_coords(x,y))
                for x,y in l.right_vertices:
                    right = np.append(right,ct.convert_to_curvilinear_coords(x,y))
                #left = np.array([ct.convert_to_curvilinear_coords(x, y) for x, y in l.left_vertices])
                #right = np.array([ct.convert_to_curvilinear_coords(x, y) for x, y in l.right_vertices])
            except CartesianProjectionDomainError:
                print("left: ", l.left_vertices)
                print("center: ", l.center_vertices)
                print("right: ", l.right_vertices)
                print("dense center: ", center_dense[0], "  -  " , center_dense[-1])
                print("Error point : ",x, "  ",y)
            # Extend first/last points to handle boundary
            #left = np.vstack([left[0] - 1000, left, left[-1] + 1000])
            #right = np.vstack([right[0] - 1000, right, right[-1] + 1000])
            self.precomputed_lane_polygons[l.lanelet_id] = (ct, left, right)

    def step(self, action: Union[np.ndarray, State]) -> Tuple[np.ndarray, float, bool, dict]:
        reward_for_safe_action = 0
        in_conflict = self.observation_collector.conflict_zone.check_in_conflict_region(self.observation_collector._ego_state)
        in_intersection = True if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys() else False
        if self.safety_verifier.safe_action_check(action[1],action[0]):
            reward_for_safe_action = 1
        else:
            a,b =  self.find_safe_acceleration(action[1])
            if a<=b:
                action[0] = max(0,a)
                reward_for_safe_action = 0.5
            else:
                if in_conflict:
                    action[0] = self.compute_steering_velocity(self.compute_steering_velocity(self.observation_collector.ego_lanelet.center_vertices))
                    action[1] = 1
                else:
                    action[0] = 0
                    action[1] = -1
        observation, reward, terminated, truncated, info = super().step(action)
        if reward_for_safe_action:
            if self.in_or_entering_intersection:
                reward += self.safe_reward(action, in_intersection, in_conflict)
        else:
            reward -= 800
        center_points = self.observation_collector.ego_lanelet.center_vertices
        closest_centerpoint = center_points[np.linalg.norm(center_points - self.observation_collector._ego_state.position, axis=1).argmin()]
        observation["distance_to_lane_end"] = traveled_distance(center_points[::,-1],closest_centerpoint)
        self.observation = observation
        self.time_step += 1
        self.safety_verifier.safeDistanceSet()
        self.in_or_entering_intersection = self.intersection_check()
        if self.in_or_entering_intersection:
            actions =self.intersection_safety()
        else:
            self.pre_intersection_lanes = None
            self.final_priority = -1
            actions = self.lane_safety()
        observation["safe_actions"] = actions
        observation["final_priority"] = np.array([self.final_priority], dtype= object)
        observation_vector = np.zeros(self.observation_space.shape)
        index = 0
        for k in observation.keys():
            size = np.prod(observation[k].shape)
            observation_vector[index: index + size] = observation[k].flat
            index += size
        return observation_vector, reward, terminated, truncated, info

    def safe_reward(self, action, in_intersection, in_conflict):
        in_conflict_after = self.observation_collector.conflict_zone.check_in_conflict_region(self.observation_collector._ego_state)
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
            return False
        if (self.observation["v_ego"]**2 / (2 * a_max) < self.observation["distance_to_lane_end"] or
                self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys()):
            if self.observation["v_ego"]**2 / (2 * a_max) < self.observation["distance_to_lane_end"]:
                print("near lane end")
            else:
                print("in intersection")
            return True
        return False

    def compute_steering_velocity(self, center_points):
        """
            Compute the steering velocity of the ego vehicle for lane keeping.
        """
        L = self.prop_ego["ego_length"]
        p = self.observation_collector._ego_state.position
        yaw = self.observation_collector._ego_state.orientation
        v = max(self.observation["v_ego"], 0.1)
        steering_angle = math.atan(L * self.observation["global_turn_rate"] / v)
        cx = center_points[:, 0]
        cy = center_points[:, 1]
        path_yaw = np.unwrap(np.arctan2(np.gradient(cy), np.gradient(cx)))
        desired_angle,_,_ = self.stanley_controller.stanley_control(p[0],p[1],yaw,v,steering_angle,cx,cy,path_yaw)
        sv = (np.clip(desired_angle, -0.5, 0.5) - steering_angle)
        return (float(np.clip(sv, -0.4, 0.4)) - self.ego_action._rescale_bias[0]) /self.ego_action._rescale_factor[0]

    def find_safe_acceleration(self, sv):
        """
            Binary search for the min and max acceleration for given steering velocity.
            Using the binary search made it has constant complexity of 18 iterations for each 36 checks in total
        """
        low, high = -1, 1
        while high - low > 1e-5:
            mid = (low + high) / 2
            if self.safety_verifier.safe_action_check(mid, sv, self.ego_action):   high = mid
            else:   low = mid
        safe_min = high
        low, high = -1, 1
        while high - low > 1e-5:
            mid = (low + high) / 2
            if self.safety_verifier.safe_action_check(mid, sv, self.ego_action):   low = mid
            else:   high = mid
        safe_max = low
        return safe_min, safe_max

    def lane_safety(self):
        """
            Returns an array of safe actions each as a tuple of (sv,(a_min,a_max)).
            This does a max of 396 checks to build the Safe action set
        """
        At_safe_l = []
        fcl_input = self.compute_steering_velocity(self.observation_collector.ego_lanelet.center_vertices)
        if self.observation_collector.ego_lanelet.adj_left_same_direction:
            if self.observation_collector.ego_lanelet.adj_right_same_direction:
                steering_velocities = np.linspace(-1, 1, 11) # left, current and right
            else:
                steering_velocities = np.linspace(fcl_input - 0.1, 1, 11) # left and current
        elif self.observation_collector.ego_lanelet.adj_right_same_direction:
            steering_velocities = np.linspace(-1, fcl_input + 0.1, 11) # current and right
        else:
            steering_velocities = np.linspace(fcl_input-0.05, fcl_input+0.05, 3) # only current lane
        for sv in steering_velocities:
            safe_min, safe_max = self.find_safe_acceleration(sv)
            if safe_min <= safe_max:    At_safe_l.append((sv,(safe_min, safe_max)))
            else: At_safe_l.append((sv,None))
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
        for cp in lane.center_vertices:
            if math.dist(pos, cp) < dist:
                dist = math.dist(pos, cp)
                curr = cp
        d_c_i = traveled_distance(lane.center_vertices[::-1],curr)
        return (d_c_i <= D_m) or (d_c_i / obstacle.state_at_time(self.time_step).velocity <= T_a)

    def priority(self, lane : Lanelet, pre_intersection_lanes):
        """
            Updates priority with:
                1 --> yield
                0 --> priority
        """
        def dir_priority(incoming_lanes: List[Lanelet]):
            intersection_lanes_obs = []
            for l in incoming_lanes:
                obs = l.get_obstacles(self.scenario.obstacles, self.time_step)
                obs = self.safety_verifier.sort_obstacles_in_lane(l.lanelet_id, obs)
                if not obs: continue
                intersection_lanes_obs.append((l, obs[-1]))
            for k in intersection_lanes_obs:
                if self.priority_condition(*k):
                    return True
            return False
        self.final_priority = 1 if dir_priority(pre_intersection_lanes[lane.lanelet_id]) else 0

    def get_per_intersection_lanes_update_priority(self, lane : Lanelet):
        pre_intersection_lanes: defaultdict[int, list[Lanelet]] = defaultdict(list)
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
            Returns an array of safe actions each as a tuple of (sv,(a_min,a_max))
        """
        At_safe_in : List[Tuple[float, Tuple[float,float]]] = []
        route_ids = self.observation_collector.navigator.route.list_ids_lanelets
        curr_index = route_ids.index(self.observation_collector.ego_lanelet.lanelet_id)
        if curr_index == len(route_ids) - 1:
            return np.array([]) # simulation is done no next route
        nxt_id = route_ids[curr_index + 1]
        nxt_lane = self.scenario.lanelet_network.find_lanelet_by_id(nxt_id)
        if not self.pre_intersection_lanes:
            self.pre_intersection_lanes = self.get_per_intersection_lanes_update_priority(nxt_lane)
        else:
            self.priority(nxt_lane, self.pre_intersection_lanes)
        if self.prop_ego["ego_length"] / 2 >= self.observation["distance_to_lane_end"]:
            fcl_input = self.compute_steering_velocity(nxt_lane.center_vertices)
            self.final_priority = 1
        else:
            if self.observation_collector.ego_lanelet.lanelet_id in self.conflict_lanes.keys():
                self.final_priority = 1
            fcl_input = self.compute_steering_velocity(self.observation_collector.ego_lanelet.center_vertices)
        steering_velocities = np.linspace(fcl_input - 0.05, fcl_input + 0.05, 3)  # only current lane
        for sv in steering_velocities:
            safe_min, safe_max = self.find_safe_acceleration(sv)
            if safe_min <= safe_max:
                At_safe_in.append((sv, (safe_min, safe_max)))
            else:
                At_safe_in.append((sv, (0,0)))
        return np.array(At_safe_in, dtype=object)
