"""Mission requirements and module boundary regression tests; robot results are mocks."""
import copy
import json
import math
import threading
import unittest
import urllib.request
import xml.etree.ElementTree as ET

from pacdata.packing import apply_placement, mask
from projects.pattern_evaluation.adapters import reference_box, reference_candidate
from projects.pattern_evaluation.contracts import ROBOT_CHECKS, buffer_observation, placement_token, state_token
from projects.pattern_evaluation.metrics import pattern_metrics
from projects.pattern_evaluation.planner import Planner, load_config, prepare_observation
from projects.pattern_evaluation.scoring import feature_vector, valid_candidates
from projects.pattern_evaluation.sequence import evaluate_sequence
from projects.pattern_evaluation.teacher import label_observation
from projects.pattern_evaluation.visualize import render_svg, scene_data


def box(identity="A", mass=5., dimensions=(.4, .4, .2), capacity=150.):
    return dict(box_id=identity, sku=identity, dimensions_m=list(dimensions), mass_kg=mass,
                top_load_capacity_N=capacity, allowed_yaw_deg=[0, 90],
                measurement_valid=True, visual_damage_observed=False)


def observation():
    return dict(request_id="mission-fixture", state_version="state-1", robot_scene_version="scene-1",
        coordinate_frame="pallet", units=dict(length="m", mass="kg", force="N"), state_verified=True,
        pallet=dict(length_m=1., width_m=1., max_height_m=1., max_mass_kg=50.),
        current_box=box(), placed_boxes=[], buffer_boxes=[], buffer_capacity=2,
        observed_preview=[], unordered_visible_hints=[], unseen_inventory={}, catalog={},
        camera=dict(dimension_sigma_m=0.), stream_terminated=True)


def candidate(identity="C0", position=(0., 0., 0.)):
    return dict(candidate_id=identity, position_m=list(position), yaw_deg=0)


def feedback(obs, cand, seconds=6., **checks):
    value = copy.deepcopy(cand)
    value["robot_validation"] = dict(state_token=state_token(prepare_observation(obs)), placement_token=placement_token(cand),
        box_id=obs["current_box"]["box_id"], validator="MOCK_FOR_UNIT_TEST_ONLY", estimated_cycle_seconds=seconds,
        checks=dict(dict.fromkeys(ROBOT_CHECKS, True), **checks))
    return value


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.obs = observation(); self.config = load_config()
        self.config.update(future_draws=2, future_depth=2)
        self.planner = Planner(self.config, use_model=False)

    def test_five_requirement_outputs_and_explainable_score(self):
        self.obs["pattern_generation_ms"] = 12.
        result = self.planner.plan(self.obs, [candidate()])
        m = result["metrics"]["after"]
        self.assertAlmostEqual(m["volume_utilization"], .032)
        self.assertAlmostEqual(m["floor_coverage_ratio"], .16)
        self.assertEqual(m["cog_m"], [.2, .2, .1])
        self.assertAlmostEqual(m["stability"]["contacts"][0]["contact_area_m2"], .16)
        self.assertEqual(m["stability"]["minimum_support_ratio"], 1.)
        self.assertIsNone(m["stability"]["collapse_probability"])
        self.assertAlmostEqual(sum(m["load_distribution"]["quadrant_load_N"]), 5*9.81)
        self.assertAlmostEqual(sum(result["score_contributions"].values()), result["score"])
        self.assertAlmostEqual(result["timing"]["combined_planning_ms"], result["decision_ms"]+12)
        self.assertFalse(result["execution_ready"])
        self.assertEqual(len(result["diagnostics"]["ranking"]), 1)

    def test_stack_loads_contact_cog_and_nonuniform_floor_map(self):
        p = self.obs["pallet"]
        placed = apply_placement(p, [], box("lower", 8.), candidate())
        placed = apply_placement(p, placed, box("upper", 2., (.2, .2, .2)), candidate(position=(.2, .2, .2)))
        m = pattern_metrics(p, placed)
        lower = m["load_distribution"]["boxes"][0]
        self.assertAlmostEqual(lower["upper_load_N"], 19.62)
        self.assertAlmostEqual(m["stability"]["contacts"][0]["supported_cog_m"][0], .22)
        self.assertEqual(m["load_distribution"]["quadrant_load_N"][1:], [0., 0., 0.])
        self.assertAlmostEqual(m["load_distribution"]["quadrant_load_N"][0], 98.1)

    def test_robot_time_changes_choice_without_changing_model_features(self):
        self.planner.config["weights"] = {k: float(k == "handling") for k in self.config["weights"]}
        slow, fast = candidate("slow"), candidate("fast", (.6, .6, 0.))
        supplied = [feedback(self.obs, slow, 18), feedback(self.obs, fast, 2)]
        result = self.planner.plan(self.obs, supplied, mode="current", execution_mode="validated")
        self.assertEqual(result["action"]["candidate"]["candidate_id"], "fast")
        self.assertTrue(result["execution_ready"])
        self.assertEqual(result["timing"]["robot_cycle_seconds_estimate"], 2)
        self.assertEqual(result["static"]["handling_source"], "EXTERNAL_ROBOT_ESTIMATE")
        original = valid_candidates(self.obs, self.config, [fast])[0][0]
        enriched = valid_candidates(self.obs, self.config, [supplied[1]])[0][0]
        self.assertEqual(feature_vector(self.obs, fast, original["static"]), feature_vector(self.obs, fast, enriched["static"]))

    def test_each_robot_check_failure_prevents_selection(self):
        for check in ROBOT_CHECKS:
            with self.subTest(check=check):
                result = self.planner.plan(self.obs, [feedback(self.obs, candidate(), **{check: False})])
                self.assertEqual(result["action"]["type"], "HOLD")
                self.assertEqual(result["diagnostics"]["rejected"][0]["stage"], "ROBOT")

    def test_rejected_robot_candidate_is_replaced_by_valid_one(self):
        blocked = feedback(self.obs, candidate("blocked"), collision_free=False)
        clear = feedback(self.obs, candidate("clear", (.6, 0., 0.)))
        result = self.planner.plan(self.obs, [blocked, clear], execution_mode="validated")
        self.assertEqual(result["action"]["candidate"]["candidate_id"], "clear")

    def test_stale_robot_context_cannot_be_executed(self):
        for change in ("state_version", "robot_scene_version", "box", "pallet", "pose", "grasp"):
            with self.subTest(change=change):
                obs = copy.deepcopy(self.obs); cand = feedback(obs, candidate())
                if change in ("state_version", "robot_scene_version"): obs[change] = "changed"
                elif change == "box": obs["current_box"]["mass_kg"] = 6.
                elif change == "pallet": obs["pallet"]["width_m"] = 1.1
                elif change == "pose": cand["position_m"][0] = .1
                else: cand["grasp_id"] = "side"
                result = self.planner.plan(obs, [cand], execution_mode="validated")
                self.assertEqual(result["action"]["type"], "HOLD")
                self.assertFalse(result["execution_ready"])
                self.assertEqual(result["diagnostics"]["rejected"][0]["robot"]["status"], "STALE")

    def test_partial_unversioned_or_missing_robot_checks_require_review(self):
        cand = feedback(self.obs, candidate()); del cand["robot_validation"]["checks"]["ik"]
        self.assertFalse(self.planner.plan(self.obs, [cand])["execution_ready"])
        self.assertEqual(self.planner.plan(self.obs, [cand], execution_mode="validated")["action"]["type"], "HOLD")
        obs = copy.deepcopy(self.obs); del obs["robot_scene_version"]
        cand = feedback(obs, candidate())
        self.assertEqual(self.planner.plan(obs, [cand])["robot"]["status"], "PARTIAL")
        self.assertEqual(self.planner.plan(self.obs, [candidate()], execution_mode="validated")["action"]["type"], "HOLD")

    def test_eoat_mass_is_included_before_external_payload_pass(self):
        self.obs["robot_profile"] = dict(max_payload_kg=6., eoat_mass_kg=2.)
        cand = feedback(self.obs, candidate())
        result = self.planner.plan(self.obs, [cand])
        self.assertEqual(result["action"]["type"], "HOLD")
        self.assertIn("ROBOT_PAYLOAD_WITH_EOAT", result["diagnostics"]["rejected"][0]["reasons"])

    def test_replanning_rejects_stale_pallet_or_load_state(self):
        self.obs["placed_boxes"] = apply_placement(self.obs["pallet"], [], box("placed"), candidate())
        self.obs["pallet"]["length_m"] = .3
        result = self.planner.plan(self.obs)
        self.assertEqual(result["action"]["reason"], "STATE_RECONCILIATION_REQUIRED")
        self.obs["pallet"]["length_m"] = 1.
        self.obs["placed_boxes"][0]["upper_load_N"] = 10.
        self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "HOLD")
        with self.assertRaises(ValueError): label_observation(self.obs, self.config)

    def test_support_cycles_and_missing_boxes_are_not_trusted(self):
        placed = apply_placement(self.obs["pallet"], [], box("placed"), candidate())
        for supporter in ("missing", "placed"):
            self.obs["placed_boxes"] = copy.deepcopy(placed)
            self.obs["placed_boxes"][0]["supporter_id"] = supporter
            self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "HOLD")

    def test_units_orientation_and_boolean_sensor_values(self):
        for field, value in (("coordinate_frame", "robot_base"), ("units", dict(length="mm", mass="kg", force="N")), ("state_verified", "true")):
            obs = dict(self.obs, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError): self.planner.plan(obs)
        self.obs["current_box"]["allowed_yaw_deg"] = [180]
        with self.assertRaises(ValueError): self.planner.plan(self.obs)
        self.obs["current_box"]["allowed_yaw_deg"] = [0]; self.obs["current_box"]["mass_kg"] = True
        self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "REMEASURE")

    def test_invalid_robot_numbers_and_non_boolean_checks_are_rejected(self):
        for val in (float("nan"), float("inf"), -1., "6", True):
            with self.subTest(val=val), self.assertRaises(ValueError):
                self.planner.plan(self.obs, [feedback(self.obs, candidate(), val)])
        cand = feedback(self.obs, candidate()); cand["robot_validation"]["checks"]["ik"] = "PASS"
        with self.assertRaises(ValueError): self.planner.plan(self.obs, [cand])

    def test_soft_budget_preserves_masks_and_teacher_is_complete(self):
        self.obs["decision_budget_ms"] = 0.000001
        values = [candidate("invalid", (-1., 0., 0.)), candidate("good")]
        result = self.planner.plan(self.obs, values)
        self.assertEqual(result["action"]["candidate"]["candidate_id"], "good")
        self.assertTrue(result["timing"]["budget_exceeded"])
        self.assertTrue(result["diagnostics"]["budget_exhausted"])
        self.assertEqual(result["diagnostics"]["detailed_rollouts"], 0)
        result = self.planner.plan(self.obs, values, mode="teacher")
        self.assertEqual(result["diagnostics"]["detailed_rollouts"], 1)

    def test_pallet_change_holds_or_buffers_without_placing(self):
        self.obs["process_state"] = "PALLET_CHANGE"
        self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "HOLD")
        self.assertEqual(self.planner.plan(self.obs, allow_buffer=True)["action"]["type"], "BUFFER_CURRENT")
        self.obs["buffer_capacity"] = 0
        self.assertEqual(self.planner.plan(self.obs, allow_buffer=True)["action"]["conveyor_command"], "STOP")
        self.obs["process_state"] = "REPACKING"
        self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "HOLD")

    def test_buffer_retrieval_keeps_waiting_current_and_accepts_robot_feedback(self):
        self.obs["current_box"]["dimensions_m"] = [2., 2., 2.]
        self.obs["buffer_boxes"] = [dict(box("shelf"), buffer_moves=1)]
        context = buffer_observation(prepare_observation(self.obs), "shelf")
        self.assertEqual(context["observed_preview"][0]["track_id"], "A")
        values = {"shelf": [feedback(context, candidate())]}
        result = self.planner.plan(self.obs, [], allow_buffer=True, execution_mode="validated", buffer_candidates=values)
        self.assertEqual(result["action"]["type"], "RETRIEVE_BUFFER")
        self.assertTrue(result["execution_ready"])
        self.assertEqual(self.obs["current_box"]["box_id"], "A")
        self.obs["buffer_boxes"][0]["visual_damage_observed"] = True
        self.assertEqual(self.planner.plan(self.obs, allow_buffer=True)["action"]["reason"], "BUFFER_REQUIRES_INSPECTION")

    def test_sequence_checks_order_support_and_unknown_future(self):
        b = box("B", 2., (.2, .2, .2))
        self.obs["observed_preview"] = [dict(track_id="B", sku="B", dimensions_m=b["dimensions_m"], nominal_mass_kg=2.,
            top_load_capacity_N=150., allowed_yaw_deg=[0, 90])]
        steps = [dict(box_id="A", candidate=candidate()), dict(box_id="B", candidate=candidate("above", (0., 0., .2)))]
        result = evaluate_sequence(self.obs, steps, self.config)
        self.assertTrue(result["packing_valid"])
        self.assertFalse(result["robot_validated"])
        self.assertAlmostEqual(result["final_metrics"]["volume_utilization"], .04)
        self.assertEqual(evaluate_sequence(self.obs, steps[::-1], self.config)["failed_step"], 0)
        invalid = copy.deepcopy(steps); invalid[1]["candidate"]["position_m"] = [.6, .6, .2]
        self.assertEqual(evaluate_sequence(self.obs, invalid, self.config)["failed_step"], 1)
        invalid[1]["box_id"] = "future_unknown"
        self.assertEqual(evaluate_sequence(self.obs, invalid, self.config)["reason"], "BOX_UNAVAILABLE_OR_CONFIRMED_ORDER_VIOLATION")

    def test_mandatory_packing_constraints(self):
        p = self.obs["pallet"]
        lower = box("lower", 5., capacity=10.)
        placed = apply_placement(p, [], lower, candidate())
        cases = [
            ([], box(), candidate(position=(.8, 0., 0.)), "BOUNDARY"),
            ([], box(dimensions=(.4, .4, 1.2)), candidate(), "HEIGHT"),
            ([], box(mass=60.), candidate(), "PALLET_MASS"),
            (placed, box("upper", 6.), candidate(position=(0., 0., .2)), "HEAVY_ON_LIGHT"),
            (placed, box("upper", 2.), candidate(position=(0., 0., .2)), "TOP_LOAD"),
            (placed, box("upper", 1.), candidate(position=(.2, 0., .2)), "SUPPORT"),
            (placed, box("upper", 1.), candidate(position=(0., 0., 0.)), "OVERLAP"),
        ]
        for state, b, c, reason in cases:
            with self.subTest(reason=reason):
                obs = observation(); obs.update(placed_boxes=state, current_box=b)
                result = self.planner.plan(obs, [c], mode="current")
                self.assertEqual(result["action"]["type"], "PALLET_CLOSE")
                self.assertIn(reason, result["diagnostics"]["rejected"][0]["reasons"])
        b = box(); b["allowed_yaw_deg"] = [0]; c = dict(candidate(), yaw_deg=90)
        self.assertIn("ROTATION", mask(p, [], b, c)["reasons"])

    def test_failed_robot_sequence_does_not_claim_packing_failure(self):
        steps = [dict(box_id="A", candidate=feedback(self.obs, candidate(), ik=False))]
        result = evaluate_sequence(self.obs, steps, self.config)
        self.assertEqual(result["status"], "ROBOT_REPLAN_REQUIRED")
        self.assertIsNone(result["packing_valid"])

    def test_reference_mm_and_orientation_conversion(self):
        value = reference_candidate(dict(i=10, j=5, z=200., o=1), 10)
        self.assertEqual(value["position_m"], [.1, .05, .2]); self.assertEqual(value["yaw_deg"], 90)
        b = reference_box(dict(box_id="B", type_id="SKU", w=400., d=300., h=200., mass=5., max_load=100., orientations=[0, 1], damaged=False))
        self.assertEqual(b["dimensions_m"], [.4, .3, .2]); self.assertEqual(b["top_load_capacity_N"], 100.)
        with self.assertRaises(ValueError): reference_candidate(dict(i=0, j=0, z=0., o=2), 10)

    def test_geometry_and_constraints_across_96_cases(self):
        checked = 0
        for side in (.8, 1., 1.2):
            for mass in (2., 5.):
                for yaw in (0, 90):
                    for xy in ((0., 0.), (.3, .3), (.7, .7), (-.1, 0.)):
                        for z in (0., .2):
                            obs = observation(); obs["pallet"]["length_m"] = side; obs["current_box"]["mass_kg"] = mass
                            c = dict(candidate_id="sweep", position_m=[*xy, z], yaw_deg=yaw)
                            result = self.planner.plan(obs, [c], mode="current")
                            if result["action"]["type"] == "PLACE_CURRENT":
                                self.assertEqual(mask(obs["pallet"], [], obs["current_box"], c)["status"], "ALLOW")
                                self.assertLessEqual(result["metrics"]["after"]["volume_utilization"], 1.)
                            else:
                                self.assertNotEqual(mask(obs["pallet"], [], obs["current_box"], c)["status"], "ALLOW")
                            checked += 1
        self.assertEqual(checked, 96)

    def test_svg_scene_and_http_sequence_contract(self):
        result = self.planner.plan(self.obs, [candidate()])
        ET.fromstring(render_svg(self.obs, result))
        self.assertEqual(scene_data(self.obs, result)["boxes"][0]["status"], "PROPOSED")
        from projects.pattern_evaluation.__main__ import make_server
        server = make_server(self.planner, port=0)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            value = dict(observation=self.obs, steps=[dict(box_id="A", candidate=candidate())])
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/evaluate-sequence", data=json.dumps(value).encode())
            with urllib.request.urlopen(req, timeout=5) as response:
                self.assertTrue(json.load(response)["packing_valid"])
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=2)


if __name__ == "__main__": unittest.main()
