import copy
import json
import math
import threading
import unittest
import urllib.request
import urllib.error

from pacdata.generate import episodes_for_group
from pacdata.observe import observe
from pacdata.packing import apply_placement

from projects.pattern_evaluation.planner import Planner, load_config, prepare_observation, to_packing_env_action
from projects.pattern_evaluation.rollout import sample_futures
from projects.pattern_evaluation.scoring import normalized_weights, priority, static_components, valid_candidates


def observation():
    ep = next(episodes_for_group(2, dict(seed=20261005, min_boxes=18, max_boxes=18), "train"))
    return observe(ep, dict(processed=0, pallet=ep["pallet"], placed=[], remeasured=[], routed=0))


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.obs = observation(); self.config = load_config(); self.config.update(future_draws=3, future_depth=3)
        self.planner = Planner(self.config, use_model=False)

    def test_mask_precedes_any_score(self):
        candidates = [dict(candidate_id="bad", position_m=[-1, 0, 0], yaw_deg=0),
                      dict(candidate_id="good", position_m=[0, 0, 0], yaw_deg=0)]
        result = self.planner.plan(self.obs, candidates, mode="teacher")
        self.assertEqual(result["action"]["candidate"]["candidate_id"], "good")
        self.assertEqual(result["diagnostics"]["rejected"][0]["status"], "REJECT")

    def test_safety_target_is_not_compensated_by_weights(self):
        self.assertGreater(priority(dict(safety=1.), -100), priority(dict(safety=.999), 100))
        self.assertGreater(priority(dict(safety=.7), -100), priority(dict(safety=.6), 100))
        self.assertGreater(priority(dict(safety=1.), .5), priority(dict(safety=1.), .4))

    def test_weights_are_normalized_and_invalid_values_rejected(self):
        self.assertAlmostEqual(sum(normalized_weights({k: 2*v for k, v in self.config["weights"].items()}).values()), 1.)
        for bad in (-1, float("nan"), float("inf")):
            weights = dict(self.config["weights"], space=bad)
            with self.assertRaises(ValueError): normalized_weights(weights)

    def test_future_draws_preserve_verified_prefix(self):
        seqs = sample_futures(self.obs, self.config)
        for seq in seqs:
            self.assertEqual([b["box_id"] for b in seq["boxes"][:len(self.obs["observed_preview"])]],
                             [t["track_id"] for t in self.obs["observed_preview"][:self.config["future_depth"]]])
        self.assertEqual(seqs, sample_futures(self.obs, self.config))

    def test_teacher_evaluates_all_valid_candidates(self):
        rows, _ = valid_candidates(self.obs, self.config)
        result = self.planner.plan(self.obs, mode="teacher")
        self.assertEqual(result["diagnostics"]["detailed_rollouts"], len(rows))
        self.assertIsNotNone(result["future"]["cvar_fit"])

    def test_missing_model_falls_back_to_full_teacher(self):
        result = self.planner.plan(self.obs)
        self.assertEqual(result["diagnostics"]["detailed_rollouts"], result["diagnostics"]["valid_candidates"])

    def test_ood_falls_back_even_with_a_ranker(self):
        class ShouldNotPredict:
            def predict(self, _): raise AssertionError("OOD must bypass the neural model")
        self.planner.model = ShouldNotPredict(); self.obs["distribution_status"] = "OOD"
        result = self.planner.plan(self.obs)
        self.assertTrue(result["diagnostics"]["ood"])
        self.assertEqual(result["diagnostics"]["detailed_rollouts"], result["diagnostics"]["valid_candidates"])

    def test_sensor_failure_remeasurement(self):
        for value in (None, float("nan"), -1):
            self.obs["current_box"]["mass_kg"] = value
            self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "REMEASURE")

    def test_damage_is_ng_not_normal_buffer(self):
        self.obs["current_box"]["visual_damage_observed"] = True
        self.assertEqual(self.planner.plan(self.obs, allow_buffer=True)["action"]["type"], "ROUTE_NG")

    def test_state_unverified_is_hold(self):
        self.obs["state_verified"] = False
        self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "HOLD")

    def test_oracle_is_rejected(self):
        self.obs["oracle"] = {}
        with self.assertRaises(ValueError): self.planner.plan(self.obs)

    def test_buffer_capacity_and_once_only(self):
        self.obs["current_box"]["dimensions_m"] = [3, 3, 3]
        self.assertEqual(self.planner.plan(self.obs, allow_buffer=True)["action"]["type"], "BUFFER_CURRENT")
        self.obs["current_box"]["buffer_moves"] = 1
        self.assertEqual(self.planner.plan(self.obs, allow_buffer=True)["action"]["type"], "PALLET_CLOSE")
        self.obs["current_box"]["buffer_moves"] = 0; self.obs["buffer_capacity"] = 0
        self.assertEqual(self.planner.plan(self.obs, allow_buffer=True)["action"]["type"], "PALLET_CLOSE")

    def test_retrieve_buffer_at_end(self):
        self.obs["buffer_boxes"] = [dict(self.obs["current_box"], buffer_moves=1)]
        self.obs["current_box"] = None
        result = self.planner.plan(self.obs, allow_buffer=True)
        self.assertEqual(result["action"]["type"], "RETRIEVE_BUFFER")

    def test_duplicate_current_box_id_is_rejected(self):
        self.obs["buffer_boxes"] = [dict(self.obs["current_box"], buffer_moves=1)]
        with self.assertRaises(ValueError): self.planner.plan(self.obs)

    def test_http_uses_same_model_and_rejects_invalid_input(self):
        from projects.pattern_evaluation.__main__ import make_server
        server = make_server(self.planner, port=0)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            req = urllib.request.Request(url+"/plan", data=json.dumps(self.obs).encode(), headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                self.assertEqual(r.status, 200)
                self.assertEqual(json.load(r)["action"]["type"], "PLACE_CURRENT")
            with urllib.request.urlopen(url+"/health", timeout=5) as r:
                self.assertEqual(json.load(r)["status"], "ok")
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(urllib.request.Request(url+"/plan", data=b'{"oracle":{}}'), timeout=5)
            self.assertEqual(raised.exception.code, 400)
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=2)

    def test_unseen_inventory_not_silently_discarded(self):
        self.obs["current_box"] = None
        self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "WAIT")
        self.obs["unseen_inventory"] = {}; self.obs["stream_terminated"] = True
        self.assertEqual(self.planner.plan(self.obs)["action"]["type"], "DONE")

    def test_safety_saturates_and_does_not_shrink_with_processed_count(self):
        candidate = valid_candidates(self.obs, self.config)[0][0]["candidate"]
        before = static_components(self.obs, candidate, self.config)
        self.obs["processed"] = 9999
        self.assertEqual(before, static_components(self.obs, candidate, self.config))

    def test_planning_is_pure_and_robot_check_remains_required(self):
        before = copy.deepcopy(self.obs); result = self.planner.plan(self.obs)
        self.assertEqual(before, self.obs)
        self.assertEqual(result["robot_feasibility"], "NOT_CHECKED")
        self.assertTrue(result["requires_robot_validation"])
        self.assertEqual(to_packing_env_action(result)["type"], "PLACE")


if __name__ == "__main__": unittest.main()
