import unittest

import numpy as np

from bandit_router import BanditAction, BanditRouter, BanditRouterConfig


class TestBanditRouter(unittest.TestCase):
    def test_basic_route_and_update(self):
        cfg = BanditRouterConfig(
            seed=123,
            delta=0.1,
            use_primal_dual=True,
            require_action_latency_safe=False,
        )
        r = BanditRouter(3, cfg)

        x = np.asarray([0.2, -0.1, 0.5], dtype=np.float32)
        actions = [BanditAction(variant="cheap"), BanditAction(variant="base")]
        action_info = {a.key(): {"cost": 1.0, "latency_safe": True} for a in actions}
        feats = {a.key(): x for a in actions}
        costs = {a.key(): 1.0 for a in actions}
        baseline = actions[0]

        chosen, meta = r.route(
            actions=actions,
            features_by_action=feats,
            cost_hat_by_action=costs,
            baseline_action=baseline,
            action_info=action_info,
        )
        self.assertIn(chosen.variant, {"cheap", "base"})

        # Update with a violation should increase Q (primal-dual)
        upd1 = r.update(
            x=x,
            action=chosen,
            cost=1.0,
            risk_violation=1,
            quality_label=1,
            label_key="k1",
        )
        self.assertTrue(upd1.get("updated"))
        self.assertGreaterEqual(r.Q, 0.0)

        # Update with no violation should not explode
        upd2 = r.update(
            x=x,
            action=chosen,
            cost=1.0,
            risk_violation=0,
            quality_label=0,
            label_key="k2",
        )
        self.assertTrue(upd2.get("updated"))
        self.assertTrue(np.isfinite(r.Q))

    def test_pending_label_ingest(self):
        cfg = BanditRouterConfig(
            seed=0,
            store_pending_when_no_label=True,
            max_pending_labels=4,
        )
        r = BanditRouter(2, cfg)
        x = np.asarray([1.0, 2.0], dtype=np.float32)
        a = BanditAction(variant="cheap")
        info = {a.key(): {"cost": 1.0, "latency_safe": True}}

        feats = {a.key(): x}
        costs = {a.key(): 1.0}

        # Route then update with missing label -> pending stored
        r.route(actions=[a], features_by_action=feats, cost_hat_by_action=costs, baseline_action=a, action_info=info)
        upd = r.update(
            x=x,
            action=a,
            cost=1.0,
            risk_violation=0,
            quality_label=None,
            label_key="join-1",
        )
        self.assertTrue(upd.get("updated"))
        self.assertTrue(upd.get("pending_stored"))
        self.assertEqual(upd.get("pending_size"), 1)

        ing = r.ingest_quality_label("join-1", 1)
        self.assertTrue(ing.get("updated"))
        self.assertEqual(ing.get("pending_size"), 0)


if __name__ == "__main__":
    unittest.main()
