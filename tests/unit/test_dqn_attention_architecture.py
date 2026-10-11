import unittest

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn import agent as agent_module
from train.dqn.networks import NETWORK_SPATIAL, SpatialQNetwork


@unittest.skipIf(agent_module.torch is None, "PyTorch is required for attention architecture tests")
class AttentionOnlyArchitectureTest(unittest.TestCase):
    def test_network_contains_no_convolution(self):
        network = SpatialQNetwork(hidden_sizes=(64, 32))
        convolutions = [
            module
            for module in network.modules()
            if isinstance(module, agent_module.torch.nn.Conv2d)
        ]
        self.assertEqual(convolutions, [])
        self.assertEqual(NETWORK_SPATIAL, "entity_action_global_local_attention_card_set_128_2global_v9")

    def test_attention_only_network_forward_backward(self):
        network = SpatialQNetwork(hidden_sizes=(64, 32))
        observations = {
            name: agent_module.torch.as_tensor(np.stack([value, value]))
            for name, value in empty_observation().items()
        }

        q_values = network(observations)

        self.assertEqual(tuple(q_values.shape), (2, ACTION_SPACE_SIZE))
        self.assertTrue(bool(agent_module.torch.isfinite(q_values).all()))
        q_values[:, (0, 1, 541, 649)].sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in network.parameters()))

    def test_shared_action_attention_blocks_are_present(self):
        network = SpatialQNetwork(hidden_sizes=(64, 32))
        self.assertTrue(hasattr(network, "action_global_attention"))
        self.assertTrue(hasattr(network, "action_local_attention"))
        self.assertTrue(hasattr(network, "action_score"))
        self.assertEqual(network.spatial_channels, 128)
        self.assertEqual(network.relation_dim, 128)
        self.assertEqual(network.card_context_channels, 257)
        self.assertEqual(len(network.action_global_attention), 2)
        for reader in (*network.action_global_attention, network.action_local_attention):
            self.assertEqual(reader.dim, 128)
            self.assertEqual(reader.heads, 4)

    def test_second_global_reader_refines_first_output_using_same_memory(self):
        network = SpatialQNetwork(hidden_sizes=(32, 16)).eval()
        observation = {
            name: agent_module.torch.as_tensor(value).unsqueeze(0)
            for name, value in empty_observation().items()
        }
        calls = []
        handles = [reader.register_forward_hook(
            lambda module, args, kwargs, output: calls.append((args, kwargs, output)),
            with_kwargs=True,
        ) for reader in network.action_global_attention]
        try:
            with agent_module.torch.no_grad():
                queries, global_features, _, _ = network._action_branches(observation)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0][0], queries)
        self.assertIs(calls[1][0][0], calls[0][2])
        self.assertIs(calls[1][0][1], calls[0][0][1])
        self.assertIs(calls[1][1]["key_mask"], calls[0][1]["key_mask"])
        self.assertIs(global_features, calls[1][2])


if __name__ == "__main__":
    unittest.main()
