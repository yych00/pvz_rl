"""PvZ Q-networks with action-conditioned global and local attention."""

from __future__ import annotations

import math
from collections.abc import Mapping

from pvz_gym.action_space import (
    ACTION_SPACE_SIZE,
    COB_ACTIONS,
    COB_TARGET_CELLS,
    COLS,
    MAX_CARD_SLOTS,
    MAX_ROWS,
    PLANT_ACTIONS,
    SHOVEL_ACTIONS,
)
from pvz_gym.observation import (
    CARD_FEATURES,
    DROPPED_CARD_FEATURES,
    GLOBAL_FEATURES,
    GRID_FEATURES,
    MAX_DROPPED_CARDS,
    MOWER_STATES,
    PLANT_FEATURES,
    PLANT_LAYERS,
    SCENE_CAPACITY,
    STATE_CAPACITY,
    TYPE_CAPACITY,
    ZOMBIE_FEATURES,
    ZOMBIE_GRID_FEATURES,
    ZOMBIE_ROSTER_SIZE,
    ZOMBIE_BOSS_CATEGORY,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None


NETWORK_LOCAL_V6 = "entity_cell_local_attention_noop_board_v6"
NETWORK_SPATIAL = "entity_action_global_local_attention_card_set_128_2global_v9"


class _RelativeAttentionBlock(nn.Module if nn is not None else object):
    """Cross/self attention with a learned bias from explicit relative geometry."""

    def __init__(self, dim: int, relation_features: int, *, heads: int = 4, efficient: bool = False):
        if nn is None:  # pragma: no cover
            raise RuntimeError("Double DQN requires PyTorch")
        if dim % heads != 0:
            raise ValueError("attention dim must be divisible by heads")
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.efficient = efficient

        self.query_norm = nn.LayerNorm(self.dim)
        self.key_norm = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        relation_hidden = max(16, self.dim // 2)
        self.relative_bias = nn.Sequential(
            nn.Linear(relation_features, relation_hidden),
            nn.ReLU(),
            nn.Linear(relation_hidden, self.heads),
        ) if relation_features else None
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 2 * self.dim),
            nn.ReLU(),
            nn.Linear(2 * self.dim, self.dim),
        )

    def forward(self, queries, key_values, relations=None, *, key_mask=None, attention_mask=None):
        batch, query_count, _ = queries.shape
        key_count = key_values.shape[1]
        if self.relative_bias is not None and (
            relations is None or relations.shape[:3] != (batch, query_count, key_count)
        ):
            raise ValueError("relative attention relation tensor has the wrong shape")

        q = self.q_proj(self.query_norm(queries)).view(
            batch, query_count, self.heads, self.head_dim
        ).transpose(1, 2)
        normalized_keys = self.key_norm(key_values)
        k = self.k_proj(normalized_keys).view(
            batch, key_count, self.heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(normalized_keys).view(
            batch, key_count, self.heads, self.head_dim
        ).transpose(1, 2)

        allowed = None
        if key_mask is not None:
            if key_mask.shape != (batch, key_count):
                raise ValueError("relative attention key mask has the wrong shape")
            allowed = key_mask[:, None, :].expand(-1, query_count, -1)
        if attention_mask is not None:
            if attention_mask.shape != (batch, query_count, key_count):
                raise ValueError("attention visibility mask has the wrong shape")
            allowed = attention_mask if allowed is None else allowed & attention_mask
        if self.efficient and self.relative_bias is None:
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None if allowed is None else allowed[:, None],
            )
        else:
            # Preserve v6 arithmetic for exact legacy checkpoint continuation.
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if self.relative_bias is not None:
                scores = scores + self.relative_bias(relations.float()).permute(0, 3, 1, 2)
            if allowed is not None:
                scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            if allowed is not None:
                weights = weights.masked_fill(~allowed[:, None], 0)
            attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(
            batch, query_count, self.dim
        )
        result = queries + self.out_proj(attended)
        result = result + self.ffn(self.ffn_norm(result))
        return result


class LegacyLocalSpatialQNetwork(nn.Module if nn is not None else object):
    """Attention-only cell/entity Q-network; no convolutional layers are used."""

    spatial_channels = 64
    relation_dim = 64
    card_context_channels = MAX_CARD_SLOTS * relation_dim
    entity_dim = 16
    type_dim = 8
    state_dim = 4
    action_families = 5
    plant_relation_features = 8
    cell_relation_features = 8

    def __init__(
        self,
        action_dim: int = ACTION_SPACE_SIZE,
        hidden_sizes: tuple[int, int] = (512, 256),
        *, reward_context: bool = False,
    ):
        if nn is None:  # pragma: no cover
            raise RuntimeError("Double DQN requires PyTorch")
        if int(action_dim) != ACTION_SPACE_SIZE:
            raise ValueError(f"SpatialQNetwork requires action_dim={ACTION_SPACE_SIZE}")
        if len(hidden_sizes) != 2 or any(int(size) <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive values")
        super().__init__()
        context_hidden, context_dim = (int(size) for size in hidden_sizes)
        self.action_dim = int(action_dim)
        self.hidden_sizes = (context_hidden, context_dim)

        self.type_embedding = nn.Embedding(TYPE_CAPACITY + 1, self.type_dim, padding_idx=0)
        self.state_embedding = nn.Embedding(STATE_CAPACITY + 1, self.state_dim, padding_idx=0)
        self.scene_embedding = nn.Embedding(SCENE_CAPACITY + 1, self.state_dim, padding_idx=0)
        self.row_embedding = nn.Embedding(MAX_ROWS + 1, self.state_dim, padding_idx=0)
        self.zombie_phase_embedding = nn.Embedding(STATE_CAPACITY + 1, self.state_dim, padding_idx=0)
        self.zombie_height_embedding = nn.Embedding(STATE_CAPACITY + 1, self.state_dim, padding_idx=0)
        self.target_row_embedding = nn.Embedding(MAX_ROWS + 1, self.state_dim, padding_idx=0)
        self.target_col_embedding = nn.Embedding(COLS + 1, self.state_dim, padding_idx=0)
        self.terrain_embedding = nn.Embedding(TYPE_CAPACITY + 1, self.state_dim, padding_idx=0)
        self.mower_embedding = nn.Embedding(MOWER_STATES, self.state_dim)
        self.plant_layer_embedding = nn.Embedding(PLANT_LAYERS, self.state_dim)

        card_input = self.type_dim + CARD_FEATURES
        self.card_encoder = nn.Sequential(
            nn.Linear(card_input, self.entity_dim), nn.ReLU(),
            nn.Linear(self.entity_dim, self.relation_dim),
        )

        zombie_input = self.type_dim + self.state_dim * 5 + ZOMBIE_FEATURES
        self.zombie_encoder = nn.Sequential(
            nn.Linear(zombie_input, self.relation_dim), nn.ReLU(),
            nn.Linear(self.relation_dim, self.spatial_channels), nn.ReLU(),
        )
        dropped_input = self.type_dim + DROPPED_CARD_FEATURES
        self.dropped_encoder = nn.Sequential(
            nn.Linear(dropped_input, self.entity_dim), nn.ReLU(),
        )

        # Each cell is encoded independently. Spatial/entity relations are added only by attention.
        plant_cell_channels = PLANT_LAYERS * (self.type_dim + self.state_dim + PLANT_FEATURES)
        cell_input = (
            plant_cell_channels + self.state_dim + GRID_FEATURES + self.state_dim + 2
            + ZOMBIE_GRID_FEATURES
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_input, self.spatial_channels), nn.ReLU(),
            nn.Linear(self.spatial_channels, self.spatial_channels),
            nn.LayerNorm(self.spatial_channels),
        )

        plant_entity_input = self.type_dim + self.state_dim * 2 + PLANT_FEATURES
        self.plant_encoder = nn.Sequential(
            nn.Linear(plant_entity_input, self.spatial_channels), nn.ReLU(),
            nn.Linear(self.spatial_channels, self.spatial_channels), nn.ReLU(),
        )
        self.no_plant_token = nn.Parameter(torch.zeros(1, 1, self.spatial_channels))
        self.no_zombie_token = nn.Parameter(torch.zeros(1, 1, self.spatial_channels))

        self.cell_plant_attention = _RelativeAttentionBlock(
            self.spatial_channels, self.plant_relation_features, heads=4
        )
        self.cell_zombie_attention = _RelativeAttentionBlock(
            self.spatial_channels, 0, heads=4
        )
        self.cell_cell_attention = _RelativeAttentionBlock(
            self.spatial_channels, self.cell_relation_features, heads=4
        )

        self.reward_context = bool(reward_context)
        context_inputs = (
            GLOBAL_FEATURES + self.state_dim + ZOMBIE_ROSTER_SIZE
            + self.card_context_channels + 2 * self.entity_dim + 3 * self.reward_context
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_inputs, context_hidden), nn.ReLU(),
            nn.Linear(context_hidden, context_dim), nn.ReLU(),
        )
        self.context_film = nn.Linear(context_dim, 2 * self.spatial_channels)
        self.value_head = nn.Linear(context_dim, 1)
        self.type_advantage_head = nn.Linear(context_dim, self.action_families)

        # Action heads are linear over cell tokens; there are no 1x1 convolutions either.
        self.plant_cell_encoder = nn.Linear(self.spatial_channels, self.relation_dim)
        self.plant_card_bias = nn.Linear(self.relation_dim, 1)
        self.plant_cell_bias = nn.Linear(self.spatial_channels, 1)
        self.shovel_head = nn.Linear(self.spatial_channels, 2)
        self.cob_head = nn.Linear(self.spatial_channels, 1)
        self.interact_head = nn.Linear(self.spatial_channels, 1)

        cell_rows = torch.arange(MAX_ROWS, dtype=torch.float32).view(MAX_ROWS, 1).expand(MAX_ROWS, COLS).reshape(-1)
        cell_cols = torch.arange(COLS, dtype=torch.float32).view(1, COLS).expand(MAX_ROWS, COLS).reshape(-1)
        coords = torch.stack((
            cell_rows / max(MAX_ROWS - 1, 1),
            cell_cols / max(COLS - 1, 1),
        ), dim=-1)
        self.register_buffer("cell_rows", cell_rows, persistent=False)
        self.register_buffer("cell_cols", cell_cols, persistent=False)
        self.register_buffer("cell_coords", coords, persistent=False)

        plant_layers = torch.arange(PLANT_LAYERS, dtype=torch.long).view(PLANT_LAYERS, 1, 1).expand(PLANT_LAYERS, MAX_ROWS, COLS).reshape(-1)
        plant_rows = torch.arange(MAX_ROWS, dtype=torch.float32).view(1, MAX_ROWS, 1).expand(PLANT_LAYERS, MAX_ROWS, COLS).reshape(-1)
        plant_cols = torch.arange(COLS, dtype=torch.float32).view(1, 1, COLS).expand(PLANT_LAYERS, MAX_ROWS, COLS).reshape(-1)
        self.register_buffer("plant_layers", plant_layers, persistent=False)
        self.register_buffer("plant_rows", plant_rows, persistent=False)
        self.register_buffer("plant_cols", plant_cols, persistent=False)

        self.register_buffer(
            "cell_plant_relations",
            self._static_relations(cell_rows, cell_cols, plant_rows, plant_cols),
            persistent=False,
        )
        self.register_buffer(
            "cell_cell_relations",
            self._static_relations(cell_rows, cell_cols, cell_rows, cell_cols),
            persistent=False,
        )
        self.register_buffer(
            "cell_plant_visibility",
            self._local_visibility(cell_rows, cell_cols, plant_rows, plant_cols),
            persistent=False,
        )
        self.register_buffer(
            "cell_cell_visibility",
            self._local_visibility(cell_rows, cell_cols, cell_rows, cell_cols),
            persistent=False,
        )

        # NOOP reads every active cell without expanding coordinate actions' local view.
        self.noop_board_query = nn.Linear(context_dim, self.spatial_channels)
        self.noop_board_attention = _RelativeAttentionBlock(self.spatial_channels, 0, heads=4)
        self.noop_board_head = nn.Linear(self.spatial_channels, 1)

    def initialize_zero_q(self) -> None:
        """Initialize every action's Q to zero for every observation.

        Only scoring layers are cleared; the encoders retain their random
        features. Exploration family weights do not affect initial Q values.
        Invoke for new agents before copying online parameters to the target.
        Checkpoint loads subsequently replace all initialized parameters.
        """
        with torch.no_grad():
            for head in (
                self.value_head, self.type_advantage_head,
                self.plant_cell_encoder, self.plant_card_bias, self.plant_cell_bias,
                self.shovel_head, self.cob_head, self.interact_head, self.noop_board_head,
            ):
                head.weight.zero_()
                head.bias.zero_()

    @staticmethod
    def _local_visibility(query_rows, query_cols, key_rows, key_cols):
        drow = key_rows - query_rows.unsqueeze(-1)
        dcol = key_cols - query_cols.unsqueeze(-1)
        return (drow == 0) | ((drow.abs() <= 1) & (dcol.abs() <= 1))

    @staticmethod
    def _normalized_x_to_grid_col(x_norm):
        """Match PvZ Board::PixelToGridX and zombie_grid lawn geometry."""
        return torch.floor((x_norm * 900.0 - 40.0) / 80.0).clamp(0, COLS - 1)

    def _active_cells(self, observation):
        active_rows = (observation["global"][:, 7] * MAX_ROWS).round().long()
        return self.cell_rows.unsqueeze(0) < active_rows.unsqueeze(1)

    @staticmethod
    def _masked_mean_max(values, mask):
        mask_f = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        masked = values.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return mean, maximum

    @staticmethod
    def _center(scores, mask=None):
        """对给定动作分数（可带掩码）执行零均值中心化。"""
        if mask is None:
            return scores - scores.mean(dim=1, keepdim=True)
        mask_f = mask.to(scores.dtype)
        mean = (scores * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return scores - mean

    @staticmethod
    def _static_relations(query_rows, query_cols, key_rows, key_cols):
        drow = (key_rows[None, :] - query_rows[:, None]) / max(MAX_ROWS - 1, 1)
        dcol = (key_cols[None, :] - query_cols[:, None]) / max(COLS - 1, 1)
        return torch.stack((
            drow,
            dcol,
            drow.abs(),
            dcol.abs(),
            (drow == 0).to(torch.float32),
            (dcol == 0).to(torch.float32),
            (dcol > 0).to(torch.float32),
            (dcol < 0).to(torch.float32),
        ), dim=-1).unsqueeze(0)

    def _cards_and_dropped(self, observation):
        cards = self.card_encoder(torch.cat((
            self.type_embedding(observation["card_type"].long()),
            observation["cards"].float(),
        ), dim=-1))
        dropped = self.dropped_encoder(torch.cat((
            self.type_embedding(observation["dropped_card_type"].long()),
            observation["dropped_cards"].float(),
        ), dim=-1))
        return cards, dropped

    def _zombie_entities(self, observation):
        zombie_mask = observation["zombie_mask"].bool()

        def category(name, embedding):
            return embedding(observation[name].long().masked_fill(~zombie_mask, 0))

        zombie_input = torch.cat((
            category("zombie_type", self.type_embedding),
            category("zombie_phase", self.zombie_phase_embedding),
            category("zombie_height", self.zombie_height_embedding),
            category("zombie_row", self.row_embedding),
            category("zombie_target_row", self.target_row_embedding),
            category("zombie_target_col", self.target_col_embedding),
            observation["zombies"].float().masked_fill(~zombie_mask.unsqueeze(-1), 0),
        ), dim=-1)
        zombies = self.zombie_encoder(zombie_input).masked_fill(~zombie_mask.unsqueeze(-1), 0)
        return zombies, zombie_mask

    def _plant_entities(self, observation):
        batch = observation["global"].shape[0]
        plant_type = observation["plant_type"].long()
        plant_state = observation["plant_state"].long()
        plant_facts = observation["plants"].float()
        layer = self.plant_layer_embedding(self.plant_layers).view(
            1, PLANT_LAYERS, MAX_ROWS, COLS, self.state_dim
        ).expand(batch, -1, -1, -1, -1)
        encoded = self.plant_encoder(torch.cat((
            self.type_embedding(plant_type),
            self.state_embedding(plant_state),
            layer,
            plant_facts,
        ), dim=-1).reshape(batch, -1, self.type_dim + 2 * self.state_dim + PLANT_FEATURES))
        mask = (plant_facts[..., 0] > 0).reshape(batch, -1)
        return encoded.masked_fill(~mask.unsqueeze(-1), 0), mask

    def _cell_tokens(self, observation):
        batch = observation["global"].shape[0]
        plant_types = self.type_embedding(observation["plant_type"].long())
        plant_states = self.state_embedding(observation["plant_state"].long())
        plants = torch.cat((plant_types, plant_states, observation["plants"].float()), dim=-1)
        plants = plants.permute(0, 2, 3, 1, 4).reshape(batch, MAX_ROWS * COLS, -1)
        terrain = self.terrain_embedding(observation["grid_terrain"].long()).reshape(
            batch, MAX_ROWS * COLS, self.state_dim
        )
        grid = observation["grid"].float().reshape(batch, MAX_ROWS * COLS, GRID_FEATURES)
        zombie_grid = observation["zombie_grid"].float().reshape(
            batch, MAX_ROWS * COLS, ZOMBIE_GRID_FEATURES
        )
        mower = self.mower_embedding(observation["mower_state"].long())
        mower = mower.unsqueeze(2).expand(-1, -1, COLS, -1).reshape(
            batch, MAX_ROWS * COLS, self.state_dim
        )
        coords = self.cell_coords.to(dtype=grid.dtype).unsqueeze(0).expand(batch, -1, -1)
        return self.cell_encoder(torch.cat((plants, terrain, grid, mower, coords, zombie_grid), dim=-1))

    def _cell_plant_relation(self, batch: int, *, dtype, device):
        relation = self.cell_plant_relations.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        null_relation = torch.zeros(
            (batch, MAX_ROWS * COLS, 1, self.plant_relation_features),
            dtype=dtype,
            device=device,
        )
        return torch.cat((relation, null_relation), dim=2)

    def _cell_zombie_visibility(self, observation, zombie_mask):
        batch = observation["global"].shape[0]
        # Match zombie_grid / PvZ PixelToGridX: LAWN_XMIN=40 and 80px cells.
        # Mask padding before coordinate arithmetic so garbage/NaNs cannot leak.
        zrow = observation["zombie_row"].long().masked_fill(~zombie_mask, 0) - 1
        zx = observation["zombies"][..., 0].float().masked_fill(~zombie_mask, 0)
        zcol = self._normalized_x_to_grid_col(zx)
        active_rows = (observation["global"][:, 7] * MAX_ROWS).round().long()
        valid = (zombie_mask & (observation["zombie_type"].long() != ZOMBIE_BOSS_CATEGORY)
                 & (zrow >= 0) & (zrow < active_rows.unsqueeze(1)))
        visible = self._local_visibility(
            self.cell_rows.view(1, -1), self.cell_cols.view(1, -1),
            zrow.unsqueeze(1), zcol.unsqueeze(1),
        ) & valid.unsqueeze(1) & self._active_cells(observation).unsqueeze(-1)
        return torch.cat((visible, visible.new_ones((batch, MAX_ROWS * COLS, 1))), dim=2)

    def _relate_cells_to_plants(self, cells, plants, plant_mask, observation):
        batch = cells.shape[0]
        null = self.no_plant_token.expand(batch, -1, -1)
        keys = torch.cat((plants, null), dim=1)
        mask = torch.cat((plant_mask, plant_mask.new_ones((batch, 1))), dim=1)
        relations = self._cell_plant_relation(batch, dtype=cells.dtype, device=cells.device)
        active = self._active_cells(observation)
        visible = self.cell_plant_visibility.unsqueeze(0) & active.unsqueeze(-1)
        visible = visible & active.repeat(1, PLANT_LAYERS).unsqueeze(1)
        visible = torch.cat((visible, visible.new_ones((batch, MAX_ROWS * COLS, 1))), dim=2)
        return self.cell_plant_attention(cells, keys, relations, key_mask=mask, attention_mask=visible)

    def _relate_cells_to_zombies(self, cells, zombies, zombie_mask, observation):
        batch = cells.shape[0]
        null = self.no_zombie_token.expand(batch, -1, -1)
        keys = torch.cat((zombies, null), dim=1)
        mask = torch.cat((zombie_mask, zombie_mask.new_ones((batch, 1))), dim=1)
        visible = self._cell_zombie_visibility(observation, zombie_mask)
        return self.cell_zombie_attention(cells, keys, key_mask=mask, attention_mask=visible)

    def _relate_cells_to_cells(self, cells, observation):
        batch = cells.shape[0]
        relations = self.cell_cell_relations.to(
            device=cells.device, dtype=cells.dtype
        ).expand(batch, -1, -1, -1)
        active = self._active_cells(observation)
        visible = self.cell_cell_visibility.unsqueeze(0) & active.unsqueeze(1) & active.unsqueeze(-1)
        return self.cell_cell_attention(cells, cells, relations, attention_mask=visible)

    def _encode(self, observation):
        if not isinstance(observation, Mapping):
            raise ValueError("SpatialQNetwork expects a structured Observation mapping")
        cards, dropped = self._cards_and_dropped(observation)
        plants, plant_mask = self._plant_entities(observation)
        zombies, zombie_mask = self._zombie_entities(observation)
        cells = self._cell_tokens(observation)

        # Read raw neighbouring cells first. Attending to cells after their
        # entity attention would import entities beyond this query's 15 cells.
        cells = self._relate_cells_to_cells(cells, observation)
        cells = self._relate_cells_to_plants(cells, plants, plant_mask, observation)
        cells = self._relate_cells_to_zombies(cells, zombies, zombie_mask, observation)

        dropped_mask = observation["dropped_cards"][..., 0] > 0
        dropped_mean, dropped_max = self._masked_mean_max(dropped, dropped_mask)
        context_input = torch.cat((
            observation["global"].float(),
            self.scene_embedding(observation["scene"].long()).flatten(1),
            observation["zombie_roster"].float(),
            cards.flatten(1),
            dropped_mean,
            dropped_max,
        ), dim=1)
        if self.reward_context:
            context_input = torch.cat((context_input, observation["reward_context"].float()), dim=1)
        context = self.context_encoder(context_input)

        scale, shift = self.context_film(context).chunk(2, dim=1)
        cells = cells * (1.0 + torch.tanh(scale).unsqueeze(1))
        cells = cells + shift.unsqueeze(1)
        return cards, cells, context

    def _plant_scores(self, cards, cells):
        encoded_cells = self.plant_cell_encoder(cells)
        scores = torch.bmm(cards, encoded_cells.transpose(1, 2)) / math.sqrt(self.relation_dim)
        scores = scores + self.plant_card_bias(cards)
        scores = scores + self.plant_cell_bias(cells).transpose(1, 2)
        return scores.reshape(cells.shape[0], PLANT_ACTIONS)

    def _shovel_scores(self, cells):
        # Action IDs group all normal shovel cells first, then all pumpkin cells.
        return self.shovel_head(cells).transpose(1, 2).reshape(cells.shape[0], SHOVEL_ACTIONS)

    def _cob_scores(self, cells):
        return self.cob_head(cells).reshape(cells.shape[0], COB_ACTIONS)

    def _interact_scores(self, cells):
        return self.interact_head(cells).reshape(cells.shape[0], COB_TARGET_CELLS)

    def _ordinary_q(self, cards, cells, context, active_cells):
        value = self.value_head(context)
        family = self.type_advantage_head(context)
        family = family - family.mean(dim=1, keepdim=True)
        # A mean over all board actions would reintroduce out-of-view entities
        # into each Q value. Center only the global family advantages above.
        plant = self._plant_scores(cards, cells)
        shovel = self._shovel_scores(cells)
        cob = self._cob_scores(cells)
        interact = self._interact_scores(cells)
        noop_query = self.noop_board_query(context).unsqueeze(1)
        noop_board = self.noop_board_attention(noop_query, cells, key_mask=active_cells)
        noop = self.noop_board_head(noop_board[:, 0])
        result = torch.cat((
            value + family[:, 0:1] + noop,
            value + family[:, 1:2] + plant,
            value + family[:, 2:3] + shovel,
            value + family[:, 3:4] + cob,
            value + family[:, 4:5] + interact,
        ), dim=1)
        if result.shape[1] != self.action_dim:
            raise RuntimeError("structured Q action layout drift")
        return result

    def forward(self, observation, *args, **kwargs):
        return self._ordinary_q(*self._encode(observation), self._active_cells(observation))


class SpatialQNetwork(LegacyLocalSpatialQNetwork):
    """Score all complete actions with shared, action-conditioned readers.

    Both branches read raw tokens: the local branch cannot import a remote
    entity through an already globally contextualized neighbouring cell.
    Board tokens retain all plant layers, terrain, mowers and zombie overflow.
    Card slots only locate native actions; card queries/memory have no slot
    embedding and the global context reads a permutation-invariant card set.
    """

    spatial_channels = 128
    relation_dim = 128
    card_context_channels = 2 * relation_dim + 1

    def __init__(self, action_dim: int = ACTION_SPACE_SIZE,
                 hidden_sizes: tuple[int, int] = (512, 256), *, reward_context: bool = False,
                 boss_projectiles: bool = False):
        super().__init__(action_dim, hidden_sizes, reward_context=reward_context)
        self.boss_projectiles = bool(boss_projectiles)
        # Retain the shared observation encoders; v6 scoring/attention modules
        # live only in LegacyLocalSpatialQNetwork for exact resume.
        for name in (
            "plant_encoder", "plant_layer_embedding", "no_plant_token", "no_zombie_token",
            "cell_plant_attention", "cell_zombie_attention", "cell_cell_attention",
            "context_film", "type_advantage_head", "plant_cell_encoder",
            "plant_card_bias", "plant_cell_bias", "shovel_head", "cob_head", "interact_head",
            "noop_board_query", "noop_board_attention", "noop_board_head",
        ):
            delattr(self, name)
        dim = self.spatial_channels
        self.action_family_embedding = nn.Embedding(self.action_families, dim)
        self.shovel_target_embedding = nn.Embedding(2, dim)
        self.action_card_projection = nn.Linear(self.relation_dim, dim)
        self.action_cell_projection = nn.Linear(dim, dim)
        self.action_position_projection = nn.Linear(2, dim, bias=False)
        self.action_context_projection = nn.Linear(self.hidden_sizes[1], dim)
        self.action_query_norm = nn.LayerNorm(dim)
        self.memory_kind_embedding = nn.Embedding(4, dim)
        if self.boss_projectiles:
            self.dynamic_entity_class_embedding = nn.Embedding(2, dim)
        self.empty_spatial_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.action_global_attention = nn.ModuleList(
            _RelativeAttentionBlock(dim, 0, efficient=True) for _ in range(2)
        )
        self.action_local_attention = _RelativeAttentionBlock(dim, 0, efficient=True)
        self.action_score = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

        # Gym order: NOOP, card-major PLANT, normal/pumpkin SHOVEL, COB, INTERACT.
        families = torch.tensor([0] + [1] * PLANT_ACTIONS + [2] * SHOVEL_ACTIONS
                                + [3] * COB_ACTIONS + [4] * COB_TARGET_CELLS)
        targets = torch.cat((torch.zeros(1, dtype=torch.long),
                             torch.arange(MAX_ROWS * COLS).repeat(MAX_CARD_SLOTS + 4)))
        self.register_buffer("action_families_index", families, persistent=False)
        self.register_buffer("action_target_cells", targets, persistent=False)

    def initialize_zero_q(self) -> None:
        # Only final scalar heads start at zero, keeping the feature path live.
        with torch.no_grad():
            for head in (self.value_head, self.action_score[-1]):
                head.weight.zero_()
                head.bias.zero_()

    def transfer_encoder(self, source_state):
        """Transfer matching v6 observation encoders, never action/value heads."""
        destination = self.state_dict()
        encoder_names = {
            "type_embedding", "state_embedding", "scene_embedding", "row_embedding",
            "zombie_phase_embedding", "zombie_height_embedding", "target_row_embedding",
            "target_col_embedding", "terrain_embedding", "mower_embedding",
            "card_encoder", "zombie_encoder", "dropped_encoder", "cell_encoder", "context_encoder",
        }
        shared = {name: value for name, value in source_state.items()
                  if name in destination and name.split(".", 1)[0] in encoder_names
                  and value.shape == destination[name].shape}
        self.load_state_dict(shared, strict=False)
        return tuple(shared)

    def _encode_action_state(self, observation):
        if not isinstance(observation, Mapping):
            raise ValueError("SpatialQNetwork expects a structured Observation mapping")
        active = self._active_cells(observation)
        active_grid = active.reshape(-1, MAX_ROWS, COLS)
        plant_mask = (observation["plants"][..., 0] > 0) & active_grid.unsqueeze(1)
        card_mask = observation["cards"][..., 0] > 0
        dropped_mask = observation["dropped_cards"][..., 0] > 0
        active_rows = active_grid[:, :, 0].sum(1)
        zombie_rows = observation["zombie_row"].long() - 1
        is_boss = observation["zombie_type"].long() == ZOMBIE_BOSS_CATEGORY
        row_valid = (zombie_rows >= 0) & (zombie_rows < active_rows.unsqueeze(1))
        zombie_mask = observation["zombie_mask"].bool() & (is_boss | row_valid)

        # Sanitize padding before embeddings/projections: even NaN garbage in
        # an inactive row must not leak into queries, context, keys or gradients.
        clean = dict(observation)
        masks = {
            "plant_type": plant_mask, "plant_state": plant_mask,
            "plants": plant_mask.unsqueeze(-1), "grid_terrain": active_grid,
            "grid": active_grid.unsqueeze(-1), "zombie_grid": active_grid.unsqueeze(-1),
            "mower_state": active_grid[:, :, 0], "card_type": card_mask,
            "cards": card_mask.unsqueeze(-1), "dropped_card_type": dropped_mask,
            "dropped_cards": dropped_mask.unsqueeze(-1),
        }
        for name, mask in masks.items():
            clean[name] = observation[name].masked_fill(~mask, 0)
        clean["zombie_mask"] = zombie_mask
        # Defensively remove placeholder geometry even for direct tensor inputs.
        clean["zombie_row"] = observation["zombie_row"].masked_fill(is_boss, 0)
        clean["zombies"] = observation["zombies"].clone()
        clean["zombies"][..., :2] = clean["zombies"][..., :2].masked_fill(is_boss.unsqueeze(-1), 0)
        cards, dropped = self._cards_and_dropped(clean)
        cards = cards.masked_fill(~card_mask.unsqueeze(-1), 0)
        cells = self._cell_tokens(clean).masked_fill(~active.unsqueeze(-1), 0)
        zombies, _ = self._zombie_entities(clean)
        dropped_mean, dropped_max = self._masked_mean_max(dropped, dropped_mask)
        card_mean, card_max = self._masked_mean_max(cards, card_mask)
        card_count = card_mask.to(cards.dtype).sum(1, keepdim=True) / MAX_CARD_SLOTS
        context_input = torch.cat((
            clean["global"].float(), self.scene_embedding(clean["scene"].long()).flatten(1),
            clean["zombie_roster"].float(), card_mean, card_max, card_count, dropped_mean, dropped_max,
        ), dim=1)
        if self.reward_context:
            context_input = torch.cat((context_input, observation["reward_context"].float()), dim=1)
        context = self.context_encoder(context_input)
        return cards, cells, zombies, context, active, zombie_mask, card_mask, clean

    def _action_queries(self, cards, cells, context, active, action_ids=None):
        batch, _, dim = cells.shape
        cell_count = MAX_ROWS * COLS
        card_features = self.action_card_projection(cards)
        if action_ids is not None:
            families = self.action_families_index[action_ids]
            targets = self.action_target_cells[action_ids]
            slots = ((action_ids - 1) // cell_count).clamp(0, MAX_CARD_SLOTS - 1)
            parameters = card_features.gather(1, slots.unsqueeze(-1).expand(-1, -1, dim))
            parameters = parameters.masked_fill((families != 1).unsqueeze(-1), 0)
            shovel = ((action_ids - 1 - PLANT_ACTIONS) // cell_count).clamp(0, 1)
            parameters = parameters + self.shovel_target_embedding(shovel).masked_fill(
                (families != 2).unsqueeze(-1), 0)
            has_target = families != 0
            target_active = active.gather(1, targets) & has_target
            target_cells = self.action_cell_projection(cells).gather(
                1, targets.unsqueeze(-1).expand(-1, -1, dim))
            target_cells = target_cells.masked_fill(~target_active.unsqueeze(-1), 0)
            positions = self.action_position_projection(self.cell_coords[targets])
            positions = positions.masked_fill(~has_target.unsqueeze(-1), 0)
            return self.action_query_norm(
                self.action_family_embedding(families) + parameters + positions + target_cells
                + self.action_context_projection(context).unsqueeze(1))
        parameters = torch.cat((
            cells.new_zeros((batch, 1, dim)),
            card_features.repeat_interleave(cell_count, dim=1),
            self.shovel_target_embedding.weight.repeat_interleave(cell_count, dim=0)
                .unsqueeze(0).expand(batch, -1, -1),
            cells.new_zeros((batch, COB_ACTIONS + COB_TARGET_CELLS, dim)),
        ), dim=1)
        has_target = self.action_families_index != 0
        target_cells = self.action_cell_projection(cells)[:, self.action_target_cells]
        target_active = active[:, self.action_target_cells] & has_target.unsqueeze(0)
        target_cells = target_cells.masked_fill(~target_active.unsqueeze(-1), 0)
        positions = self.action_position_projection(self.cell_coords[self.action_target_cells])
        positions = positions.masked_fill(~has_target.unsqueeze(-1), 0)
        return self.action_query_norm(
            self.action_family_embedding(self.action_families_index).unsqueeze(0)
            + parameters + positions.unsqueeze(0) + target_cells
            + self.action_context_projection(context).unsqueeze(1)
        )

    def _action_branches(self, observation, action_ids=None):
        cards, cells, zombies, context, active, zombie_mask, card_mask, clean = (
            self._encode_action_state(observation)
        )
        queries = self._action_queries(cards, cells, context, active, action_ids)
        batch = cells.shape[0]
        kinds = self.memory_kind_embedding.weight
        projectile_visibility = None
        if self.boss_projectiles:
            projectiles, projectile_mask, projectile_visibility = self._boss_projectile_entities(clean, active)
            zombies = zombies + self.dynamic_entity_class_embedding.weight[0]
            zombies = zombies.masked_fill(~zombie_mask.unsqueeze(-1), 0)
            entities = torch.cat((zombies, projectiles), dim=1)
            entity_mask = torch.cat((zombie_mask, projectile_mask), dim=1)
        else:
            entities, entity_mask = zombies, zombie_mask
        spatial_memory = torch.cat((cells + kinds[0], entities + kinds[1],
                                    self.empty_spatial_token.expand(batch, -1, -1)), dim=1)
        spatial_mask = torch.cat((active, entity_mask, active.new_ones((batch, 1))), dim=1)
        global_memory = torch.cat((
            spatial_memory, self.action_card_projection(cards) + kinds[2],
            self.action_context_projection(context).unsqueeze(1) + kinds[3],
        ), dim=1)
        global_mask = torch.cat((spatial_mask, card_mask, active.new_ones((batch, 1))), dim=1)
        # Refine action queries twice against the same raw global memory.
        global_features = queries
        for reader in self.action_global_attention:
            global_features = reader(global_features, global_memory, key_mask=global_mask)

        # Local keys are raw cells/entities, with no global or two-hop mixing.
        cell_visibility = self.cell_cell_visibility.unsqueeze(0) & active.unsqueeze(1)
        zombie_visibility = self._cell_zombie_visibility(clean, zombie_mask)
        if projectile_visibility is not None:
            zombie_visibility = torch.cat((zombie_visibility[:, :, :-1], projectile_visibility,
                                           zombie_visibility[:, :, -1:]), dim=2)
        visibility = torch.cat((cell_visibility, zombie_visibility), dim=2)
        if action_ids is None:
            visibility = visibility[:, self.action_target_cells]
            has_target = (self.action_families_index != 0).view(1, -1, 1)
        else:
            targets = self.action_target_cells[action_ids]
            visibility = visibility.gather(1, targets.unsqueeze(-1).expand(-1, -1, visibility.shape[-1]))
            has_target = (self.action_families_index[action_ids] != 0).unsqueeze(-1)
        visibility = visibility & has_target
        local_features = self.action_local_attention(
            queries, spatial_memory, key_mask=spatial_mask, attention_mask=visibility,
        )
        # Waiting has no target and uses the same scorer with only global input.
        local_features = local_features * has_target
        return queries, global_features, local_features, context

    def _boss_projectile_entities(self, observation, active):
        """Reuse zombie type/position/row encoding, with an explicit entity class.

        Projectiles have no HP, phase or zombie target: their unused channels are
        zero, while the class embedding distinguishes those absent facts.
        """
        kinds = observation["boss_projectile_type"].long()
        rows = observation["boss_projectile_row"].long()
        positions = observation["boss_projectiles"].float()
        active_rows = active.reshape(-1, MAX_ROWS, COLS)[:, :, 0].sum(1)
        mask = (observation["boss_projectile_mask"].bool() & (kinds >= 1) & (kinds <= 2)
                & (rows >= 1) & (rows <= active_rows.unsqueeze(1)))
        rows = rows.masked_fill(~mask, 0)
        positions = positions.masked_fill(~mask.unsqueeze(-1), 0)
        # Reserve two category IDs unused by native zombie types.
        types = (TYPE_CAPACITY - 2 + kinds).masked_fill(~mask, 0)
        zero_category = torch.zeros_like(rows)
        facts = F.pad(positions, (0, ZOMBIE_FEATURES-positions.shape[-1]))
        inputs = torch.cat((self.type_embedding(types),
                            self.zombie_phase_embedding(zero_category),
                            self.zombie_height_embedding(zero_category), self.row_embedding(rows),
                            self.target_row_embedding(zero_category), self.target_col_embedding(zero_category),
                            facts), dim=-1)
        encoded = self.zombie_encoder(inputs) + self.dynamic_entity_class_embedding.weight[1]
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0)
        columns = self._normalized_x_to_grid_col(positions[..., 0])
        visible = self._local_visibility(self.cell_rows.view(1, -1), self.cell_cols.view(1, -1),
                                          (rows-1).unsqueeze(1), columns.unsqueeze(1))
        visible = visible & mask.unsqueeze(1) & active.unsqueeze(-1)
        return encoded, mask, visible

    def forward(self, observation, *args, **kwargs):
        queries, global_features, local_features, context = self._action_branches(observation)
        scores = self.action_score(torch.cat((queries, global_features, local_features), dim=-1))
        return self.value_head(context) + scores.squeeze(-1)

    def forward_selected_actions(self, observation, actions):
        """Score only the requested complete actions without changing Q semantics.

        A vector selects one action per observation; a B x K matrix is useful
        for grouped queries and equivalence validation. Encoders and memories
        remain shared with full forward, while attention uses only K queries.
        """
        if actions.dtype != torch.long:
            raise ValueError("selected actions must have dtype torch.long")
        squeeze = actions.ndim == 1
        ids = actions.unsqueeze(1) if squeeze else actions
        if ids.ndim != 2 or ids.shape[0] != observation["global"].shape[0] or not ids.shape[1]:
            raise ValueError("selected actions must have shape B or B x K")
        if ids.device != observation["global"].device:
            raise ValueError("selected actions and observation must share a device")
        if bool(((ids < 0) | (ids >= self.action_dim)).any()):
            raise ValueError("selected action ID is outside the action space")
        queries, global_features, local_features, context = self._action_branches(observation, ids)
        scores = self.action_score(torch.cat((queries, global_features, local_features), dim=-1))
        values = self.value_head(context) + scores.squeeze(-1)
        return values.squeeze(1) if squeeze else values


__all__ = ["NETWORK_SPATIAL", "NETWORK_LOCAL_V6", "SpatialQNetwork", "LegacyLocalSpatialQNetwork"]
