"""Shared exposure/rate multi-task neural challenger for unified model v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import EventDistribution, RawPrediction
from ..features import CATEGORICAL_FEATURES
from ..schema import EVENTS, distribution_family
from .config import NeuralV3Config
from .context import V3FeatureEncoder
from .gbdt import training_weights


@dataclass
class ExposureRateNeural:
    config: NeuralV3Config = field(default_factory=NeuralV3Config)
    events: tuple[str, ...] = EVENTS
    encoder: V3FeatureEncoder = field(default_factory=V3FeatureEncoder)
    network: object | None = None
    cardinalities: list[int] = field(default_factory=list)
    active_events: set[str] = field(default_factory=set)
    event_caps: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _torch():
        try:
            import torch
            # LightGBM and PyTorch are exercised in the same bounded-search
            # process.  A single CPU thread avoids the macOS OpenMP/runtime
            # contention that can otherwise crash during LayerNorm.
            torch.set_num_threads(1)
            return torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for ExposureRateNeural") from exc

    def _build_network(self, n_numeric: int):
        torch = self._torch()
        nn = torch.nn
        cards = self.cardinalities
        base_dims = [min(32, max(4, int(round(c ** 0.25 * 4)))) for c in cards]
        dims = [max(2, int(round(d * self.config.embedding_scale))) for d in base_dims]
        active = sorted(self.active_events)

        class Net(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.embeddings = nn.ModuleList([
                    nn.Embedding(card + 1, dim) for card, dim in zip(cards, dims)
                ])
                width = sum(dims) + n_numeric
                layers = []
                for _ in range(outer.config.depth):
                    layers.extend([
                        nn.Linear(width, outer.config.hidden),
                        nn.LayerNorm(outer.config.hidden),
                        nn.GELU(),
                        nn.Dropout(outer.config.dropout),
                    ])
                    width = outer.config.hidden
                self.trunk = nn.Sequential(*layers)
                self.appearance = nn.Linear(width, 1)
                self.minutes = nn.Linear(width, 2)
                self.heads = nn.ModuleDict({event: nn.Linear(width, 2) for event in active})
                task_names = ["appearance", "minutes", *active]
                self.task_log_vars = nn.ParameterDict({
                    name: nn.Parameter(torch.zeros(())) for name in task_names
                })

            def forward(self, cat, num):
                embeddings = [layer(cat[:, i]) for i, layer in enumerate(self.embeddings)]
                shared = self.trunk(torch.cat(embeddings + [num], dim=1))
                return {
                    "appearance": self.appearance(shared).squeeze(1),
                    "minutes": self.minutes(shared),
                    "events": {name: head(shared) for name, head in self.heads.items()},
                }

        return Net(self)

    def _arrays(self, frame: pd.DataFrame):
        cat, numeric = self.encoder.transform(frame)
        minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0).to_numpy("float32")
        minute_mask = frame["available__minutes"].fillna(False).astype(bool).to_numpy()
        played = (minutes > 0).astype("float32")
        targets = np.zeros((len(frame), len(self.active_events)), dtype="float32")
        masks = np.zeros_like(targets, dtype=bool)
        for i, event in enumerate(sorted(self.active_events)):
            values = pd.to_numeric(frame[event], errors="coerce").fillna(0).to_numpy("float32")
            family = distribution_family(event)
            if family == "bernoulli":
                targets[:, i] = np.clip(values, 0, 1)
                masks[:, i] = frame[f"available__{event}"].fillna(False).astype(bool).to_numpy()
            else:
                rate = values / np.maximum(minutes, 1e-6) * 80.0
                targets[:, i] = np.clip(
                    rate, 0, self.event_caps.get(event, np.inf),
                )
                masks[:, i] = (
                    frame[f"available__{event}"].fillna(False).astype(bool).to_numpy()
                    & (minutes > 0)
                )
        return (
            cat, numeric, minutes, minute_mask, played, targets, masks,
        )

    def _weighted_task(self, name, loss):
        torch = self._torch()
        log_var = torch.clamp(self.network.task_log_vars[name], -2.0, 2.0)
        return torch.exp(-log_var) * loss + log_var

    def _loss(self, output, minutes, minute_mask, played, targets, masks):
        torch = self._torch()
        losses = []
        if self.config.use_appearance and minute_mask.any():
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["appearance"][minute_mask], played[minute_mask],
            )
            losses.append(self._weighted_task("appearance", loss))
        positive = minute_mask & (minutes > 0)
        if positive.any():
            raw = output["minutes"][positive]
            truth = torch.log1p(minutes[positive])
            log_var = raw[:, 1].clamp(-6, 6)
            loss = 0.5 * (log_var + (truth - raw[:, 0]) ** 2 / torch.exp(log_var))
            losses.append(self._weighted_task("minutes", loss.mean()))
        for i, event in enumerate(sorted(self.active_events)):
            valid = masks[:, i]
            if not valid.any():
                continue
            raw = output["events"][event][valid]
            truth = targets[valid, i]
            family = distribution_family(event)
            if family == "bernoulli":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(raw[:, 0], truth)
            elif family == "lognormal":
                log_truth = torch.log1p(truth)
                log_var = raw[:, 1].clamp(-6, 6)
                row_loss = 0.5 * (
                    log_var + (log_truth - raw[:, 0]) ** 2 / torch.exp(log_var)
                )
                exposure = (minutes[valid] / 80.0).clamp(0.05, 1.0)
                loss = (row_loss * exposure).sum() / exposure.sum()
            else:
                rate = torch.nn.functional.softplus(raw[:, 0]) + 1e-6
                dispersion = torch.nn.functional.softplus(raw[:, 1]) + 0.05
                total = rate + dispersion
                log_prob = (
                    torch.lgamma(truth + dispersion)
                    - torch.lgamma(dispersion)
                    - torch.lgamma(truth + 1)
                    + dispersion * (torch.log(dispersion) - torch.log(total))
                    + truth * (torch.log(rate) - torch.log(total))
                )
                row_loss = -log_prob
                exposure = (minutes[valid] / 80.0).clamp(0.05, 1.0)
                loss = (row_loss * exposure).sum() / exposure.sum()
            losses.append(self._weighted_task(event, loss))
        if not losses:
            return torch.zeros((), device=minutes.device, requires_grad=True)
        return torch.stack(losses).mean()

    def fit(self, train: pd.DataFrame,
            validation: pd.DataFrame | None = None) -> "ExposureRateNeural":
        torch = self._torch()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.encoder.fit(train)
        minutes = pd.to_numeric(train["minutes"], errors="coerce").fillna(0).to_numpy(float)
        self.active_events = set()
        for event in self.events:
            available = train[f"available__{event}"].fillna(False).astype(bool).to_numpy()
            eligible = available if distribution_family(event) == "bernoulli" else (
                available & (minutes > 0)
            )
            if int(eligible.sum()) >= 20:
                self.active_events.add(event)
        self.event_caps = {}
        for event in self.active_events:
            if distribution_family(event) == "bernoulli":
                self.event_caps[event] = 1.0
                continue
            valid = (
                train[f"available__{event}"].fillna(False).astype(bool).to_numpy()
                & (minutes > 0)
            )
            values = pd.to_numeric(train.loc[valid, event], errors="coerce").to_numpy(float)
            rates = values / np.maximum(minutes[valid], 1e-6) * 80.0
            self.event_caps[event] = float(
                np.quantile(rates, self.config.rate_cap_quantile)
            )
        self.cardinalities = [
            len(self.encoder.categories[col]) for col in CATEGORICAL_FEATURES
        ]
        train_arrays = self._arrays(train)
        validation_arrays = self._arrays(validation) if validation is not None and len(validation) else None
        self.network = self._build_network(train_arrays[1].shape[1])
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.network.to(device)
        tensors = [torch.from_numpy(a) for a in train_arrays]
        dataset = torch.utils.data.TensorDataset(*tensors)
        row_weights = training_weights(
            train, self.config.weighting, self.config.time_half_life_days,
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(row_weights.astype("float64")),
            num_samples=len(train), replacement=True,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, sampler=sampler,
        )
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.config.epochs, 1),
        )
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.config.epochs):
            self.network.train()
            for batch in loader:
                batch = [item.to(device) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.network(batch[0], batch[1]), *batch[2:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                optimizer.step()
                with torch.no_grad():
                    for value in self.network.task_log_vars.values():
                        value.clamp_(-2.0, 2.0)
            scheduler.step()
            score = (
                self._validation_loss(validation_arrays, device)
                if validation_arrays is not None else float(loss.detach().cpu())
            )
            print(
                f"v3 epoch {epoch + 1:02d}/{self.config.epochs}: validation_loss={score:.5f}",
                flush=True,
            )
            if score < best - 1e-4:
                best = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.cpu()
        return self

    def _validation_loss(self, arrays, device) -> float:
        torch = self._torch()
        self.network.eval()
        with torch.no_grad():
            tensors = [torch.from_numpy(a).to(device) for a in arrays]
            return float(self._loss(self.network(tensors[0], tensors[1]), *tensors[2:]).cpu())

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        torch = self._torch()
        if self.network is None:
            raise RuntimeError("ExposureRateNeural is not fit")
        cat, numeric = self.encoder.transform(frame)
        self.network.eval()
        with torch.no_grad():
            output = self.network(
                torch.from_numpy(cat.astype("int64")),
                torch.from_numpy(numeric.astype("float32")),
            )
        p_play = torch.sigmoid(output["appearance"]).numpy()
        if not self.config.use_appearance:
            p_play = np.ones(len(frame))
        minute_raw = output["minutes"].numpy()
        minute_var = np.exp(np.clip(minute_raw[:, 1], -6, 6))
        conditional_minutes = np.clip(
            np.expm1(minute_raw[:, 0] + minute_var / 2), 1, 80,
        )
        expected_minutes = np.clip(p_play * conditional_minutes, 0, 80)
        event_raw = {name: value.numpy() for name, value in output["events"].items()}
        predictions = []
        for i, row in enumerate(frame.itertuples(index=False)):
            events = {}
            for event in sorted(self.active_events):
                raw = event_raw[event][i]
                family = distribution_family(event)
                if family == "bernoulli":
                    mean = float(1 / (1 + np.exp(-raw[0])) * p_play[i])
                    dispersion = 1.0
                elif family == "lognormal":
                    variance = float(np.exp(np.clip(raw[1], -6, 6)))
                    rate = float(np.clip(
                        np.expm1(raw[0] + variance / 2), 0,
                        self.event_caps.get(event, np.inf),
                    ))
                    mean = rate * float(expected_minutes[i]) / 80.0
                    dispersion = max((np.exp(variance) - 1) * max(mean * mean, 1e-6), 1e-6)
                else:
                    rate = float(np.clip(
                        np.logaddexp(0, raw[0]), 0,
                        self.event_caps.get(event, np.inf),
                    ))
                    mean = rate * float(expected_minutes[i]) / 80.0
                    dispersion = float(np.logaddexp(0, raw[1]) + 0.05)
                events[event] = EventDistribution(family, max(mean, 0.0), dispersion)
            mmean = float(expected_minutes[i])
            predictions.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=str(row.player_id),
                player_name=str(row.player_name), team=str(row.team),
                opponent=str(row.opponent), position=str(row.position),
                is_forward=bool(row.is_forward), events=events,
                minutes=EventDistribution("lognormal", mmean, max(25.0, mmean * 0.5)),
                metadata={
                    "model": "exposure_rate_neural_v3",
                    "p_play": float(p_play[i]),
                    "conditional_minutes": float(conditional_minutes[i]),
                    "competition_level": str(row.competition_level),
                },
            ))
        return predictions

    def save(self, path: Path) -> None:
        torch = self._torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config.to_dict(), "events": self.events,
            "encoder": self.encoder, "cardinalities": self.cardinalities,
            "active_events": sorted(self.active_events),
            "event_caps": self.event_caps,
            "state": self.network.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path) -> "ExposureRateNeural":
        torch = cls._torch()
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            config=NeuralV3Config.from_dict(artifact["config"]),
            events=tuple(artifact["events"]),
        )
        model.encoder = artifact["encoder"]
        model.cardinalities = artifact["cardinalities"]
        model.active_events = set(artifact["active_events"])
        model.event_caps = artifact.get("event_caps", {})
        model.network = model._build_network(len(model.encoder.numeric_columns))
        model.network.load_state_dict(artifact["state"])
        return model
