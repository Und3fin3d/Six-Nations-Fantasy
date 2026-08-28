"""Compact shared multi-task neural model for Apple-silicon/local training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import EventDistribution, RawPrediction
from .features import CATEGORICAL_FEATURES, FeatureEncoder
from .schema import EVENTS, distribution_family


@dataclass
class NeuralTrainingConfig:
    hidden: int = 128
    dropout: float = .12
    epochs: int = 12
    batch_size: int = 1024
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 17


class UniversalNeuralModel:
    """Shared embeddings/trunk with masked distributional raw-event heads."""

    def __init__(self, events: tuple[str, ...] = EVENTS, config: NeuralTrainingConfig | None = None):
        self.events = tuple(events)
        self.targets = ("minutes",) + self.events
        self.config = config or NeuralTrainingConfig()
        self.encoder = FeatureEncoder()
        self.network = None
        self._cardinalities: list[int] = []
        self.active_targets: set[str] = set()

    @staticmethod
    def _torch():
        try:
            import torch
            return torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for the primary unified model; install requirements-unified.txt"
            ) from exc

    def _build_network(self, n_numeric: int):
        torch = self._torch()
        nn = torch.nn
        cards = self._cardinalities
        dims = [min(32, max(4, int(round(c ** .25 * 4)))) for c in cards]

        class Net(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.embeddings = nn.ModuleList([nn.Embedding(c + 1, d) for c, d in zip(cards, dims)])
                width = sum(dims) + n_numeric
                h = outer.config.hidden
                self.trunk = nn.Sequential(
                    nn.Linear(width, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(outer.config.dropout),
                    nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(outer.config.dropout),
                )
                self.heads = nn.ModuleDict({name: nn.Linear(h, 2) for name in outer.targets})

            def forward(self, cat, num):
                embedded = [layer(cat[:, i]) for i, layer in enumerate(self.embeddings)]
                shared = self.trunk(torch.cat(embedded + [num], dim=1))
                return {name: head(shared) for name, head in self.heads.items()}

        return Net(self)

    def _arrays(self, frame: pd.DataFrame):
        cat, numeric = self.encoder.transform(frame)
        y = np.column_stack([pd.to_numeric(frame[t], errors="coerce") for t in self.targets]).astype("float32")
        mask = np.column_stack([
            frame[f"available__{t}"].fillna(False).to_numpy(bool) & np.isfinite(y[:, i])
            for i, t in enumerate(self.targets)
        ])
        y = np.nan_to_num(y, nan=0.0).clip(0)
        return cat.astype("int64"), numeric.astype("float32"), y, mask

    def _loss(self, output, y, mask):
        torch = self._torch()
        total = torch.zeros((), device=y.device)
        used = 0
        for i, target in enumerate(self.targets):
            if target not in self.active_targets:
                continue
            valid = mask[:, i]
            if not valid.any():
                continue
            raw = output[target][valid]
            truth = y[valid, i]
            family = distribution_family(target)
            if family == "bernoulli":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(raw[:, 0], truth.clamp(0, 1))
            elif family == "negative_binomial":
                mean = torch.nn.functional.softplus(raw[:, 0]) + 1e-5
                dispersion = torch.nn.functional.softplus(raw[:, 1]) + .05
                probs = mean / (dispersion + mean)
                loss = -torch.distributions.NegativeBinomial(total_count=dispersion, probs=probs).log_prob(truth).mean()
            else:
                log_y = torch.log1p(truth)
                mean = raw[:, 0]
                log_var = raw[:, 1].clamp(-6, 6)
                loss = .5 * (log_var + (log_y - mean) ** 2 / torch.exp(log_var))
                loss = loss.mean()
            total = total + loss
            used += 1
        return total / max(used, 1)

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> "UniversalNeuralModel":
        torch = self._torch()
        torch.manual_seed(self.config.seed)
        self.encoder.fit(train)
        self.active_targets = {
            target for target in self.targets
            if int(train[f"available__{target}"].fillna(False).sum()) >= 20
        }
        self._cardinalities = [len(self.encoder.categories[c]) for c in CATEGORICAL_FEATURES]
        train_arrays = self._arrays(train)
        val_arrays = self._arrays(validation) if validation is not None and len(validation) else None
        self.network = self._build_network(train_arrays[1].shape[1])
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.network.to(device)
        dataset = torch.utils.data.TensorDataset(*[torch.from_numpy(a) for a in train_arrays])
        generator = torch.Generator().manual_seed(self.config.seed)
        domain = train["competition_level"].fillna("unknown")
        counts = domain.map(domain.value_counts()).to_numpy(float)
        weights = torch.from_numpy((1.0 / counts).astype("float64"))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train), replacement=True, generator=generator,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, sampler=sampler,
        )
        opt = torch.optim.AdamW(self.network.parameters(), lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.config.epochs):
            self.network.train()
            for cat, num, y, mask in loader:
                cat, num, y, mask = cat.to(device), num.to(device), y.to(device), mask.to(device)
                opt.zero_grad(set_to_none=True)
                loss = self._loss(self.network(cat, num), y, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                opt.step()
            score = self._validation_loss(val_arrays, device) if val_arrays is not None else float(loss.detach())
            print(f"epoch {epoch + 1:02d}/{self.config.epochs}: validation_nll={score:.5f}", flush=True)
            if score < best - 1e-4:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
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
            cat, num, y, mask = [torch.from_numpy(a).to(device) for a in arrays]
            return float(self._loss(self.network(cat, num), y, mask).cpu())

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        torch = self._torch()
        if self.network is None:
            raise RuntimeError("model is not fit")
        cat, numeric = self.encoder.transform(frame)
        self.network.eval()
        with torch.no_grad():
            out = self.network(torch.from_numpy(cat.astype("int64")), torch.from_numpy(numeric.astype("float32")))
        params = {k: v.numpy() for k, v in out.items()}
        predictions = []
        for i, row in enumerate(frame.itertuples(index=False)):
            events = {}
            for event in self.events:
                if event not in self.active_targets:
                    continue
                raw = params[event][i]
                family = distribution_family(event)
                if family == "bernoulli":
                    mean, dispersion = 1 / (1 + np.exp(-raw[0])), 1.0
                elif family == "negative_binomial":
                    mean = float(np.logaddexp(0, raw[0]))
                    dispersion = float(np.logaddexp(0, raw[1]) + .05)
                else:
                    log_var = float(np.clip(raw[1], -6, 6))
                    mean = float(np.expm1(raw[0] + np.exp(log_var) / 2).clip(0))
                    dispersion = float((np.exp(log_var) - 1) * max(mean * mean, 1e-6))
                events[event] = EventDistribution(family, float(mean), max(float(dispersion), 1e-6))
            mraw = params["minutes"][i]
            mvar = float(np.exp(np.clip(mraw[1], -6, 6)))
            mmean = float(np.expm1(mraw[0] + mvar / 2).clip(0, 80))
            predictions.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=str(row.player_id), player_name=str(row.player_name),
                team=str(row.team), opponent=str(row.opponent), position=str(row.position),
                is_forward=bool(row.is_forward), events=events,
                minutes=EventDistribution("lognormal", mmean, max((np.exp(mvar)-1)*max(mmean*mmean,1e-6),1e-6)),
                metadata={"model": "universal_neural", "competition_level": row.competition_level},
            ))
        return predictions

    def save(self, path: Path) -> None:
        torch = self._torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "events": self.events, "config": self.config.__dict__, "encoder": self.encoder,
            "cardinalities": self._cardinalities, "active_targets": sorted(self.active_targets),
            "state": self.network.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path) -> "UniversalNeuralModel":
        torch = cls._torch()
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(tuple(artifact["events"]), NeuralTrainingConfig(**artifact["config"]))
        model.encoder = artifact["encoder"]
        model._cardinalities = artifact["cardinalities"]
        default_inactive = {"fifty_22", "lineout_steals", "scrums_won", "kicks_retained", "potm"}
        model.active_targets = set(artifact.get(
            "active_targets", [t for t in model.targets if t not in default_inactive]
        ))
        model.network = model._build_network(len(model.encoder.numeric_columns))
        model.network.load_state_dict(artifact["state"])
        return model
