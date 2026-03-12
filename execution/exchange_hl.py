from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


MAINNET_API_URL = "https://api.hyperliquid.xyz"
TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"


@dataclass(frozen=True)
class VenueConfig:
    mode: str = "paper"
    network: str = "mainnet"
    account_address: str | None = None
    vault_address: str | None = None
    secret_key_env: str = "HL_SECRET_KEY"
    paper_account_value: float = 10_000.0
    min_trade_notional_usd: float = 25.0
    max_single_order_notional_usd: float = 500.0
    slippage: float = 0.01
    state_path: str = "execution/state.json"
    log_dir: str = "execution/logs"
    kill_switch_path: str = "execution/STOP"
    max_data_lag_hours: float = 3.0
    default_leverage: int = 2
    leverage_overrides: dict[str, int] = field(default_factory=lambda: {"BTC": 3, "ETH": 3})
    target_gross_notional_usd: float | None = None

    def base_url(self) -> str:
        return TESTNET_API_URL if self.network == "testnet" else MAINNET_API_URL

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["base_url"] = self.base_url()
        return payload


class HyperliquidExecutionClient:
    def __init__(self, venue: VenueConfig):
        self.venue = venue
        self._info = None
        self._exchange = None
        if venue.mode in {"paper", "live"}:
            self._info = self._build_info()
        if venue.mode == "live":
            self._exchange = self._build_exchange()

    def current_positions(self, state: dict[str, Any]) -> dict[str, float]:
        if self.venue.mode == "paper":
            return {coin: float(sz) for coin, sz in state.get("paper_positions", {}).items()}
        address = self._require_account_address()
        raw = self._info.user_state(address)
        positions = {}
        for row in raw.get("assetPositions", []):
            position = row.get("position", {})
            coin = position.get("coin")
            if not coin:
                continue
            positions[coin] = float(position.get("szi", 0.0))
        return positions

    def account_value(self) -> float:
        if self.venue.mode == "paper":
            return float(self.venue.paper_account_value)
        address = self._require_account_address()
        raw = self._info.user_state(address)
        summary = raw.get("marginSummary") or raw.get("crossMarginSummary") or {}
        return float(summary.get("accountValue", 0.0))

    def mid_prices(self, coins: list[str]) -> dict[str, float]:
        if self._info is None:
            return {}
        mids = self._info.all_mids()
        return {coin: float(mids[coin]) for coin in coins if coin in mids}

    def apply_instructions(self, instructions, state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.venue.mode == "paper":
            return self._apply_paper(instructions, state)
        return self._apply_live(instructions, state)

    def _apply_paper(self, instructions, state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        positions = dict(state.get("paper_positions", {}))
        fills = []
        for instruction in instructions:
            if instruction.status != "trade":
                fills.append({"coin": instruction.coin, "status": instruction.status, "reason": instruction.reason})
                continue
            positions[instruction.coin] = instruction.target_size
            if abs(positions[instruction.coin]) <= 1e-12:
                positions.pop(instruction.coin, None)
            fills.append(
                {
                    "coin": instruction.coin,
                    "status": "filled",
                    "side": instruction.side,
                    "size": instruction.delta_size,
                    "price": instruction.price,
                    "mode": "paper",
                }
            )
        next_state = dict(state)
        next_state["paper_positions"] = positions
        return fills, next_state

    def _apply_live(self, instructions, state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        fills = []
        for instruction in instructions:
            if instruction.status != "trade":
                fills.append({"coin": instruction.coin, "status": instruction.status, "reason": instruction.reason})
                continue
            leverage = self._desired_leverage(instruction.coin)
            leverage_response = self._set_leverage(instruction.coin, leverage)
            if leverage_response is not None:
                fills.append({"coin": instruction.coin, "status": "set_leverage", "leverage": leverage, "response": leverage_response})
            if instruction.current_size != 0.0 and instruction.target_size == 0.0:
                response = self._submit_close(instruction.coin, abs(instruction.current_size), instruction.price)
                fills.append({"coin": instruction.coin, "status": "submitted_close", "response": response})
                continue
            if instruction.current_size * instruction.target_size < 0.0 and instruction.current_size != 0.0:
                close_response = self._submit_close(instruction.coin, abs(instruction.current_size), instruction.price)
                fills.append({"coin": instruction.coin, "status": "submitted_close", "response": close_response})
                open_response = self._submit_open_with_retry(
                    instruction.coin,
                    is_buy=instruction.target_size > 0.0,
                    sz=abs(instruction.target_size),
                    px=instruction.price,
                    slippage=self.venue.slippage,
                    size_decimals=getattr(instruction, 'size_decimals', 4),
                )
                fills.append({"coin": instruction.coin, "status": "submitted_open", "response": open_response})
                continue
            response = self._submit_open_with_retry(
                instruction.coin,
                is_buy=instruction.delta_size > 0.0,
                sz=abs(instruction.delta_size),
                px=instruction.price,
                slippage=self.venue.slippage,
                size_decimals=getattr(instruction, 'size_decimals', 4),
            )
            fills.append({"coin": instruction.coin, "status": "submitted_delta", "response": response})
        return fills, dict(state)


    def _submit_close(self, coin: str, sz: float, px: float):
        return self._exchange.market_close(coin, sz=abs(sz), px=px, slippage=self.venue.slippage)

    def _submit_open_with_retry(self, coin: str, *, is_buy: bool, sz: float, px: float, slippage: float, size_decimals: int):
        attempt_size = float(sz)
        responses = []
        for drop in range(0, min(int(size_decimals), 4) + 1):
            adj_decimals = max(int(size_decimals) - drop, 0)
            attempt_size = self._round_toward_zero(attempt_size, adj_decimals)
            if attempt_size <= 0:
                continue
            resp = self._exchange.market_open(coin, is_buy=is_buy, sz=attempt_size, px=px, slippage=slippage)
            responses.append({'size': attempt_size, 'response': resp})
            if not self._response_has_error(resp):
                return {'attempts': responses, 'final_size': attempt_size}
        return {'attempts': responses, 'error': 'all_attempts_failed'}

    def _response_has_error(self, response) -> bool:
        payload = response.get('response') if isinstance(response, dict) else None
        if isinstance(payload, str):
            return True
        statuses = (((payload or {}).get('data') or {}).get('statuses')) if isinstance(payload, dict) else None
        if not statuses:
            return False
        for item in statuses:
            if 'error' in item:
                return True
        return False

    def _round_toward_zero(self, value: float, decimals: int) -> float:
        factor = 10 ** int(decimals)
        if factor <= 0:
            return float(value)
        if value >= 0:
            return float(int(value * factor) / factor)
        return float(-int(abs(value) * factor) / factor)

    def _desired_leverage(self, coin: str) -> int:
        leverage = int(self.venue.leverage_overrides.get(coin, self.venue.default_leverage))
        return max(1, leverage)

    def _set_leverage(self, coin: str, leverage: int):
        if self._exchange is None or self.venue.mode != "live":
            return None
        try:
            return self._exchange.update_leverage(leverage, coin, is_cross=True)
        except Exception as exc:
            return {"error": str(exc), "coin": coin, "requested_leverage": leverage}

    def _build_info(self):
        try:
            from hyperliquid.info import Info
        except ImportError as exc:
            raise RuntimeError("Install optional execution dependencies with `pip install -e .[execution]`.") from exc
        return Info(self.venue.base_url(), skip_ws=True)

    def _build_exchange(self):
        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
        except ImportError as exc:
            raise RuntimeError("Install optional execution dependencies with `pip install -e .[execution]`.") from exc
        secret = os.environ.get(self.venue.secret_key_env)
        if not secret:
            raise RuntimeError(f"Environment variable {self.venue.secret_key_env} is required for live execution.")
        wallet = Account.from_key(secret)
        return Exchange(
            wallet,
            self.venue.base_url(),
            account_address=self.venue.account_address,
            vault_address=self.venue.vault_address,
        )

    def _require_account_address(self) -> str:
        address = self.venue.account_address or self.venue.vault_address
        if not address:
            raise RuntimeError("`account_address` or `vault_address` is required for paper/live position reconciliation.")
        return address
