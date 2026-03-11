"""Tests for data_collection.cli module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from energy_modelling.data_collection.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture()
def _mock_all_steps(mocker: MockerFixture) -> dict[str, MagicMock]:
    """Mock all download/join functions so no real API calls are made."""
    mocks = {
        "prices": mocker.patch(
            "energy_modelling.data_collection.cli.download_prices",
            return_value=Path("data/raw/prices_da.parquet"),
        ),
        "generation": mocker.patch(
            "energy_modelling.data_collection.cli.download_generation",
            return_value=Path("data/raw/generation.parquet"),
        ),
        "weather": mocker.patch(
            "energy_modelling.data_collection.cli.download_weather",
            return_value=Path("data/raw/weather.parquet"),
        ),
        "join": mocker.patch(
            "energy_modelling.data_collection.cli.join_datasets",
            return_value=Path("data/processed/dataset_de_lu.parquet"),
        ),
    }
    return mocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCLI:
    @pytest.mark.usefixtures("_mock_all_steps")
    def test_help(self, runner: CliRunner) -> None:
        """--help should exit cleanly and show usage."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Download and join energy market data" in result.output

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_default_runs_all_steps(
        self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]
    ) -> None:
        """Without --step, all four steps should run."""
        result = runner.invoke(main, ["--years", "2024"])
        assert result.exit_code == 0
        _mock_all_steps["prices"].assert_called_once()
        _mock_all_steps["generation"].assert_called_once()
        _mock_all_steps["weather"].assert_called_once()
        _mock_all_steps["join"].assert_called_once()

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_step_prices_only(
        self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]
    ) -> None:
        """--step prices should only call download_prices."""
        result = runner.invoke(main, ["--years", "2024", "--step", "prices"])
        assert result.exit_code == 0
        _mock_all_steps["prices"].assert_called_once()
        _mock_all_steps["generation"].assert_not_called()
        _mock_all_steps["weather"].assert_not_called()
        _mock_all_steps["join"].assert_not_called()

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_step_generation_only(
        self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]
    ) -> None:
        result = runner.invoke(main, ["--years", "2024", "--step", "generation"])
        assert result.exit_code == 0
        _mock_all_steps["generation"].assert_called_once()
        _mock_all_steps["prices"].assert_not_called()

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_step_weather_only(
        self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]
    ) -> None:
        result = runner.invoke(main, ["--years", "2024", "--step", "weather"])
        assert result.exit_code == 0
        _mock_all_steps["weather"].assert_called_once()
        _mock_all_steps["prices"].assert_not_called()

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_step_join_only(self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]) -> None:
        result = runner.invoke(main, ["--step", "join"])
        assert result.exit_code == 0
        _mock_all_steps["join"].assert_called_once()
        _mock_all_steps["prices"].assert_not_called()

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_force_flag(self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]) -> None:
        """--force should be passed through to download functions."""
        result = runner.invoke(main, ["--years", "2024", "--step", "prices", "--force"])
        assert result.exit_code == 0
        call_kwargs = _mock_all_steps["prices"].call_args[1]
        assert call_kwargs["force"] is True

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_kaggle_flag(self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]) -> None:
        """--kaggle should be passed through to join_datasets."""
        result = runner.invoke(main, ["--step", "join", "--kaggle"])
        assert result.exit_code == 0
        call_kwargs = _mock_all_steps["join"].call_args[1]
        assert call_kwargs["kaggle"] is True

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_multiple_years(self, runner: CliRunner, _mock_all_steps: dict[str, MagicMock]) -> None:
        """Multiple --years flags should be collected."""
        result = runner.invoke(main, ["--years", "2023", "--years", "2024", "--step", "prices"])
        assert result.exit_code == 0
        config_arg = _mock_all_steps["prices"].call_args[0][0]
        assert config_arg.years == [2023, 2024]

    @pytest.mark.usefixtures("_mock_all_steps")
    def test_invalid_step(self, runner: CliRunner) -> None:
        """An invalid --step should fail."""
        result = runner.invoke(main, ["--step", "invalid"])
        assert result.exit_code != 0
