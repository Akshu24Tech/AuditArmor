"""
Tests for the compliance analysis workflow.
"""

import json
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from extraction_review.clients import fake
from extraction_review.config import EXTRACTED_DATA_COLLECTION
from extraction_review.metadata_workflow import MetadataResponse
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import (
    ComplianceAnalysisResult,
    ComplianceStartEvent,
    ConflictItem,
)
from extraction_review.process_file import workflow as process_file_workflow
from workflows.events import StartEvent


def get_extraction_schema() -> dict:
    """Load the extraction schema from the unified config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text())
    return config["extract"]["json_schema"]


def mock_llm_response() -> ComplianceAnalysisResult:
    """Create a mock compliance analysis result."""
    return ComplianceAnalysisResult(
        conflicts=[
            ConflictItem(
                conflict_summary="Policy allows 30-day data retention but regulation requires 7-day limit",
                regulation_page=1,
                regulation_text="Data must be deleted within 7 days of collection.",
                policy_page=1,
                policy_text="Data may be retained for up to 30 days.",
                severity="critical",
                recommendation="Update policy to require 7-day data deletion.",
            ),
        ],
        total_conflicts=1,
        critical_count=1,
        major_count=0,
        minor_count=0,
    )


@pytest.mark.asyncio
async def test_process_file_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    if fake is None:
        warnings.warn(
            "Skipping test because it cannot be mocked. Set `FAKE_LLAMA_CLOUD=true` in your environment to enable this test..."
        )
        return

    # Load two test files (regulation and policy)
    regulation_file_id = fake.files.preload(path="tests/files/test.pdf")
    policy_file_id = fake.files.preload(path="tests/files/test.pdf")

    # Mock the LLM's structured prediction
    mock_result = mock_llm_response()
    with patch(
        "extraction_review.process_file.OpenAI"
    ) as mock_openai_class:
        mock_llm = AsyncMock()
        mock_llm.astructured_predict = AsyncMock(return_value=mock_result)
        mock_openai_class.return_value = mock_llm

        result = await process_file_workflow.run(
            start_event=ComplianceStartEvent(
                regulation_file_id=regulation_file_id,
                policy_file_id=policy_file_id,
            )
        )

    assert result is not None
    # All generated agent data IDs are alphanumeric strings with 7 characters
    assert isinstance(result, str)
    assert len(result) == 7


@pytest.mark.asyncio
async def test_metadata_workflow() -> None:
    result = await metadata_workflow.run(start_event=StartEvent())
    assert isinstance(result, MetadataResponse)
    assert result.extracted_data_collection == EXTRACTED_DATA_COLLECTION
    assert result.json_schema == get_extraction_schema()
