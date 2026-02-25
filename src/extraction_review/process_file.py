import asyncio
import logging
from typing import Annotated, Literal

from llama_cloud import AsyncLlamaCloud
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import EXTRACTED_DATA_COLLECTION, ParseConfig

logger = logging.getLogger(__name__)


class ComplianceStartEvent(StartEvent):
    """Start event for compliance analysis with two documents."""

    regulation_file_id: str = Field(description="File ID of the government regulation PDF")
    policy_file_id: str = Field(description="File ID of the company policy PDF")


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class ParseJobsStartedEvent(Event):
    """Both documents are being parsed."""

    pass


class ConflictItem(BaseModel):
    """A single compliance conflict between regulation and policy."""

    conflict_summary: str = Field(description="Brief description of the conflict")
    regulation_page: int = Field(description="Page number in the regulation document")
    regulation_text: str = Field(description="Text snippet from the regulation")
    policy_page: int | None = Field(description="Page number in policy, or null if missing")
    policy_text: str | None = Field(description="Text snippet from policy, or null if missing")
    severity: Literal["critical", "major", "minor"] = Field(description="Severity level")
    recommendation: str = Field(description="Suggested action to resolve")


class ComplianceAnalysisResult(BaseModel):
    """Complete compliance analysis results."""

    conflicts: list[ConflictItem] = Field(default_factory=list)
    total_conflicts: int = Field(default=0)
    critical_count: int = Field(default=0)
    major_count: int = Field(default=0)
    minor_count: int = Field(default=0)


class AnalysisCompleteEvent(Event):
    """Compliance analysis is complete."""

    result: ComplianceAnalysisResult


class ComplianceState(BaseModel):
    regulation_file_id: str | None = None
    regulation_filename: str | None = None
    regulation_parse_job_id: str | None = None
    policy_file_id: str | None = None
    policy_filename: str | None = None
    policy_parse_job_id: str | None = None


def get_llm() -> LLM:
    """LLM for compliance analysis."""
    return OpenAI(model="gpt-4o", temperature=0)


class ComplianceAnalysisWorkflow(Workflow):
    """Analyze company policy against government regulations to identify compliance gaps.

    Compares two documents and identifies every instance where the company policy
    contradicts or fails to meet the regulatory requirements.
    """

    @step()
    async def start_parsing(
        self,
        event: ComplianceStartEvent,
        ctx: Context[ComplianceState],
        llama_cloud_client: Annotated[AsyncLlamaCloud, Resource(get_llama_cloud_client)],
        parse_config: Annotated[
            ParseConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="parse",
                label="Parse Settings",
                description="Configuration for document parsing quality",
            ),
        ],
    ) -> ParseJobsStartedEvent:
        """Parse both regulation and policy documents to extract text with page numbers."""
        ctx.write_event_to_stream(
            Status(level="info", message="Starting document analysis...")
        )

        # Get file metadata for both documents
        from llama_cloud.types.file_query_params import Filter

        files = await llama_cloud_client.files.query(
            filter=Filter(file_ids=[event.regulation_file_id, event.policy_file_id])
        )
        file_map = {f.id: f.name for f in files.items}

        regulation_name = file_map.get(event.regulation_file_id, "regulation.pdf")
        policy_name = file_map.get(event.policy_file_id, "policy.pdf")

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Parsing regulation: {regulation_name} and policy: {policy_name}",
            )
        )

        # Start parse jobs for both documents in parallel
        regulation_job, policy_job = await asyncio.gather(
            llama_cloud_client.parsing.create(
                tier=parse_config.settings.tier,
                version=parse_config.settings.version,
                file_id=event.regulation_file_id,
                project_id=project_id,
            ),
            llama_cloud_client.parsing.create(
                tier=parse_config.settings.tier,
                version=parse_config.settings.version,
                file_id=event.policy_file_id,
                project_id=project_id,
            ),
        )

        async with ctx.store.edit_state() as state:
            state.regulation_file_id = event.regulation_file_id
            state.regulation_filename = regulation_name
            state.regulation_parse_job_id = regulation_job.id
            state.policy_file_id = event.policy_file_id
            state.policy_filename = policy_name
            state.policy_parse_job_id = policy_job.id

        return ParseJobsStartedEvent()

    @step()
    async def analyze_compliance(
        self,
        event: ParseJobsStartedEvent,
        ctx: Context[ComplianceState],
        llama_cloud_client: Annotated[AsyncLlamaCloud, Resource(get_llama_cloud_client)],
        llm: Annotated[LLM, Resource(get_llm)],
    ) -> AnalysisCompleteEvent:
        """Wait for parsing to complete and analyze for compliance conflicts."""
        state = await ctx.store.get_state()

        ctx.write_event_to_stream(
            Status(level="info", message="Extracting text from documents...")
        )

        # Wait for both parse jobs to complete
        await asyncio.gather(
            llama_cloud_client.parsing.wait_for_completion(state.regulation_parse_job_id),
            llama_cloud_client.parsing.wait_for_completion(state.policy_parse_job_id),
        )

        # Get results with page-level markdown
        regulation_result, policy_result = await asyncio.gather(
            llama_cloud_client.parsing.get(state.regulation_parse_job_id, expand=["markdown"]),
            llama_cloud_client.parsing.get(state.policy_parse_job_id, expand=["markdown"]),
        )

        # Build page-indexed text for each document
        regulation_pages = self._extract_pages(regulation_result)
        policy_pages = self._extract_pages(policy_result)

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Analyzing {len(regulation_pages)} regulation pages against {len(policy_pages)} policy pages...",
            )
        )

        # Format documents for analysis
        regulation_text = self._format_document_with_pages(regulation_pages, "REGULATION")
        policy_text = self._format_document_with_pages(policy_pages, "COMPANY POLICY")

        # Analyze for compliance conflicts using LLM
        result = await self._analyze_conflicts(llm, regulation_text, policy_text)

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Found {result.total_conflicts} compliance conflicts ({result.critical_count} critical, {result.major_count} major, {result.minor_count} minor)",
            )
        )

        return AnalysisCompleteEvent(result=result)

    @step()
    async def save_results(
        self,
        event: AnalysisCompleteEvent,
        ctx: Context[ComplianceState],
        llama_cloud_client: Annotated[AsyncLlamaCloud, Resource(get_llama_cloud_client)],
    ) -> StopEvent:
        """Save compliance analysis results for review."""
        state = await ctx.store.get_state()
        result = event.result

        # Build the data record
        data = {
            "regulation_file_id": state.regulation_file_id,
            "regulation_filename": state.regulation_filename,
            "policy_file_id": state.policy_file_id,
            "policy_filename": state.policy_filename,
            **result.model_dump(),
        }

        # Save to Agent Data
        item = await llama_cloud_client.beta.agent_data.agent_data(
            data=data,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )

        ctx.write_event_to_stream(
            Status(level="info", message="Compliance analysis complete and saved.")
        )

        return StopEvent(result=item.id)

    def _extract_pages(self, parse_result) -> dict[int, str]:
        """Extract page content from parse result."""
        pages = {}
        if parse_result.markdown and parse_result.markdown.pages:
            for page in parse_result.markdown.pages:
                page_num = page.page_number
                content = page.markdown if hasattr(page, "markdown") else ""
                if content:
                    pages[page_num] = content
        return pages

    def _format_document_with_pages(self, pages: dict[int, str], doc_type: str) -> str:
        """Format document content with page markers."""
        sections = [f"=== {doc_type} ==="]
        for page_num in sorted(pages.keys()):
            sections.append(f"\n--- Page {page_num} ---\n{pages[page_num]}")
        return "\n".join(sections)

    async def _analyze_conflicts(
        self, llm: LLM, regulation_text: str, policy_text: str
    ) -> ComplianceAnalysisResult:
        """Use LLM to identify compliance conflicts."""
        prompt = PromptTemplate(
            """You are a compliance analyst comparing a government regulation against a company policy.

Your task is to identify every instance where the company policy:
1. CONTRADICTS a requirement in the regulation
2. FAILS TO MEET a standard specified in the regulation
3. OMITS a required element that the regulation mandates

For each conflict found, provide:
- A clear summary of the conflict
- The exact page number and relevant text from the REGULATION
- The page number and text from the COMPANY POLICY (use null if the requirement is completely missing)
- Severity: "critical" (legal risk/major violation), "major" (significant gap), or "minor" (small discrepancy)
- A specific recommendation to resolve the issue

Be thorough and identify ALL conflicts, even minor ones. Quote the actual text from both documents.

{regulation_text}

{policy_text}

Analyze these documents and identify all compliance conflicts."""
        )

        result = await llm.astructured_predict(
            ComplianceAnalysisResult,
            prompt,
            regulation_text=regulation_text,
            policy_text=policy_text,
        )

        # Ensure counts are correct
        result.total_conflicts = len(result.conflicts)
        result.critical_count = sum(1 for c in result.conflicts if c.severity == "critical")
        result.major_count = sum(1 for c in result.conflicts if c.severity == "major")
        result.minor_count = sum(1 for c in result.conflicts if c.severity == "minor")

        return result


workflow = ComplianceAnalysisWorkflow(timeout=None)

if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        client = get_llama_cloud_client()
        regulation_file = await client.files.create(
            file=Path("regulation.pdf").open("rb"),
            purpose="parse",
        )
        policy_file = await client.files.create(
            file=Path("policy.pdf").open("rb"),
            purpose="parse",
        )
        await workflow.run(
            start_event=ComplianceStartEvent(
                regulation_file_id=regulation_file.id,
                policy_file_id=policy_file.id,
            )
        )

    asyncio.run(main())
