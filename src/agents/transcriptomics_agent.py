from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


# Import the old system's data manager and tools 
# Note: These imports will need to be adjusted based on your new framework's structure
try:
    # Try to import from the old system - adjust paths as needed
    import sys
    import os
    
    from ..memory.data_manager import DataManager
    from ..service.utils import get_logger
    from agent_tools.geo_service import GEOService
    from agent_tools.quality_service import QualityService
    from agent_tools.clustering_service import ClusteringService
    from agent_tools.enhanced_singlecell_service import EnhancedSingleCellService
    
    logger = get_logger(__name__)
    
    # Initialize data manager (this might need to be adjusted based on your new framework)
    data_manager = DataManager()
    
    # Transcriptomics tools adapted from the old system
    @tool
    def download_geo_dataset(geo_id: str) -> str:
        """Download dataset from GEO using accession number."""
        try:
            geo_service = GEOService(data_manager)
            result = geo_service.download_dataset(geo_id.strip())
            logger.info(f"Downloaded GEO dataset: {geo_id}")
            return result
        except Exception as e:
            logger.error(f"Error downloading GEO dataset {geo_id}: {e}")
            return f"Error downloading dataset: {str(e)}"

    @tool
    def get_data_summary(query: str = "") -> str:
        """Get summary of currently loaded data."""
        if not data_manager.has_data():
            return "No data loaded. Use download_geo_dataset to load data from GEO."
        
        try:
            summary = data_manager.get_data_summary()
            response = f"Data loaded: {summary['shape'][0]} cells × {summary['shape'][1]} genes"
            
            if 'source' in data_manager.current_metadata:
                response += f" from {data_manager.current_metadata['source']}"
            
            return response
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return f"Error getting data summary: {str(e)}"

    @tool
    def assess_data_quality(query: str) -> str:
        """Assess quality of RNA-seq data with QC metrics."""
        try:
            quality_service = QualityService(data_manager)
            result = quality_service.assess_quality()
            logger.info("Quality assessment completed")
            return result
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return f"Error assessing data quality: {str(e)}"

    @tool
    def cluster_cells(query: str) -> str:
        """Perform clustering and UMAP visualization."""
        try:
            clustering_service = ClusteringService(data_manager)
            result = clustering_service.cluster_and_visualize()
            logger.info("Clustering analysis completed")
            return result
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return f"Error performing clustering: {str(e)}"

    @tool
    def detect_doublets(query: str) -> str:
        """Detect doublets in single-cell data using Scrublet."""
        try:
            enhanced_sc_service = EnhancedSingleCellService(data_manager)
            result = enhanced_sc_service.detect_doublets()
            logger.info("Doublet detection completed")
            return result
        except Exception as e:
            logger.error(f"Error in doublet detection: {e}")
            return f"Error detecting doublets: {str(e)}"

    @tool
    def annotate_cell_types(query: str) -> str:
        """Annotate cell types based on marker genes."""
        try:
            enhanced_sc_service = EnhancedSingleCellService(data_manager)
            result = enhanced_sc_service.annotate_cell_types()
            logger.info("Cell type annotation completed")
            return result
        except Exception as e:
            logger.error(f"Error in cell type annotation: {e}")
            return f"Error annotating cell types: {str(e)}"

    @tool
    def find_marker_genes(query: str) -> str:
        """Find marker genes for clusters or cell types."""
        try:
            enhanced_sc_service = EnhancedSingleCellService(data_manager)
            result = enhanced_sc_service.find_marker_genes()
            logger.info("Marker gene analysis completed")
            return result
        except Exception as e:
            logger.error(f"Error finding marker genes: {e}")
            return f"Error finding marker genes: {str(e)}"

    # Tools list for the transcriptomics agent
    tools = [
        download_geo_dataset,
        get_data_summary,
        assess_data_quality,
        cluster_cells,
        detect_doublets,
        annotate_cell_types,
        find_marker_genes
    ]

except ImportError as e:
    logger = None
    print(f"Warning: Could not import transcriptomics dependencies: {e}")
    print("Falling back to basic web search functionality")
    
    # Fallback tools if transcriptomics imports fail
    web_search = DuckDuckGoSearchResults(name="WebSearch")
    tools = [web_search, calculator]


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a transcriptomics domain expert specializing in RNA-seq analysis (both single-cell and bulk).
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    <Task>
    You will receive specific analysis instructions from the supervisor. Your job is to:
    1. Check if data is available or needs to be downloaded
    2. Execute the requested analysis using available tools
    3. Provide a comprehensive summary of results with specific findings
    4. Use standard parameters unless specific ones are provided
    </Task>

    <Available Tools>
    You have access to these transcriptomics analysis tools:
    - download_geo_dataset: Download dataset from GEO using accession number (e.g., GSE109564)
    - get_data_summary: Get summary of currently loaded data
    - assess_data_quality: Assess quality of RNA-seq data with QC metrics
    - cluster_cells: Perform clustering and UMAP visualization
    - detect_doublets: Detect doublets in single-cell data using Scrublet
    - annotate_cell_types: Annotate cell types based on marker genes
    - find_marker_genes: Find marker genes for clusters or cell types
    </Available Tools>

    <Analysis Process>
    1. Always start by checking data availability with get_data_summary
    2. Download data from GEO if needed using download_geo_dataset
    3. Assess data quality if working with single-cell data
    4. Execute the specific analysis requested (clustering, marker genes, etc.)
    5. Provide clear results with specific metrics and findings

    <Guidelines>
    - Use standard, well-established parameters for analyses unless specific ones are provided
    - Always include specific numbers in your results (e.g., "15 clusters identified", "2,543 genes detected")
    - Explain what each analysis step accomplishes
    - If analysis fails, try once more with adjusted parameters
    - Be thorough but concise in your explanations
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


transcriptomics_agent = agent.compile()
