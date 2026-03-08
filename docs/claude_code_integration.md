# Using SciLink with Claude Code

SciLink can run as an MCP (Model Context Protocol) server, making its analysis and planning tools available directly inside Claude Code. This lets you analyze experimental data through natural conversation — Claude calls SciLink's tools automatically based on your requests.

## Setup

### 1. Install SciLink with MCP support

```bash
pip install -e ".[mcp]"
```

### 2. Register the MCP server

```bash
claude mcp add scilink -s user \
  -e GEMINI_API_KEY=your-gemini-key \
  -e UNSAFE_EXECUTION_OK=true \
  -e "PATH=$(dirname $(which python)):$PATH" \
  -- $(which scilink) serve --mode analyze
```

**Flags explained:**
- `-s user` — available from any directory (use `-s project` to limit to one repo)
- `-e GEMINI_API_KEY=...` — API key for the LLM backend (also supports `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- `-e UNSAFE_EXECUTION_OK=true` — allows code execution without a sandbox prompt (required since there's no terminal to approve interactively)
- `-e PATH=...` — ensures the `python` command is found during code execution
- `--mode analyze` — expose analysis tools (use `plan` for planning tools, or `both` for all)

### 3. Restart Claude Code

Start a new Claude Code session. You should see the SciLink MCP server connecting on startup.

## Available tools

Once connected, Claude Code has access to these SciLink tools:

| Tool | Description |
|------|-------------|
| `scilink_examine_data` | Inspect a data file (type, shape, suggested agents) |
| `scilink_convert_metadata` | Convert a text description to structured metadata |
| `scilink_load_metadata` | Load metadata from a JSON file or directory |
| `scilink_select_agent` | Choose an analysis agent (FFT, SAM, Hyperspectral, CurveFitting) |
| `scilink_preview_image` | Preview a microscopy image |
| `scilink_run_analysis` | Run analysis with the selected agent |
| `scilink_list_results` | List completed analyses in the session |
| `scilink_get_recommendations` | Get follow-up experiment suggestions |
| `scilink_assess_novelty` | Literature search for novelty of findings |
| `scilink_synthesize_knowledge` | Build reusable knowledge from results |
| `scilink_save_checkpoint` | Save session state |
| `scilink_show_available_agents` | List available analysis agents |

## Usage examples

Just chat naturally with Claude Code. It will call the SciLink tools as needed.

### Analyze a spectrum

```
Examine /path/to/xps_data.csv and tell me what kind of data it is
```

### Set metadata and run analysis

```
This is XPS Ti 2p data from a TiO2 thin film collected with Al K-alpha at 1486.6 eV.
Load the metadata, select the curve fitting agent, and analyze with the xps skill.
```

### Analyze a microscopy image

```
Examine /path/to/sem_image.tif, preview it, and run particle segmentation
```

### Batch analysis

```
Examine the directory /path/to/spectra/ and run a series analysis
with temperature as the control variable
```

### Get follow-up suggestions

```
List my analysis results and get measurement recommendations for the most recent one
```

## Server options

The `scilink serve` command accepts several flags:

```bash
scilink serve --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemini-3.1-pro-preview` | LLM model for analysis agents |
| `--mode` | `both` | `analyze`, `plan`, or `both` |
| `--autonomy` | `autonomous` | `autonomous`, `supervised`, or `co-pilot` |
| `--session-dir` | auto-generated | Directory for session outputs |
| `--api-key` | from env vars | Override API key |
| `--base-url` | none | OpenAI-compatible endpoint |

## Planning tools

When using `--mode plan` or `--mode both`, additional planning tools are available:

| Tool | Description |
|------|-------------|
| `scilink_list_workspace_files` | List files in the session directory |
| `scilink_generate_initial_plan` | Generate an experimental plan |
| `scilink_generate_implementation_code` | Add implementation code to a plan |
| `scilink_run_economic_analysis` | Techno-economic analysis |
| `scilink_refine_plan_with_results` | Refine plan based on results |
| `scilink_refine_implementation_code` | Update implementation code |
| `scilink_analyze_file` | Analyze a result file for optimization |
| `scilink_analyze_batch` | Batch analysis for optimization |
| `scilink_reset_analysis_logic` | Reset the scalarizer |
| `scilink_run_optimization` | Run Bayesian optimization |
| `scilink_plan_save_checkpoint` | Save planning session state |
| `scilink_discard_plan` | Discard the current plan |
| `scilink_show_directory_guide` | Show project directory structure |

## Autonomy modes

Control how much approval SciLink requires before executing actions:

```bash
scilink serve --autonomy autonomous   # default — all tools run immediately
scilink serve --autonomy supervised   # high-impact tools pause for approval
scilink serve --autonomy co-pilot     # most tools pause for approval
```

In **supervised** and **co-pilot** modes, high-impact tools (like `run_analysis`, `run_optimization`) return a `needs_input` response instead of executing. The MCP client then shows the proposed action to the user and calls `scilink_respond` with `"yes"` to approve or `"no"` to cancel.

Tools that require approval:
- **Co-pilot**: `run_analysis`, `select_agent`, `assess_novelty`, `get_recommendations`, `run_optimization`, `generate_initial_plan`, `generate_implementation_code`, `run_economic_analysis`, `discard_plan`
- **Supervised**: `run_analysis`, `run_optimization`, `discard_plan`

## Session outputs

Analysis results are saved to `~/scilink_mcp_sessions/session_<timestamp>/`. Each analysis run creates a subdirectory with:
- Fitted curves and plots
- `analysis_results.json` with detailed findings
- Scientific claims and recommendations

## Troubleshooting

### Server not connecting
Verify the server works standalone:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | scilink serve --mode analyze 2>/dev/null
```

### `python` not found during analysis
Ensure the `PATH` env var in the MCP config includes the directory containing `python`. Use:
```bash
-e "PATH=$(dirname $(which python)):$PATH"
```

### Tools not appearing
- Run `claude mcp list` to verify the server is registered
- Make sure `scilink serve` is available (the feature branch must be installed)
- Restart Claude Code after adding the MCP server

### Managing the server
```bash
claude mcp list              # list configured servers
claude mcp get scilink       # show config details
claude mcp remove scilink    # remove the server
```
