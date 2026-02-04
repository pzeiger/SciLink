import os
import sys
import subprocess
import re
import uuid
import tempfile
import logging

from .auth import get_api_key

DEFAULT_TIMEOUT = 120

# Description of what the LLM is instructed to do
LLM_EXECUTION_DESCRIPTION = """
WHAT THIS SYSTEM DOES:
  An LLM (Large Language Model) will generate and execute Python code on your
  machine to perform scientific data analysis. The LLM is instructed to:

  • Write Python scripts for curve fitting, spectral analysis, and data processing
  • Use scientific libraries: NumPy, SciPy, scikit-learn, matplotlib, pandas
  • Read input data files you provide (CSV, NPY, TXT, etc.)
  • Save output files (plots, results) to the designated output directory
  • Execute the generated code automatically without manual review

  The LLM is NOT instructed to:
  • Access the internet or make network requests
  • Modify system files or install software
  • Access files outside the working/output directories
  • Execute shell commands beyond running Python scripts

  However, AI-generated code can behave unexpectedly. A sandbox provides
  protection against unintended actions.
""".strip()


def is_in_colab():
    """Check for Google Colab environment."""
    if 'COLAB_GPU' in os.environ or 'GCE_METADATA_TIMEOUT' in os.environ:
        return True
    if 'google.colab' in sys.modules:
        return True
    return False


def check_security_sandbox_indicators(verbose=False):
    """Check for OS-level sandboxing indicators."""
    score = 0
    positive_indicators = []

    # Tier 1: High-Confidence Environments (Score: 10)
    if is_in_colab():
        score += 10
        positive_indicators.append("google_colab")
        if verbose:
            logging.info("High-Confidence Indicator: Google Colab environment detected.")
        return score, positive_indicators

    # Tier 2: Strong Indicators (Score: 5)
    # Docker/Container check
    if os.path.exists('/.dockerenv') or ('docker' in (open('/proc/1/cgroup').read() if os.path.exists('/proc/1/cgroup') else '')):
        score += 5
        positive_indicators.append("docker_container")
        if verbose:
            logging.info("Strong Indicator: Docker or container environment detected.")

    # Virtual Machine check
    try:
        if sys.platform.startswith("linux"):
            result = subprocess.run(['systemd-detect-virt'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip() != 'none':
                score += 5
                positive_indicators.append(f"virtual_machine:{result.stdout.strip()}")
                if verbose:
                    logging.info(f"Strong Indicator: Virtual Machine detected ('{result.stdout.strip()}').")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Tier 3: Corroborating Evidence (Score: 2)
    try:
        mac = ':'.join(re.findall('..', f'{uuid.getnode():012x}'))
        vm_mac_prefixes = ["08:00:27", "00:05:69", "00:0c:29", "00:1c:14", "00:50:56"]
        if any(mac.lower().startswith(prefix) for prefix in vm_mac_prefixes):
            score += 2
            positive_indicators.append("vm_mac_address")
            if verbose:
                logging.info("Corroborating Indicator: VM-associated MAC address found.")
    except Exception:
        pass

    return score, list(set(positive_indicators))


def prompt_user_for_unsafe_execution(show_llm_description=True):
    """
    Prompt the user to decide whether to proceed without a sandbox.
    
    Args:
        show_llm_description: If True, display what the LLM is instructed to do.
    
    Returns:
        bool: True if user chooses to proceed, False to abort.
    """
    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        logging.warning("Non-interactive environment detected. Cannot prompt user.")
        logging.warning("Set UNSAFE_EXECUTION_OK=true to bypass sandbox check in non-interactive mode.")
        return False
    
    print("\n" + "=" * 74)
    print("⚠️  WARNING: NO SECURITY SANDBOX DETECTED ⚠️")
    print("=" * 74)
    
    if show_llm_description:
        print()
        print(LLM_EXECUTION_DESCRIPTION)
        print()
        print("-" * 74)
    
    print("""
WHY A SANDBOX IS RECOMMENDED:
  While the LLM is instructed to perform only safe operations, AI-generated
  code can sometimes behave unexpectedly due to:
  • Misinterpretation of instructions
  • Hallucinated or incorrect code patterns  
  • Edge cases in data that trigger unusual behavior

POTENTIAL RISKS WITHOUT A SANDBOX:
  • Accidental file modifications outside intended directories
  • High CPU/memory usage from inefficient generated code
  • Unexpected interactions with your Python environment

HOW TO RUN SAFELY:
  1. Docker (Recommended):  Run in a container using the provided Dockerfile
  2. Virtual Machine:       Use VMware, VirtualBox, or a cloud VM
  3. Google Colab:          Use Colab's free isolated environment
""")
    print("=" * 74)
    
    while True:
        try:
            response = input("\n❓ Proceed WITHOUT sandbox protection? (y=yes, n=abort) [N]: ").strip().lower()
            if response in ('y', 'yes'):
                print()
                logging.warning("⚠️  User acknowledged risks and chose to proceed without sandbox.")
                return True
            elif response in ('n', 'no', ''):
                print()
                logging.info("User chose to abort. No code will be executed.")
                return False
            else:
                print("   Please enter 'y' to proceed or 'n' to abort.")
        except (EOFError, KeyboardInterrupt):
            print("\n\nAborted by user.")
            return False


def get_execution_description():
    """
    Return a description of what the LLM execution system does.
    
    This can be used to inform users before they start using the system.
    
    Returns:
        str: Multi-line description of LLM execution behavior.
    """
    return LLM_EXECUTION_DESCRIPTION


def enforce_security_sandbox(required_score=4, allow_override=False, interactive=True,
                              show_llm_description=True):
    """
    Enforce security sandbox requirement before code execution.
    
    Args:
        required_score: Minimum sandbox score required (default: 4)
        allow_override: Allow UNSAFE_EXECUTION_OK env var to bypass (default: False)
        interactive: If True, prompt user when sandbox not detected (default: True)
        show_llm_description: If True, show what the LLM does when prompting (default: True)
    
    Returns:
        bool: True if execution should proceed, False if it should abort.
        
    Raises:
        RuntimeError: If sandbox check fails and interactive=False and no override.
    """
    # Check for environment variable override
    if allow_override and os.environ.get("UNSAFE_EXECUTION_OK", "false").lower() == "true":
        logging.warning("⚠️  WARNING: Safety check explicitly bypassed by environment variable.")
        logging.warning("         Executing on the host machine at your own risk.")
        return True

    logging.info("Running Security Sandbox Check...")
    score, indicators = check_security_sandbox_indicators(verbose=True)

    if score >= required_score:
        friendly_name = indicators[0] if indicators else "Unknown Sandbox"
        logging.info(f"✅ Security Check Passed (Score: {score}, Indicator: {friendly_name})")
        logging.info("   OS-level isolated environment detected. Proceeding safely.")
        return True
    
    # Sandbox not detected
    logging.warning(f"⚠️  Security Check: No sandbox detected (Score: {score}, Required: {required_score})")
    
    if interactive:
        # Prompt user for decision
        if prompt_user_for_unsafe_execution(show_llm_description=show_llm_description):
            return True
        else:
            raise RuntimeError("User chose to abort execution due to missing sandbox.")
    else:
        # Non-interactive mode: fail hard
        error_msg = f"""
{'='*74}
❌ SECURITY CHECK FAILED: NO SANDBOX DETECTED ❌
{'='*74}

{LLM_EXECUTION_DESCRIPTION}

{'='*74}
CURRENT STATUS:
  Sandbox Score: {score} (Required: {required_score})
  Mode: Non-interactive (cannot prompt for confirmation)
{'='*74}

TO PROCEED, CHOOSE ONE OPTION:

  1. Docker (Recommended):
     docker run -it your-scilink-image

  2. Virtual Machine:
     Run inside VMware, VirtualBox, or a cloud VM

  3. Google Colab:
     Use Colab's isolated notebook environment

  4. Environment Variable (accepts all risk):
     export UNSAFE_EXECUTION_OK=true

  5. Interactive Mode:
     Use interactive=True to be prompted for confirmation

{'='*74}
"""
        logging.error(error_msg)
        raise RuntimeError("Security sandbox requirement not met. Halting execution for safety.")


class ScriptExecutor:
    """
    Executes AI-generated Python scripts with optional sandbox enforcement.
    
    This executor runs Python code generated by an LLM for scientific analysis.
    The LLM is instructed to use common scientific libraries (NumPy, SciPy,
    scikit-learn, matplotlib, pandas) for data analysis tasks.
    
    Security:
        When enforce_sandbox=True, the executor verifies it's running in an
        isolated environment (Docker, VM, or Colab) before executing any code.
        If no sandbox is detected, the user is prompted to confirm they want
        to proceed (in interactive mode) or execution is blocked.
    
    Args:
        timeout: Maximum script execution time in seconds (default: 120)
        mp_api_key: Materials Project API key for materials science queries
        enforce_sandbox: Require sandbox verification before execution
        allow_unsafe_override: Allow UNSAFE_EXECUTION_OK env var to bypass
        interactive_sandbox: Prompt user if no sandbox detected (vs. hard fail)
        show_llm_description: Show what the LLM does when prompting user
    
    Example:
        # Standard usage with sandbox enforcement
        executor = ScriptExecutor(enforce_sandbox=True)
        result = executor.execute_script("import numpy as np; print(np.pi)")
        
        # Non-interactive (for CI/CD - will fail without sandbox)
        executor = ScriptExecutor(enforce_sandbox=True, interactive_sandbox=False)
        
        # See what the LLM is instructed to do
        print(executor.get_execution_description())
    """
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, mp_api_key: str = None,
                 enforce_sandbox: bool = True, allow_unsafe_override: bool = False,
                 interactive_sandbox: bool = True, show_llm_description: bool = True):
        self.timeout = timeout
        self.mp_api_key = mp_api_key or get_api_key('materials_project') or os.getenv("MP_API_KEY")
        self.enforce_sandbox = enforce_sandbox
        self.allow_unsafe_override = allow_unsafe_override
        self.interactive_sandbox = interactive_sandbox
        self.show_llm_description = show_llm_description
        self._user_accepted_risk = False
        self._sandbox_verified = False
        
        if self.enforce_sandbox:
            self._sandbox_verified, self._user_accepted_risk = self._check_sandbox_once()
        
        logging.info(f"ScriptExecutor initialized (timeout: {self.timeout}s, sandbox: {self._get_sandbox_status()})")

    def _get_sandbox_status(self) -> str:
        """Get a human-readable sandbox status."""
        if not self.enforce_sandbox:
            return "not enforced"
        elif self._sandbox_verified:
            return "verified"
        elif self._user_accepted_risk:
            return "user-accepted-risk"
        else:
            return "blocked"

    def _check_sandbox_once(self) -> tuple:
        """
        Perform sandbox check once at initialization.
        
        Returns:
            tuple: (sandbox_verified: bool, user_accepted_risk: bool)
        """
        score, _ = check_security_sandbox_indicators(verbose=False)
        
        if score >= 4:
            # Sandbox detected
            try:
                enforce_security_sandbox(
                    allow_override=self.allow_unsafe_override,
                    interactive=False  # Don't prompt, just verify
                )
                return True, False
            except RuntimeError:
                return False, False
        else:
            # No sandbox - need user confirmation or override
            try:
                enforce_security_sandbox(
                    allow_override=self.allow_unsafe_override,
                    interactive=self.interactive_sandbox,
                    show_llm_description=self.show_llm_description
                )
                # If we get here, user accepted or override was set
                return False, True
            except RuntimeError:
                return False, False

    def get_execution_description(self) -> str:
        """
        Get a description of what the LLM execution system does.
        
        Returns:
            str: Multi-line description of LLM behavior and capabilities.
        """
        return LLM_EXECUTION_DESCRIPTION

    def get_sandbox_status(self) -> dict:
        """
        Get detailed sandbox status information.
        
        Returns:
            dict: Status information including score, indicators, and decisions.
        """
        score, indicators = check_security_sandbox_indicators(verbose=False)
        return {
            "enforcement_enabled": self.enforce_sandbox,
            "sandbox_score": score,
            "required_score": 4,
            "sandbox_detected": score >= 4,
            "indicators": indicators,
            "user_accepted_risk": self._user_accepted_risk,
            "execution_allowed": not self.enforce_sandbox or self._sandbox_verified or self._user_accepted_risk,
            "status": self._get_sandbox_status()
        }

    def execute_script(self, script_content: str, working_dir: str = None) -> dict:
        """
        Execute a Python script.
        
        Args:
            script_content: Python code to execute
            working_dir: Optional working directory for execution
            
        Returns:
            dict: Result with 'status' ('success' or 'error') and output/message
        """
        if self.enforce_sandbox and not (self._sandbox_verified or self._user_accepted_risk):
            return {
                "status": "error", 
                "message": "Execution blocked: No sandbox detected and user declined to proceed.\n\n"
                          "To understand what this system does, call executor.get_execution_description()"
            }
        
        logging.info("Executing AI-generated Python script...")
        if self._sandbox_verified:
            logging.info("   Environment: Sandbox verified ✓")
        elif self._user_accepted_risk:
            logging.warning("   Environment: No sandbox (user accepted risk) ⚠️")
        
        original_cwd = os.getcwd()
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)

        temp_script_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=os.getcwd()) as tf:
                tf.write(script_content)
                temp_script_file = tf.name
            
            env = os.environ.copy()
            if self.mp_api_key:
                env['MP_API_KEY'] = self.mp_api_key

            result = subprocess.run(
                ['python', os.path.basename(temp_script_file)],
                capture_output=True, text=True, timeout=self.timeout, env=env, check=False
            )
            
            logging.debug(f"STDOUT:\n{result.stdout}")
            logging.debug(f"STDERR:\n{result.stderr}")

            if result.returncode == 0:
                return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
            else:
                error_msg = f"Script execution failed with return code {result.returncode}.\nSTDERR:\n{result.stderr}"
                return {"status": "error", "message": error_msg}

        except subprocess.TimeoutExpired:
            error_msg = f"Script execution timed out after {self.timeout} seconds."
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred during script execution: {e}"
            return {"status": "error", "message": error_msg}
        finally:
            os.chdir(original_cwd)
            if temp_script_file and os.path.exists(temp_script_file):
                os.remove(temp_script_file)
