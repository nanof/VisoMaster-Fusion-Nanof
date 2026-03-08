import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path

# --- Configuration ---

# Directory containing the ONNX models, relative to this script
MODEL_DIR = "model_assets"

# Subdirectory for backing up original models
BACKUP_DIR_NAME = "unopt-backup"

# Log file name
LOG_FILE = "optimize_models.log"

# List of models to skip during optimization
MODEL_EXCEPTIONS = {
    "w600k_r50.onnx",
    "yunet_n_640_640.onnx",
    "2dfan4.onnx",
    "codeformer_fp16.onnx",
    "cscs_arcface_model.onnx",
    "cscs_id_adapter.onnx",
    "faceparser_fp16.onnx",
    "det_10g.onnx",
    "inswapper_128.fp16.onnx",
    "ghost_arcface_backbone.onnx",
    "ghost_unet_2_block.onnx",
    "ghost_unet_3_block.onnx",
    "realesr-general-x4v3.onnx",
}

# --- Script Logic ---

def setup_logging():
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def find_onnxruntime_script(script_name):
    """
    Finds the full path to a script within the onnxruntime tools directory.
    This is more robust than hardcoding the path.
    """
    try:
        import onnxruntime
        ort_path = Path(onnxruntime.__file__).parent
        script_path = ort_path / "tools" / script_name
        if script_path.is_file():
            return str(script_path)
        else:
            logging.error(f"Could not find '{script_name}' at '{script_path}'")
            return None
    except ImportError:
        logging.error("onnxruntime is not installed. Please install it to run this script.")
        return None

def run_command(command):
    """Executes a command and logs its output."""
    process_str = ' '.join(f'"{c}"' if ' ' in c else c for c in command)
    logging.info(f"Executing: {process_str}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            logging.info(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logging.warning(f"STDERR:\n{result.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT:\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"STDERR:\n{e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logging.error(f"Command not found: '{command[0]}'. Make sure the required tools are installed and in your PATH.")
        return False


def main():
    """Main function to orchestrate the model optimization process."""
    setup_logging()
    
    # --- Pre-flight Checks ---
    if not Path(MODEL_DIR).is_dir():
        logging.error(f"Model directory not found: '{MODEL_DIR}'. Please run this script from your application's main directory.")
        return

    shape_infer_script = find_onnxruntime_script("symbolic_shape_infer.py")
    if not shape_infer_script:
        logging.error("Aborting due to missing symbolic_shape_infer.py script.")
        return

    # --- Setup Backup Directory ---
    backup_path = Path(MODEL_DIR) / BACKUP_DIR_NAME
    backup_path.mkdir(exist_ok=True)
    logging.info(f"Backup directory is '{backup_path}'")
    
    # --- Process Models ---
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx") and os.path.isfile(os.path.join(MODEL_DIR, f))]

    if not model_files:
        logging.warning(f"No .onnx models found in '{MODEL_DIR}'.")
        return

    logging.info(f"Found {len(model_files)} ONNX models to process.")

    for model_file in model_files:
        logging.info("-" * 60)
        
        if model_file in MODEL_EXCEPTIONS:
            logging.info(f"Skipping model due to exception rule: {model_file}")
            continue

        logging.info(f"Processing model: {model_file}")

        base_name = Path(model_file).stem
        original_path = Path(MODEL_DIR) / model_file
        
        # Define intermediate and final file paths
        opt_path = Path(MODEL_DIR) / f"{base_name}_opt.onnx"
        opt_sym_path = Path(MODEL_DIR) / f"{base_name}_opt-sym.onnx"
        final_backup_path = backup_path / model_file

        # List of temporary files to clean up
        temp_files = [opt_path, opt_sym_path]
        
        try:
            # 1. Run onnxsim
            logging.info(f"Step 1: Running ONNX Simplifier on {model_file}")
            onnxsim_cmd = [
                sys.executable, "-m", "onnxsim",
                str(original_path),
                str(opt_path)
            ]
            if not run_command(onnxsim_cmd):
                raise RuntimeError("ONNX Simplifier failed.")

            # 2. Run symbolic shape inference
            logging.info(f"Step 2: Running Symbolic Shape Inference on {opt_path.name}")
            shape_infer_cmd = [
                sys.executable, shape_infer_script,
                "--input", str(opt_path),
                "--output", str(opt_sym_path),
                "--auto_merge" # Often useful for performance
            ]
            if not run_command(shape_infer_cmd):
                raise RuntimeError("Symbolic shape inference failed.")

            # 3. Backup and replace
            logging.info(f"Step 3: Backing up original model and replacing with optimized version.")
            shutil.move(original_path, final_backup_path)
            logging.info(f"  -> Moved '{original_path.name}' to '{final_backup_path}'")
            shutil.move(opt_sym_path, original_path)
            logging.info(f"  -> Moved '{opt_sym_path.name}' to '{original_path.name}'")
            
            logging.info(f"Successfully optimized and replaced {model_file}\n")

        except Exception as e:
            logging.error(f"Failed to process {model_file}. Error: {e}")
            logging.error(f"The original model '{original_path.name}' will be left in place.")
            # If an error occurs, the original file is already in its place, so no action is needed.

        finally:
            # Clean up intermediate files regardless of success or failure
            for temp_file in temp_files:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        logging.info(f"Cleaned up temporary file: {temp_file.name}")
                    except OSError as e:
                        logging.error(f"Error removing temporary file {temp_file.name}: {e}")

    logging.info("-" * 60)
    logging.info("Optimization process finished.")


if __name__ == "__main__":
    main()
