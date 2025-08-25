"""
Streamlit GUI for Large Scale Multi-Agent Simulator (LS_MAS)
============================================================

This GUI provides an easy-to-use interface for running different simulation examples
by executing the existing launcher scripts with configurable parameters.

Authors: 
- Generated with GitHub Copilot assistance

Usage:
    streamlit run streamlit_gui.py
"""

import streamlit as st
import sys
import os
import pathlib
import yaml
import datetime
import subprocess
import threading
import time
from io import StringIO
import shutil



# Get the current directory
current_dir = pathlib.Path(__file__).parent



# Page configuration
st.set_page_config(
    page_title="LS_MAS Simulator GUI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config file: {e}")
        return None





def check_launcher_files():
    """Check which launcher files are available by scanning the Examples directory"""
    examples_dir = current_dir / "Examples"
    
    if not examples_dir.exists():
        return {}
    
    available_launchers = {}
    
    # Scan for all *_launcher.py files
    for launcher_file in examples_dir.glob("*_launcher.py"):
        # Extract the base name (remove _launcher.py suffix)
        base_name = launcher_file.stem.replace("_launcher", "")
        
        # Create a display name by capitalizing and replacing underscores with spaces
        display_name = " ".join(word.capitalize() for word in base_name.split("_"))
        
        # Add "Simulation" suffix if not already present
        if not display_name.lower().endswith("simulation") and not display_name.lower().endswith("gym"):
            display_name += " Simulation"
        
        available_launchers[display_name] = True
    
    return available_launchers



def get_python_executable():
    """Get the correct Python executable path for the current environment"""
    # Try to detect if we're in a virtual environment
    venv_python = current_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    
    # Fallback to system Python
    return "python"



def run_launcher_script(launcher_name, config_path=None, show_output_callback=None):
    """Run a launcher script as a subprocess with real-time output"""
    try:
        launcher_path = current_dir / "Examples" / launcher_name
        
        if not launcher_path.exists():
            return False, f"Launcher file not found: {launcher_path}"
        
        python_exe = get_python_executable()
        
        # Run the launcher script with real-time output
        process = subprocess.Popen(
            [python_exe, str(launcher_path)],
            cwd=str(current_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        output_lines = []
        
        # Read output line by line in real-time
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line.strip())
            if show_output_callback:
                show_output_callback(line.strip())
        
        process.stdout.close()
        return_code = process.wait()
        
        full_output = '\n'.join(output_lines)
        
        if return_code == 0:
            return True, f"Simulation completed successfully!\n\nOutput:\n{full_output}"
        else:
            return False, f"Simulation failed with return code {return_code}\n\nOutput:\n{full_output}"
            
    except Exception as e:
        return False, f"Error running simulation: {e}"
    

def dict_form_generation(params_dict: dict, Key_Names: dict, id: str=""):
    for key,value in params_dict.items():
        if isinstance(value,(int,float)):
            new_num = st.number_input(f"{Key_Names.get(key, key)}=",value=value,key=f"{id}_{key}")
            params_dict[key] = new_num
        elif isinstance(value,str):
            new_str = st.text_input(f"{Key_Names.get(key, key)}=",value=value,key=f"{id}_{key}")
            params_dict[key] = new_str
        elif isinstance(value,dict):
            st.subheader(f"{Key_Names.get(key, key)} Parameters")
            dict_form_generation(value, Key_Names, id=f"{id}_{key}")
        elif isinstance(value,list):
            new_list_str = st.text_area(f"{Key_Names.get(key, key)}", value=str(value), key=f"{id}_{key}")
            params_dict[key] = eval(f"{new_list_str}") if new_list_str else []
        else:
            st.warning(f"Unsupported parameter type for {key}: {type(value)}")

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🤖 IntelliSwarm Dashboard (iSwarm)")
    st.markdown("# Interactive GUI for Multi-Agent Simulations")
    
    # Check if launcher files exist
    examples_dir = current_dir / "Examples"
    if not examples_dir.exists():
        st.error("Examples directory not found. Please check your installation.")
        return
    
    # ============== Sidebar for simulation selection =================
    st.sidebar.header("Simulation Selection")
    
    # Check available launchers
    available_launchers = check_launcher_files()
    available_simulations = [name for name, available in available_launchers.items() if available]
    
    if not available_simulations:
        st.error("No launcher files found in the Examples directory!")
        return
    
    simulation_type = st.sidebar.selectbox(
        "Select Simulation Type",
        available_simulations
    )

    # =========== Load configuration =============

    config_filename = f"{simulation_type.replace(' Simulation', '').replace(' ', '_').lower()}_config.yaml"

    config_path = current_dir / "Configuration" / config_filename
    
    config = load_config(config_path)
    
    if config is None:
        st.error(f"Could not load configuration file: {config_path}")
        return

    
    # ========== Main content area ==============

    col1, col2 = st.columns([2, 1])

    Key_Names = {"N": "Number of Agents",
                 "state_dim": "State Dimension",
                 "mode": "Generation Mode",
                 "id": "Population ID",
                 "dt": "Time Step",
                 "T": "Total Time",
                 "background_color": "Background Color",
                 "render_mode": "Render Mode",
                 "render_dt": "Render Time Step",
                 "activate": "Activation Flag (1 for True, 0 for False)",
                 "log_freq": "Logging Frequency",
                 "save_freq": "Saving Frequency",
                 "save_data_freq": "Data Saving Frequency",
                 "comment_enable": "Comment Enable Flag (1 for True, 0 for False)",
                 "log_path": "Log File Path",
                 "log_name": "Log File Name"
    }

    with col1:
        st.header(f"{simulation_type} Configuration")

        # Display and allow editing of configuration parameters
        dict_form_generation(config, Key_Names,"Config")

        # Show current configuration as expandable YAML
        with st.expander("View Full Configuration (YAML)", expanded=False):
            config_yaml = yaml.dump(config, default_flow_style=False)
            st.code(config_yaml, language='yaml')
    
    
    with col2:

        st.header("Simulation Control")
        
        # Run simulation button
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            
            
            # Save modified config
            try:
                with open(config_path, 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
            except Exception as e:
                st.error(f"Error saving temporary config: {e}")

            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Starting simulation...")
            eta=""
            
            with st.spinner(f"Running {simulation_type}...") as spinner:
                try:
                    # Create containers for real-time output
                    ETA_label = st.text("Remaining Execution Time: N/A")
                    output_container = st.empty()
                    output_lines = []
                    

                    def update_output(line):
                        line_l = line.split(":")
                        if line_l[0] == "Processing":
                            pct_split = line.split("%")
                            completion_str = pct_split[0]
                            completion = completion_str.split(" ")[-1]  # Get the number before '%'
                            progress = float(completion) / 100
                            ETA_label.text(f"Remaining Execution Time: {line.split('ETA:')[-1].strip()}")
                            progress_bar.progress(progress)
                        if line.strip():  # Only show non-empty lines
                            output_lines.append(line)
                            # Show last 3 lines of output
                            recent_output = '\n'.join(output_lines[-3:])
                            output_container.code(recent_output, language='text')
                    
                    code, message = run_launcher_script(
                        simulation_type.replace(' Simulation', '').replace(' ', '_').lower() + "_launcher.py", 
                        str(config_path),
                        show_output_callback=update_output
                    )
                    
                    progress_bar.progress(100)
                    
                    if code:
                        status_text.text("Simulation completed successfully!")
                        st.success("✅ Simulation completed successfully!")
                    else:
                        status_text.text("Simulation failed!")
                        st.error(f"❌ Simulation failed: {message}")

                    # Show information about output files
                    logs_dir = current_dir / "logs"
                    if logs_dir.exists():
                        st.info(f"📁 Output files saved to: {logs_dir}")

                        # List recent files
                        try:
                            recent_files = sorted(logs_dir.glob("*"), key=os.path.getmtime, reverse=True)[:5]
                            if recent_files:
                                st.write("**Recent output files:**")
                                for file in recent_files:
                                    st.write(f"- {file.name}")
                        except Exception:
                            pass
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text("Error occurred!")
                    st.error(f"Unexpected error: {e}")
        
        # Information section
        st.subheader("📊 Simulation Info")
        st.info(f"""
        **Selected Type:** {simulation_type}
        
        **Config File:** {config_filename}
        
        **Output Location:** logs/
        
        **Available Formats:** CSV, NPZ, MAT, TXT
        """)
        
        # Quick help
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to use:**
            1. Select a simulation type from the sidebar
            2. Adjust parameters in the main panel
            3. Click "Run Simulation" to start
            4. Check the logs folder for output files
            """)

if __name__ == "__main__":
    main()
