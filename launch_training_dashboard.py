#!/usr/bin/env python3
"""
Arbor Training Dashboard Launcher

This script makes it easy to start both training and the dashboard
simultaneously in separate processes.

Usage:
    python launch_training_dashboard.py
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

class TrainingDashboardLauncher:
    """Launcher for Arbor training with dashboard."""
    
    def __init__(self):
        self.training_process = None
        self.dashboard_process = None
        self.running = True
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n🛑 Shutting down processes...")
        self.running = False
        
        if self.training_process:
            self.training_process.terminate()
        
        if self.dashboard_process:
            self.dashboard_process.terminate()
        
        print("✅ Processes terminated")
        sys.exit(0)
    
    def launch_training(self):
        """Launch the training process."""
        print("🚀 Starting training process...")
        
        # Check if training script exists
        training_script = "examples/training_with_dashboard.py"
        if not Path(training_script).exists():
            print(f"❌ Training script not found: {training_script}")
            return None
        
        # Start training process
        try:
            self.training_process = subprocess.Popen(
                [sys.executable, training_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"✅ Training started (PID: {self.training_process.pid})")
            return self.training_process
        except Exception as e:
            print(f"❌ Failed to start training: {e}")
            return None
    
    def launch_dashboard(self):
        """Launch the Streamlit dashboard."""
        print("📊 Starting dashboard...")
        
        # Check if dashboard exists
        dashboard_script = "arbor/tracking/dashboard.py"
        if not Path(dashboard_script).exists():
            print(f"❌ Dashboard script not found: {dashboard_script}")
            return None
        
        # Start dashboard process
        try:
            self.dashboard_process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", 
                    dashboard_script, 
                    "--server.port=8501",
                    "--server.headless=true",
                    "--browser.gatherUsageStats=false"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"✅ Dashboard started (PID: {self.dashboard_process.pid})")
            print("🌐 Dashboard available at: http://localhost:8501")
            return self.dashboard_process
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return None
    
    def monitor_processes(self):
        """Monitor running processes and display output."""
        print("\n" + "="*60)
        print("🔍 Monitoring processes... (Ctrl+C to stop)")
        print("="*60)
        
        try:
            while self.running:
                # Check training process
                if self.training_process:
                    # Read training output
                    if self.training_process.poll() is None:  # Still running
                        try:
                            output = self.training_process.stdout.readline()
                            if output:
                                print(f"[TRAINING] {output.strip()}")
                        except:
                            pass
                    else:
                        print("⚠️ Training process finished")
                        self.training_process = None
                
                # Check dashboard process
                if self.dashboard_process:
                    if self.dashboard_process.poll() is not None:  # Finished
                        print("⚠️ Dashboard process finished")
                        self.dashboard_process = None
                
                # Small delay
                time.sleep(0.1)
                
                # Exit if both processes are done
                if not self.training_process and not self.dashboard_process:
                    break
        
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
    
    def run(self):
        """Main execution function."""
        print("🌳 Arbor Training Dashboard Launcher")
        print("=" * 40)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Launch dashboard first (takes a bit to start up)
        dashboard_success = self.launch_dashboard() is not None
        
        if dashboard_success:
            print("⏳ Waiting for dashboard to initialize...")
            time.sleep(3)
        
        # Launch training
        training_success = self.launch_training() is not None
        
        if not training_success and not dashboard_success:
            print("❌ Failed to start both processes")
            return
        
        if training_success:
            time.sleep(1)  # Let training start
        
        # Monitor both processes
        if training_success or dashboard_success:
            self.monitor_processes()

def main():
    """Main function."""
    launcher = TrainingDashboardLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
