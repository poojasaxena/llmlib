#!/usr/bin/env python3
"""
Sleep prevention CLI tools - system-level sleep management
These tools manage system sleep/suspend prevention without requiring torch
"""

import os
import sys
import subprocess
import signal
from pathlib import Path

# Global PID file for tracking sleep prevention processes
PID_FILE = Path.home() / ".llmlib_sleep_prevention.pid"

def _get_sleep_prevention_command():
    """Get the appropriate sleep prevention command for the current system."""
    if sys.platform == "darwin":  # macOS
        return ["caffeinate", "-d"]
    elif sys.platform.startswith("linux"):  # Linux
        # Check if systemd-inhibit is available
        try:
            subprocess.run(["systemd-inhibit", "--version"], 
                         capture_output=True, check=True)
            return ["systemd-inhibit", "--what=sleep:idle:handle-lid-switch", 
                   "--why=LLM Training", "--mode=block", "sleep", "infinity"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to xset (if X11 is available)
            try:
                subprocess.run(["xset", "q"], capture_output=True, check=True)
                return ["xset", "s", "off", "-dpms"]
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None
    else:
        return None

def disable_sleep():
    """Disable system sleep/suspend."""
    print("🚫 Disabling system sleep/suspend...")
    
    # Check if already disabled
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            # Check if process is still running
            os.kill(old_pid, 0)
            print(f"Sleep prevention is already active (PID: {old_pid})")
            return
        except (OSError, ValueError, ProcessLookupError):
            # PID file exists but process is dead, remove stale file
            PID_FILE.unlink(missing_ok=True)
    
    cmd = _get_sleep_prevention_command()
    if not cmd:
        print("❌ Sleep prevention not supported on this system")
        return
    
    try:
        if sys.platform == "darwin" or "systemd-inhibit" in cmd[0]:
            # Start background process
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
            
            # Save PID
            with open(PID_FILE, 'w') as f:
                f.write(str(process.pid))
            
            print(f"✅ Sleep prevention enabled (PID: {process.pid})")
            print("   Use 'llmlib-enable-sleep' to re-enable sleep")
            
        else:  # xset fallback
            subprocess.run(cmd, check=True)
            print("✅ Sleep prevention enabled using xset")
            print("   Use 'llmlib-enable-sleep' to re-enable sleep")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to disable sleep: {e}")
        sys.exit(1)

def enable_sleep():
    """Re-enable system sleep/suspend."""
    print("💤 Re-enabling system sleep/suspend...")
    
    # Kill any running sleep prevention process
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            
            # Kill the process
            os.kill(pid, signal.SIGTERM)
            PID_FILE.unlink()
            print(f"✅ Stopped sleep prevention process (PID: {pid})")
            
        except (OSError, ValueError, ProcessLookupError):
            print("⚠️  Sleep prevention process not found (may have already stopped)")
            PID_FILE.unlink(missing_ok=True)
    
    # Additional cleanup for different systems
    if sys.platform.startswith("linux"):
        try:
            # Re-enable screen saver and power management
            subprocess.run(["xset", "s", "on", "+dpms"], 
                         capture_output=True, check=False)
        except FileNotFoundError:
            pass  # xset not available
    
    print("✅ System sleep/suspend re-enabled")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--enable":
        enable_sleep()
    else:
        disable_sleep()