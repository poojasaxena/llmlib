#!/usr/bin/env python3
"""
TMux Session Manager for LLM Training

Handles long-running training jobs in persistent tmux sessions with
proper session management, monitoring, and recovery capabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from llmlib.utils.logger import get_logger

logger = get_logger(__name__)


class TmuxSessionManager:
    """Manages tmux sessions for long-running training jobs."""
    
    def __init__(self):
        self.session_prefix = "llmlib"
        
    def _run_tmux_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a tmux command and return the result."""
        try:
            result = subprocess.run(
                ['tmux'] + cmd,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            if "no server running" in str(e.stderr):
                raise RuntimeError("tmux server not running. Please start tmux first.")
            raise RuntimeError(f"tmux command failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("tmux not found. Please install tmux first.")
    
    def list_sessions(self) -> List[Dict[str, str]]:
        """List all tmux sessions."""
        try:
            result = self._run_tmux_command(['list-sessions', '-F', '#{session_name}:#{session_created}:#{session_attached}'])
            sessions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        sessions.append({
                            'name': parts[0],
                            'created': parts[1],
                            'attached': parts[2]
                        })
            return sessions
        except RuntimeError:
            return []
    
    def session_exists(self, session_name: str) -> bool:
        """Check if a tmux session exists."""
        try:
            self._run_tmux_command(['has-session', '-t', session_name])
            return True
        except RuntimeError:
            return False
    
    def create_session(self, session_name: str, command: str, working_dir: Optional[Path] = None) -> bool:
        """Create a new tmux session with the given command."""
        try:
            if self.session_exists(session_name):
                logger.warning(f"Session '{session_name}' already exists")
                return False
            
            cmd = ['new-session', '-d', '-s', session_name]
            if working_dir:
                cmd.extend(['-c', str(working_dir)])
            cmd.append(command)
            
            self._run_tmux_command(cmd)
            logger.info(f"✅ Created tmux session: {session_name}")
            return True
            
        except RuntimeError as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def attach_session(self, session_name: str) -> bool:
        """Attach to an existing tmux session."""
        try:
            if not self.session_exists(session_name):
                logger.error(f"Session '{session_name}' does not exist")
                return False
            
            # Use os.execvp to replace current process with tmux attach
            os.execvp('tmux', ['tmux', 'attach-session', '-t', session_name])
            
        except Exception as e:
            logger.error(f"Failed to attach to session: {e}")
            return False
    
    def kill_session(self, session_name: str) -> bool:
        """Kill a tmux session."""
        try:
            if not self.session_exists(session_name):
                logger.warning(f"Session '{session_name}' does not exist")
                return False
                
            self._run_tmux_command(['kill-session', '-t', session_name])
            logger.info(f"✅ Killed session: {session_name}")
            return True
            
        except RuntimeError as e:
            logger.error(f"Failed to kill session: {e}")
            return False
    
    def send_keys(self, session_name: str, keys: str) -> bool:
        """Send keys to a tmux session."""
        try:
            if not self.session_exists(session_name):
                logger.error(f"Session '{session_name}' does not exist")
                return False
                
            self._run_tmux_command(['send-keys', '-t', session_name, keys, 'Enter'])
            return True
            
        except RuntimeError as e:
            logger.error(f"Failed to send keys: {e}")
            return False
    
    def capture_pane(self, session_name: str, lines: int = 50) -> Optional[str]:
        """Capture output from a tmux session."""
        try:
            if not self.session_exists(session_name):
                logger.error(f"Session '{session_name}' does not exist")
                return None
                
            result = self._run_tmux_command([
                'capture-pane', '-t', session_name, '-p', '-S', f'-{lines}'
            ])
            return result.stdout
            
        except RuntimeError as e:
            logger.error(f"Failed to capture pane: {e}")
            return None
    
    def get_training_sessions(self) -> List[Dict[str, str]]:
        """Get all llmlib training sessions with enhanced status information."""
        all_sessions = self.list_sessions()
        training_sessions = []
        
        for session in all_sessions:
            # Check if session name starts with our prefix OR contains llmlib training commands
            is_llmlib_session = (
                session['name'].startswith(f"{self.session_prefix}-") or
                self._is_llmlib_training_session(session['name'])
            )
            
            if is_llmlib_session:
                # Get enhanced status information
                output = self.capture_pane(session['name'], 10)
                status_info = self._analyze_training_status(session['name'])
                health = self._get_session_health(session['name'])
                
                training_sessions.append({
                    **session,
                    'recent_output': output.strip() if output else "",
                    'status_info': status_info,
                    'health': health
                })
        
        return training_sessions
    
    def _is_llmlib_training_session(self, session_name: str) -> bool:
        """Check if a session is running llmlib training commands."""
        try:
            output = self.capture_pane(session_name, 20)
            if output:
                # Look for llmlib training indicators
                llmlib_indicators = [
                    'llmlib-train-pipeline',
                    'llmlib.cli.train_pipeline_cli',
                    'Starting robust model training',
                    'Training attempt'
                ]
                return any(indicator in output for indicator in llmlib_indicators)
        except:
            pass
        return False
    
    def _analyze_training_status(self, session_name: str) -> Dict[str, str]:
        """Analyze the current status of a training session."""
        try:
            output = self.capture_pane(session_name, 50)
            if not output:
                return {"status": "unknown", "details": "No output available"}
            
            lines = output.strip().split('\n')
            recent_lines = [line for line in lines[-10:] if line.strip()]
            
            # Check for completion indicators
            if any('✅ Training pipeline completed successfully' in line for line in recent_lines):
                return {"status": "completed", "details": "Training completed successfully"}
            
            # Check for error indicators
            error_indicators = [
                'ERROR', 'FAILED', 'Exception', 'Traceback', 'Error:',
                'Failed to', 'Cannot', 'Unable to', 'CUDA out of memory'
            ]
            for line in recent_lines:
                if any(indicator in line for indicator in error_indicators):
                    return {"status": "failed", "details": f"Error detected: {line.strip()}"}
            
            # Check for hanging/stuck indicators
            last_line = recent_lines[-1] if recent_lines else ""
            
            # If the last output is very old (more than 5 minutes of same content)
            timestamp_patterns = ['2025-', '2024-', '2023-']  # Look for timestamps
            if any(pattern in last_line for pattern in timestamp_patterns):
                # Try to extract timestamp and compare
                import re
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', last_line)
                if timestamp_match:
                    try:
                        from datetime import datetime
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        time_diff = (datetime.now() - log_time).total_seconds() / 60
                        if time_diff > 5:  # More than 5 minutes old
                            return {"status": "stuck", "details": f"No activity for {time_diff:.1f} minutes"}
                    except:
                        pass
            
            # Check for specific training states
            if 'Training attempt' in last_line and 'Starting' not in output[-200:]:
                return {"status": "stuck", "details": "Training attempt may be hanging"}
            
            if 'Starting training pipeline' in output and '✅' not in output[-500:]:
                return {"status": "running", "details": "Training pipeline in progress"}
            
            if 'modern-gpt-train' in output:
                return {"status": "training", "details": "Model training in progress"}
            
            return {"status": "active", "details": "Session active"}
            
        except Exception as e:
            return {"status": "error", "details": f"Status check failed: {str(e)}"}
    
    def _get_session_health(self, session_name: str) -> str:
        """Get a health indicator emoji for the session."""
        status_info = self._analyze_training_status(session_name)
        status = status_info["status"]
        
        health_map = {
            "completed": "✅",
            "running": "🟢", 
            "training": "🟡",
            "active": "🔵",
            "stuck": "🟠",
            "failed": "❌",
            "error": "🔴",
            "unknown": "❓"
        }
        return health_map.get(status, "❓")
    
    def generate_session_name(self, config_path: Path) -> str:
        """Generate a unique session name for a training job."""
        timestamp = datetime.now().strftime("%m%d_%H%M")
        config_name = config_path.stem
        return f"{self.session_prefix}-{config_name}-{timestamp}"


def start_training_session(config_path: Path, session_name: Optional[str] = None, 
                         dry_run: bool = False, max_retries: int = 3, 
                         timeout: int = 8, auto_confirm: bool = False) -> bool:
    """Start a training session in tmux."""
    tmux = TmuxSessionManager()
    
    # Generate session name if not provided
    if not session_name:
        session_name = tmux.generate_session_name(config_path)
    
    logger.info(f"🚀 Starting training session: {session_name}")
    
    # Build the training command with absolute path and proper quoting
    config_path_abs = config_path.resolve()
    cmd_parts = ['llmlib-train-pipeline', '--config', f'"{config_path_abs}"']
    if dry_run:
        cmd_parts.append('--dry-run')
    if auto_confirm:
        cmd_parts.append('--auto-confirm')
    if max_retries != 3:
        cmd_parts.extend(['--max-retries', str(max_retries)])
    if timeout != 8:
        cmd_parts.extend(['--timeout', str(timeout)])
    
    # Always add --skip-sudo for tmux sessions to avoid password prompts
    cmd_parts.append('--skip-sudo')
    
    command = ' '.join(cmd_parts)
    
    # Debug: Log the exact command being executed
    logger.info(f"🔍 Command to execute: {command}")
    logger.info(f"📁 Working directory: {config_path.parent}")
    
    # Create the session
    success = tmux.create_session(session_name, command, config_path.parent)
    
    if success:
        logger.info(f"✅ Training started in session: {session_name}")
        logger.info(f"📺 To attach: tmux attach-session -t {session_name}")
        logger.info(f"📺 Or use: llmlib tmux attach {session_name}")
        logger.info(f"🔍 To monitor: llmlib tmux status")
        return True
    else:
        logger.error("❌ Failed to start training session")
        return False


def list_training_sessions():
    """List all active training sessions with enhanced status information."""
    tmux = TmuxSessionManager()
    sessions = tmux.get_training_sessions()
    
    if not sessions:
        print("🔍 No active training sessions found")
        print("💡 Use 'llmlib tmux start --config <config>' to start a new session")
        return
    
    print(f"🚀 Active Training Sessions ({len(sessions)}):")
    print("=" * 80)
    
    for session in sessions:
        created_time = datetime.fromtimestamp(int(session['created']))
        attached = "📺 Attached" if session['attached'] == '1' else "🔌 Detached"
        health = session.get('health', '❓')
        status_info = session.get('status_info', {})
        
        print(f"{health} Session: {session['name']}")
        print(f"  Status: {attached}")
        print(f"  Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Handle status_info properly (it's a dict)
        if isinstance(status_info, dict):
            print(f"  Health: {status_info.get('status', 'unknown').title()}")
            
            if status_info.get('details'):
                print(f"  Details: {status_info['details']}")
        
        if session['recent_output']:
            # Show last meaningful line instead of truncated text
            lines = [line.strip() for line in session['recent_output'].split('\n') if line.strip()]
            if lines:
                last_line = lines[-1]
                if len(last_line) > 80:
                    last_line = last_line[:77] + "..."
                print(f"  Last: {last_line}")
        
        # Add action suggestions based on status
        status = status_info.get('status', '') if isinstance(status_info, dict) else ''
        if status == 'stuck':
            print(f"  💡 Suggestion: Check session with 'llmlib tmux attach {session['name']}'")
        elif status == 'failed':
            print(f"  💡 Suggestion: Check logs with 'llmlib tmux attach {session['name']}'")
        elif status == 'completed':
            print(f"  💡 Suggestion: Session can be cleaned up with 'llmlib tmux kill {session['name']}'")
        
        print()


def attach_to_session(session_name: Optional[str] = None):
    """Attach to a training session."""
    tmux = TmuxSessionManager()
    
    # If no session name provided, show available sessions
    if not session_name:
        sessions = tmux.get_training_sessions()
        if not sessions:
            logger.error("No training sessions found")
            return False
        
        if len(sessions) == 1:
            session_name = sessions[0]['name']
            logger.info(f"Attaching to only available session: {session_name}")
        else:
            print("Available sessions:")
            for i, session in enumerate(sessions):
                print(f"  {i+1}. {session['name']}")
            
            try:
                choice = int(input("Choose session (number): ")) - 1
                if 0 <= choice < len(sessions):
                    session_name = sessions[choice]['name']
                else:
                    logger.error("Invalid choice")
                    return False
            except (ValueError, KeyboardInterrupt):
                logger.error("Invalid input")
                return False
    
    return tmux.attach_session(session_name)


def monitor_sessions():
    """Monitor all training sessions with live updates and enhanced status."""
    tmux = TmuxSessionManager()
    
    try:
        while True:
            os.system('clear')
            print("🖥️  LLMLIB Training Monitor (tmux sessions)")
            print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            sessions = tmux.get_training_sessions()
            if not sessions:
                print("🔍 No active training sessions")
                print("💡 Use 'llmlib tmux start --config <config>' to start a new session")
            else:
                for session in sessions:
                    created_time = datetime.fromtimestamp(int(session['created']))
                    attached = "📺 Attached" if session['attached'] == '1' else "🔌 Detached"
                    health = session.get('health', '❓')
                    status_info = session.get('status_info', {})
                    
                    print(f"\n{health} {session['name']}")
                    print(f"   Connection: {attached}")
                    print(f"   Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if isinstance(status_info, dict):
                        print(f"   Health: {status_info.get('status', 'unknown').title()}")
                        if status_info.get('details'):
                            print(f"   Details: {status_info['details'][:60]}...")
                    
                    # Get recent output (last 2 lines)
                    output = tmux.capture_pane(session['name'], 5)
                    if output:
                        lines = [line.strip() for line in output.strip().split('\n') if line.strip()][-2:]
                        for line in lines:
                            if line:
                                # Truncate long lines for display
                                display_line = line[:70] + "..." if len(line) > 70 else line
                                print(f"   📝 {display_line}")
                    
                    # Show alerts for problematic sessions
                    status = status_info.get('status', '') if isinstance(status_info, dict) else ''
                    if status == 'stuck':
                        print(f"   � ALERT: Session may be hanging!")
                    elif status == 'failed':
                        print(f"   ❌ ALERT: Training failed!")
            
            print("\n" + "=" * 80)
            print("Press Ctrl+C to stop monitoring | 'a' + session_name to attach")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")


def kill_training_session(session_name: Optional[str] = None, kill_all: bool = False):
    """Kill training sessions."""
    tmux = TmuxSessionManager()
    
    if kill_all:
        sessions = tmux.get_training_sessions()
        if not sessions:
            print("🔍 No training sessions to kill")
            return
        
        print(f"🛑 Killing {len(sessions)} training sessions...")
        for session in sessions:
            tmux.kill_session(session['name'])
        return
    
    if not session_name:
        sessions = tmux.get_training_sessions()
        if not sessions:
            logger.error("No training sessions found")
            return
        
        print("Available sessions to kill:")
        for i, session in enumerate(sessions):
            print(f"  {i+1}. {session['name']}")
        
        try:
            choice = int(input("Choose session to kill (number): ")) - 1
            if 0 <= choice < len(sessions):
                session_name = sessions[choice]['name']
            else:
                logger.error("Invalid choice")
                return
        except (ValueError, KeyboardInterrupt):
            logger.error("Invalid input")
            return
    
    tmux.kill_session(session_name)


def diagnose_session(session_name: Optional[str] = None):
    """Diagnose issues with a training session."""
    tmux = TmuxSessionManager()
    
    # If no session name provided, show available sessions
    if not session_name:
        sessions = tmux.get_training_sessions()
        if not sessions:
            logger.error("No training sessions found")
            return
        
        print("Available sessions to diagnose:")
        for i, session in enumerate(sessions):
            health = session.get('health', '❓')
            status_info = session.get('status_info', {})
            status = status_info.get('status', 'unknown') if isinstance(status_info, dict) else 'unknown'
            print(f"  {i+1}. {health} {session['name']} ({status})")
        
        try:
            choice = int(input("Choose session to diagnose (number): ")) - 1
            if 0 <= choice < len(sessions):
                session_name = sessions[choice]['name']
            else:
                logger.error("Invalid choice")
                return
        except (ValueError, KeyboardInterrupt):
            logger.error("Invalid input")
            return
    
    # Perform detailed diagnosis
    print(f"🔍 Diagnosing session: {session_name}")
    print("=" * 60)
    
    if not tmux.session_exists(session_name):
        print("❌ Session does not exist!")
        return
    
    # Get detailed status
    status_info = tmux._analyze_training_status(session_name)
    print(f"Status: {status_info['status'].upper()}")
    print(f"Details: {status_info['details']}")
    print()
    
    # Get full recent output
    output = tmux.capture_pane(session_name, 30)
    if output:
        print("📝 Recent Output (last 30 lines):")
        print("-" * 40)
        lines = output.strip().split('\n')
        for i, line in enumerate(lines[-15:], 1):  # Show last 15 lines
            print(f"{i:2d}: {line}")
        print("-" * 40)
    
    # Provide recommendations
    status = status_info['status']
    print("\n💡 Recommendations:")
    
    if status == 'stuck':
        print("  • Session appears to be hanging")
        print(f"  • Try attaching to check: llmlib tmux attach {session_name}")
        print("  • Consider killing and restarting if truly stuck")
        print(f"  • Kill command: llmlib tmux kill {session_name}")
        
    elif status == 'failed':
        print("  • Training has failed with errors")
        print(f"  • Attach to see full error: llmlib tmux attach {session_name}")
        print("  • Check config file and fix issues before restarting")
        print("  • Consider cleaning up failed session")
        
    elif status == 'completed':
        print("  • Training completed successfully!")
        print(f"  • You can clean up this session: llmlib tmux kill {session_name}")
        
    elif status == 'running' or status == 'training':
        print("  • Training is progressing normally")
        print(f"  • Monitor progress: llmlib tmux monitor")
        print(f"  • Attach to see details: llmlib tmux attach {session_name}")
        
    else:
        print("  • Status unclear, manual inspection recommended")
        print(f"  • Attach to session: llmlib tmux attach {session_name}")


def main():
    """Main CLI entry point for tmux session management."""
    parser = argparse.ArgumentParser(
        description="TMux Session Manager for LLM Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start training in tmux session')
    start_parser.add_argument('--config', type=str, required=True, help='Config file path')
    start_parser.add_argument('--name', type=str, help='Session name (auto-generated if not provided)')
    start_parser.add_argument('--dry-run', action='store_true', help='Dry run validation only')
    start_parser.add_argument('--max-retries', type=int, default=3, help='Max retry attempts')
    start_parser.add_argument('--timeout', type=int, default=8, help='Timeout in hours')
    start_parser.add_argument('--auto-confirm', action='store_true', help='Skip confirmation')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List active training sessions')
    
    # Attach command
    attach_parser = subparsers.add_parser('attach', help='Attach to training session')
    attach_parser.add_argument('session', nargs='?', help='Session name (interactive if not provided)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor training sessions')
    
    # Kill command
    kill_parser = subparsers.add_parser('kill', help='Kill training sessions')
    kill_parser.add_argument('session', nargs='?', help='Session name (interactive if not provided)')
    kill_parser.add_argument('--all', action='store_true', help='Kill all training sessions')
    
    # Status command (alias for list)
    status_parser = subparsers.add_parser('status', help='Show training session status')
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Diagnose session issues')
    diagnose_parser.add_argument('session', nargs='?', help='Session name (interactive if not provided)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'start':
            config_path = Path(args.config).resolve()
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                sys.exit(1)
            
            start_training_session(
                config_path=config_path,
                session_name=args.name,
                dry_run=args.dry_run,
                max_retries=args.max_retries,
                timeout=args.timeout,
                auto_confirm=args.auto_confirm
            )
        
        elif args.command in ['list', 'status']:
            list_training_sessions()
        
        elif args.command == 'attach':
            attach_to_session(args.session)
        
        elif args.command == 'monitor':
            monitor_sessions()
        
        elif args.command == 'kill':
            kill_training_session(args.session, args.all)
        
        elif args.command == 'diagnose':
            diagnose_session(args.session)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
