#!/usr/bin/env python3
"""
Training Monitor CLI

Monitors training progress, system resources, and provides utilities for
managing long-running training jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from llmlib.utils.logger import get_logger

logger = get_logger(__name__)


def get_gpu_stats() -> Dict[str, Any]:
    """Get current GPU statistics."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, name, mem_used, mem_total, util, temp = line.split(', ')
                gpus.append({
                    'index': int(idx),
                    'name': name,
                    'memory_used': int(mem_used),
                    'memory_total': int(mem_total),
                    'memory_percent': round(int(mem_used) / int(mem_total) * 100, 1),
                    'utilization': int(util),
                    'temperature': int(temp)
                })
        return {'gpus': gpus, 'available': True}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'gpus': [], 'available': False}


def get_system_stats() -> Dict[str, Any]:
    """Get system resource statistics."""
    import psutil
    import shutil
    
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory info
    memory = psutil.virtual_memory()
    
    # Disk info
    total, used, free = shutil.disk_usage(Path.cwd())
    
    return {
        'cpu': {
            'percent': cpu_percent,
            'count': cpu_count
        },
        'memory': {
            'total': memory.total,
            'used': memory.used,
            'percent': memory.percent,
            'available': memory.available
        },
        'disk': {
            'total': total,
            'used': used,
            'free': free,
            'percent': round(used / total * 100, 1)
        }
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format."""
    value = float(bytes_value)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def monitor_resources(interval: int = 30, duration: Optional[int] = None):
    """Monitor system resources continuously."""
    logger.info(f"🔍 Starting resource monitor (interval: {interval}s)")
    if duration:
        logger.info(f"⏱️  Will run for {duration} seconds")
    
    start_time = time.time()
    
    try:
        while True:
            current_time = datetime.now()
            elapsed = time.time() - start_time
            
            # Get stats
            gpu_stats = get_gpu_stats()
            sys_stats = get_system_stats()
            
            # Clear screen and print header
            os.system('clear')
            print("=" * 80)
            print(f"🖥️  LLMLIB TRAINING MONITOR - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Elapsed: {timedelta(seconds=int(elapsed))}")
            print("=" * 80)
            
            # System resources
            print("\n📊 SYSTEM RESOURCES:")
            print(f"  CPU: {sys_stats['cpu']['percent']:.1f}% ({sys_stats['cpu']['count']} cores)")
            print(f"  Memory: {sys_stats['memory']['percent']:.1f}% "
                  f"({format_bytes(sys_stats['memory']['used'])}/{format_bytes(sys_stats['memory']['total'])})")
            print(f"  Disk: {sys_stats['disk']['percent']:.1f}% "
                  f"({format_bytes(sys_stats['disk']['free'])} free)")
            
            # GPU resources
            if gpu_stats['available']:
                print("\n🎮 GPU RESOURCES:")
                for gpu in gpu_stats['gpus']:
                    print(f"  GPU {gpu['index']} ({gpu['name']}):")
                    print(f"    Utilization: {gpu['utilization']}%")
                    print(f"    Memory: {gpu['memory_percent']}% "
                          f"({gpu['memory_used']}/{gpu['memory_total']} MB)")
                    print(f"    Temperature: {gpu['temperature']}°C")
            else:
                print("\n🎮 GPU: Not available or nvidia-smi not found")
            
            # Check for training processes
            print("\n🚀 TRAINING PROCESSES:")
            try:
                result = subprocess.run(['pgrep', '-f', 'modern-gpt-train'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    print(f"  Found {len(pids)} training process(es): {', '.join(pids)}")
                else:
                    print("  No training processes found")
            except:
                print("  Could not check training processes")
            
            print("\n" + "=" * 80)
            print("Press Ctrl+C to stop monitoring")
            
            # Check duration
            if duration and elapsed >= duration:
                logger.info(f"✅ Monitoring completed after {duration} seconds")
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Monitoring stopped by user")


def find_training_logs(project_dir: Optional[Path] = None) -> list[Path]:
    """Find training log files."""
    search_paths = []
    
    if project_dir:
        search_paths.append(project_dir)
    
    # Add common log locations
    search_paths.extend([
        Path.cwd(),
        Path.home() / "logs",
        Path("/tmp")
    ])
    
    log_files = []
    for path in search_paths:
        if path.exists():
            # Look for common log patterns
            for pattern in ["*.log", "training_*.txt", "llm_*.log"]:
                log_files.extend(path.glob(pattern))
    
    return sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)


def tail_logs(log_file: Path, lines: int = 50):
    """Tail training logs."""
    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        return
    
    logger.info(f"📜 Tailing last {lines} lines from: {log_file}")
    logger.info("=" * 60)
    
    try:
        result = subprocess.run(['tail', '-f', '-n', str(lines), str(log_file)], 
                              check=True)
    except KeyboardInterrupt:
        logger.info("\n🛑 Log tailing stopped")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to tail logs: {e}")


def kill_training_processes():
    """Kill all running training processes."""
    try:
        result = subprocess.run(['pgrep', '-f', 'modern-gpt-train'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            logger.info(f"🛑 Found {len(pids)} training process(es) to kill")
            
            for pid in pids:
                logger.info(f"Killing process {pid}...")
                subprocess.run(['kill', '-TERM', pid], check=False)
            
            # Wait a bit, then force kill if needed
            time.sleep(5)
            result = subprocess.run(['pgrep', '-f', 'modern-gpt-train'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                remaining_pids = result.stdout.strip().split('\n')
                logger.warning(f"Force killing {len(remaining_pids)} remaining processes")
                for pid in remaining_pids:
                    subprocess.run(['kill', '-KILL', pid], check=False)
            
            logger.info("✅ Training processes killed")
        else:
            logger.info("ℹ️  No training processes found")
            
    except Exception as e:
        logger.error(f"Failed to kill training processes: {e}")


def main():
    """Main CLI entry point for training monitoring utilities."""
    parser = argparse.ArgumentParser(
        description="Training Monitor and Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor system resources')
    monitor_parser.add_argument('--interval', type=int, default=30, 
                               help='Update interval in seconds (default: 30)')
    monitor_parser.add_argument('--duration', type=int, 
                               help='Monitor duration in seconds (default: infinite)')
    
    # GPU command
    gpu_parser = subparsers.add_parser('gpu', help='Show GPU status')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show training logs')
    logs_parser.add_argument('--file', type=str, help='Specific log file to tail')
    logs_parser.add_argument('--lines', type=int, default=50, 
                            help='Number of lines to show (default: 50)')
    
    # Kill command
    kill_parser = subparsers.add_parser('kill', help='Kill training processes')
    kill_parser.add_argument('--force', action='store_true', 
                            help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'monitor':
        monitor_resources(args.interval, args.duration)
    
    elif args.command == 'gpu':
        gpu_stats = get_gpu_stats()
        if gpu_stats['available']:
            print("🎮 GPU Status:")
            for gpu in gpu_stats['gpus']:
                print(f"  GPU {gpu['index']} ({gpu['name']}):")
                print(f"    Utilization: {gpu['utilization']}%")
                print(f"    Memory: {gpu['memory_percent']}% "
                      f"({gpu['memory_used']}/{gpu['memory_total']} MB)")
                print(f"    Temperature: {gpu['temperature']}°C")
        else:
            print("❌ GPU not available or nvidia-smi not found")
    
    elif args.command == 'logs':
        if args.file:
            log_file = Path(args.file)
        else:
            log_files = find_training_logs()
            if not log_files:
                logger.error("No log files found")
                return
            log_file = log_files[0]  # Most recent
            logger.info(f"Using most recent log file: {log_file}")
        
        tail_logs(log_file, args.lines)
    
    elif args.command == 'kill':
        if not args.force:
            response = input("🤔 Are you sure you want to kill all training processes? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Operation cancelled")
                return
        
        kill_training_processes()


if __name__ == '__main__':
    main()
