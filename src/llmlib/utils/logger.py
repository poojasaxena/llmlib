import logging
import sys

class CompactFormatter(logging.Formatter):
    """Minimal formatter - just time, level, and message with perfect alignment."""
    
    def format(self, record):
        # Just minutes:seconds for very compact time
        time_part = self.formatTime(record, "%M:%S")
        
        # Ultra-short level codes for compactness
        level_map = {
            'DEBUG': 'D',
            'INFO': 'I', 
            'WARNING': 'W',
            'ERROR': 'E',
            'CRITICAL': 'C'
        }
        level_part = level_map.get(record.levelname, record.levelname[0])
        
        # Format: MM:SS | L | message (super compact but aligned)
        return f"{time_part} | {level_part} | {record.getMessage()}"

# Centralized logging config with compact formatter
def setup_logger():
    """Set up the compact logger format."""
    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handler with compact formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CompactFormatter())
    
    # Set up root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

# Initialize on import
setup_logger()

def get_logger(name: str) -> logging.Logger:
    """Return a logger instance for the given module/class."""
    return logging.getLogger(name)
