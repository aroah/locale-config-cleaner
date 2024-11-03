import pandas as pd
import logging
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVProcessor:
    def __init__(self):
        self.expected_column_count = None
        self.problematic_lines = []

    def process_csv(self, csv_content: str) -> tuple[pd.DataFrame, list]:
        """
        Process CSV content with robust error handling for inconsistent column counts.
        
        Args:
            csv_content (str): The CSV content as a string
            
        Returns:
            tuple: (processed_df, list of error messages)
        """
        try:
            # First pass to determine expected column count
            first_pass = pd.read_csv(
                StringIO(csv_content), 
                nrows=5,  # Read first few rows to determine structure
                on_bad_lines='skip'
            )
            self.expected_column_count = len(first_pass.columns)
            logger.info(f"Expected column count: {self.expected_column_count}")

            # Custom line processing to identify problematic lines
            lines = csv_content.split('\n')
            cleaned_lines = []
            for i, line in enumerate(lines, 1):
                if not line.strip():  # Skip empty lines
                    continue
                    
                fields = line.split(',')
                if len(fields) != self.expected_column_count:
                    error_msg = f"Line {i}: Found {len(fields)} fields instead of {self.expected_column_count}"
                    logger.warning(error_msg)
                    self.problematic_lines.append({
                        'line_number': i,
                        'content': line,
                        'field_count': len(fields)
                    })
                    
                    # Attempt to fix the line by combining quoted fields
                    fixed_line = self._fix_quoted_fields(line)
                    if fixed_line:
                        cleaned_lines.append(fixed_line)
                    else:
                        # If can't fix, truncate or pad the line
                        fixed_line = self._normalize_line(fields, self.expected_column_count)
                        cleaned_lines.append(fixed_line)
                else:
                    cleaned_lines.append(line)

            # Process the cleaned content
            cleaned_content = '\n'.join(cleaned_lines)
            df = pd.read_csv(
                StringIO(cleaned_content),
                on_bad_lines='warn',
                encoding='utf-8',
                low_memory=False
            )

            error_messages = [
                f"Found inconsistent line at {p['line_number']}: {p['content'][:100]}... "
                f"(fields: {p['field_count']})"
                for p in self.problematic_lines
            ]

            return df, error_messages

        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

    def _fix_quoted_fields(self, line: str) -> str:
        """
        Attempt to fix lines with quoted fields containing commas.
        """
        try:
            # Use pandas to properly parse quoted fields
            df = pd.read_csv(StringIO(line), header=None, quoting=1)
            if len(df.columns) == self.expected_column_count:
                return line
        except:
            pass
        return None

    def _normalize_line(self, fields: list, expected_count: int) -> str:
        """
        Normalize line to have the expected number of fields.
        """
        if len(fields) > expected_count:
            # Combine extra fields if they appear to be incorrectly split
            fields = fields[:expected_count-1] + [','.join(fields[expected_count-1:])]
        else:
            # Pad with empty values if there are too few fields
            fields.extend([''] * (expected_count - len(fields)))
        
        return ','.join(fields)

    def get_validation_summary(self) -> dict:
        """
        Return a summary of validation results.
        """
        return {
            'total_problematic_lines': len(self.problematic_lines),
            'problematic_line_numbers': [p['line_number'] for p in self.problematic_lines],
            'expected_column_count': self.expected_column_count
        }
