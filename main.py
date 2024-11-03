from io import StringIO
import streamlit as st
import pandas as pd
import csv
from typing import List, Tuple, Dict
import logging
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EXPECTED_TOTAL_COLUMNS = 48

class ProcessingSummary:
    def __init__(self):
        self.errors = defaultdict(list)
        self.warnings = defaultdict(list)
        self.success_messages = []
        self.total_rows = 0
        self.rows_with_issues = 0
    
    def add_message(self, message: str, line_num: int = None):
        if any(error_term in message.lower() for error_term in ['error', 'missing']):
            key = message.split(':', 1)[0] if ':' in message else message
            if line_num:
                self.errors[key].append(line_num)
            else:
                self.errors[key].append(message)
            self.rows_with_issues += 1
        elif 'fixed' in message.lower() or 'warning' in message.lower():
            key = message.split(':', 1)[0] if ':' in message else message
            if line_num:
                self.warnings[key].append(line_num)
            else:
                self.warnings[key].append(message)
            self.rows_with_issues += 1
        else:
            self.success_messages.append(message)

def fix_row_with_extra_commas(row: List[str], expected_columns: int) -> Tuple[List[str], str]:
    """
    Fix a row that has more or fewer columns than expected by truncating or padding.
    
    Args:
        row: The row to fix
        expected_columns: The expected number of columns
    
    Returns:
        Tuple of (fixed_row, message)
    """
    if len(row) > expected_columns:
        fix_message = f"Fixed row with {len(row)} columns by truncating to {expected_columns} columns"
        return row[:expected_columns], fix_message
    elif len(row) < expected_columns:
        fix_message = f"Fixed row with {len(row)} columns by padding to {expected_columns} columns"
        return row + [''] * (expected_columns - len(row)), fix_message
    return row, ""

def custom_csv_parser(file_obj) -> Tuple[List[Dict], ProcessingSummary]:
    """
    Custom CSV parser to handle inconsistent column counts and fix rows with extra commas
    while preserving required columns.
    """
    summary = ProcessingSummary()
    parsed_data = []
    required_columns = ['Key', 'Description', 'Customizable', 'Can Be Empty']
    
    try:
        # Use StringIO to read the file content
        content = StringIO(file_obj.getvalue().decode('utf-8'))
        
        # Count total rows for progress bar
        total_rows = sum(1 for _ in content) - 1  # Subtract header row
        content.seek(0)
        
        # Use csv.reader with proper quoting
        reader = csv.reader(
            content,
            quoting=csv.QUOTE_MINIMAL,
            delimiter=',',
            quotechar='"'
        )
        
        # Get header row
        header = next(reader)
        
        # Validate total number of columns
        if len(header) != EXPECTED_TOTAL_COLUMNS:
            summary.add_message(f"Invalid number of columns: got {len(header)}, expected {EXPECTED_TOTAL_COLUMNS}")
            # Fix header if needed
            if len(header) < EXPECTED_TOTAL_COLUMNS:
                header = header + [f'Column_{i+1}' for i in range(len(header), EXPECTED_TOTAL_COLUMNS)]
            else:
                header = header[:EXPECTED_TOTAL_COLUMNS]
        
        # Verify required columns exist
        missing_columns = set(required_columns) - set(header)
        if missing_columns:
            summary.add_message(f"Missing required columns: {', '.join(missing_columns)}")
            return [], summary
        
        # Get indices of required columns
        req_col_indices = {col: header.index(col) for col in required_columns if col in header}
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        
        # Process each row
        for line_num, row in enumerate(reader, start=2):
            try:
                # Update progress
                progress = (line_num - 1) / total_rows
                progress_bar.progress(progress)
                
                summary.total_rows += 1
                
                if len(row) != EXPECTED_TOTAL_COLUMNS:
                    # Fix row with extra or missing commas
                    fixed_row, fix_msg = fix_row_with_extra_commas(row, EXPECTED_TOTAL_COLUMNS)
                    if fix_msg:
                        summary.add_message(fix_msg, line_num)
                        logger.info(f"Line {line_num}: {fix_msg}")
                    row = fixed_row
                
                if len(row) < max(req_col_indices.values()) + 1:
                    msg = f"Cannot fix row - missing required columns"
                    summary.add_message(msg, line_num)
                    logger.error(f"Line {line_num}: {msg}")
                    continue
                
                # Create row data preserving required columns
                row_data = {}
                for i, col in enumerate(header):
                    if i < len(row):
                        row_data[col] = row[i]
                    else:
                        row_data[col] = ''
                
                parsed_data.append(row_data)
                
            except Exception as e:
                summary.add_message(f"Error processing row: {str(e)}", line_num)
                logger.error(f"Line {line_num}: Error processing row - {str(e)}")
        
        # Complete progress bar
        progress_bar.progress(1.0)
        # Remove progress bar
        progress_bar.empty()
        
        summary.add_message("CSV processing completed successfully")
    
    except Exception as e:
        summary.add_message(f"Error parsing CSV: {str(e)}")
        return [], summary
    
    return parsed_data, summary

def process_csv(uploaded_file):
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Parse CSV with custom parser that handles and fixes problematic rows
        parsed_data, summary = custom_csv_parser(uploaded_file)
        if not parsed_data:
            return None, summary
        
        df = pd.DataFrame(parsed_data)
        
        # Ensure required columns are present and preserved
        preserved_columns = ['Key', 'Description', 'Customizable', 'Can Be Empty']
        missing_columns = set(preserved_columns) - set(df.columns)
        if missing_columns:
            summary.add_message(f"Missing required columns: {', '.join(missing_columns)}")
            return None, summary
        
        # Create new DataFrame with preserved columns first
        new_df = pd.DataFrame()
        
        # Add preserved columns first
        for col in preserved_columns:
            if col in df.columns:
                new_df[col] = df[col]
        
        # Add remaining columns
        remaining_cols = [col for col in df.columns if col not in preserved_columns]
        new_df = pd.concat([new_df, df[remaining_cols]], axis=1)
        
        # Validate total number of columns
        if len(new_df.columns) != EXPECTED_TOTAL_COLUMNS:
            summary.add_message(f"Warning: Output file has {len(new_df.columns)} columns, expected {EXPECTED_TOTAL_COLUMNS}")
        
        return new_df, summary
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        summary = ProcessingSummary()
        summary.add_message(f"Error processing CSV: {str(e)}")
        return None, summary

def main():
    st.title("📋 Clean Locale Configuration File")
    st.write("Upload a CSV file to clean and validate its contents.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Process the uploaded file
        df, summary = process_csv(uploaded_file)
        
        if df is not None:
            # Data Preview Container
            with st.container():
                st.markdown("### 📊 Data Preview")
                st.dataframe(df)
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Download Options Container
            with st.container():
                st.markdown("### ⬇️ Download Options")
                col1, col2 = st.columns(2)
                
                # Download button for processed CSV file
                with col1:
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="processed_locale_config.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"  # Green styling
                    )
                
                # Download button for JSON file
                with col2:
                    json_str = df.to_json(orient='records', force_ascii=False)
                    json_bytes = json_str.encode('utf-8')
                    st.download_button(
                        label="Download JSON",
                        data=json_bytes,
                        file_name="processed_locale_config.json",
                        mime="application/json",
                        use_container_width=True,
                        type="primary"  # Green styling
                    )
                st.markdown("<br>", unsafe_allow_html=True)

            # Processing Results Container
            with st.container():
                st.markdown("### 📑 Processing Results")
                
                # Display total statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows Processed", summary.total_rows)
                with col2:
                    st.metric("Rows with Issues", summary.rows_with_issues)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Create tabs for different message types
                if summary.errors or summary.warnings or summary.success_messages:
                    tab_list = []
                    if summary.errors:
                        tab_list.append("❌ Errors")
                    if summary.warnings:
                        tab_list.append("⚠️ Warnings")
                    if summary.success_messages:
                        tab_list.append("✅ Success")
                    
                    tabs = st.tabs(tab_list)
                    
                    tab_index = 0
                    if summary.errors:
                        with tabs[tab_index]:
                            for error_type, lines in summary.errors.items():
                                if isinstance(lines[0], int):
                                    st.error(f"{error_type} at lines: {', '.join(map(str, lines))}")
                                else:
                                    st.error(lines[0])
                        tab_index += 1
                    
                    if summary.warnings:
                        with tabs[tab_index]:
                            for warning_type, lines in summary.warnings.items():
                                if isinstance(lines[0], int):
                                    st.warning(f"{warning_type} at lines: {', '.join(map(str, lines))}")
                                else:
                                    st.warning(lines[0])
                        tab_index += 1
                    
                    if summary.success_messages:
                        with tabs[tab_index]:
                            for msg in summary.success_messages:
                                st.success(msg)

if __name__ == "__main__":
    main()
