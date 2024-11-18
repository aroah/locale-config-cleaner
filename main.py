from io import StringIO
import streamlit as st
import pandas as pd
import csv
from typing import List, Tuple, Dict
import logging
import json
from collections import defaultdict
from langdetect import detect, LangDetectException
import openai
from time import sleep, time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EXPECTED_TOTAL_COLUMNS = 48
LANGUAGE_COLUMNS = [
    'en_US', 'en_GB', 'fr_FR', 'el_GR', 'tr_TR', 'zh_CN', 'ar_WW', 'de_DE', 
    'es_ES', 'es_MX', 'fr_CA', 'it_IT', 'ja_JP', 'ko_KR', 'nl_NL', 'pt_PT', 
    'ru_RU', 'sv_SE', 'pl_PL', 'cs_CZ', 'da_DK', 'fi_FI', 'hi_IN', 'hu_HU', 
    'nb_NO', 'pt_BR', 'ro_RO', 'th_TH', 'zh_TW', 'ca_ES'
]

LANG_CODE_MAPPING = {
    'en': ['en_US', 'en_GB'],
    'fr': ['fr_FR', 'fr_CA'],
    'el': ['el_GR'],
    'tr': ['tr_TR'],
    'zh': ['zh_CN', 'zh_TW'],
    'ar': ['ar_WW'],
    'de': ['de_DE'],
    'es': ['es_ES', 'es_MX'],
    # ... add other mappings
}

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

def custom_csv_parser(file_obj) -> Tuple[List[Dict], ProcessingSummary, float]:
    """Custom CSV parser that now returns processing time"""
    start_time = time()
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
            return [], summary, 0
        
        # Get indices of required columns
        req_col_indices = {col: header.index(col) for col in required_columns if col in header}
        
        # Initialize progress bar in a placeholder
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
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
        progress_placeholder.empty()
        
        summary.add_message("CSV processing completed successfully")
    except Exception as e:
        summary.add_message(f"Error parsing CSV: {str(e)}")
        return [], summary, 0
    
    processing_time = time() - start_time
    return parsed_data, summary, processing_time

def detect_language_gpt(text: str, api_key: str) -> str:
    """Use GPT to detect language when langdetect fails"""
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a language detection expert. Respond only with the ISO 639-1 language code."},
                {"role": "user", "content": f"What language is this text in? Respond only with the language code: {text}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.error(f"GPT language detection failed: {str(e)}")
        return None

def detect_text_language(text: str, api_key: str) -> str:
    """Detect language using langdetect first, fallback to GPT"""
    if not isinstance(text, str) or not text.strip():
        return None
    
    try:
        return detect(text)
    except LangDetectException:
        return detect_language_gpt(text, api_key)

def find_correct_column(detected_lang: str) -> str:
    """Find the correct column for a detected language code"""
    for lang_code, columns in LANG_CODE_MAPPING.items():
        if detected_lang == lang_code:
            return columns[0]  # Return first matching column
    return None

def detect_language_gpt_batch(texts: List[str], api_key: str) -> List[str]:
    """Use GPT to detect language for multiple texts at once"""
    client = openai.OpenAI(api_key=api_key)
    try:
        # Combine texts into a single request
        combined_text = "\n---\n".join([f"Text {i+1}: {text}" for i, text in enumerate(texts)])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a language detection expert. For each text provided, respond with its ISO 639-1 language code. Format your response as a comma-separated list of codes in order."},
                {"role": "user", "content": f"What language are these texts in? Respond only with comma-separated language codes:\n{combined_text}"}
            ],
            temperature=0
        )
        
        # Parse the response into a list of language codes
        lang_codes = [code.strip().lower() for code in response.choices[0].message.content.split(',')]
        return lang_codes
    except Exception as e:
        logger.error(f"GPT batch language detection failed: {str(e)}")
        return [None] * len(texts)

def process_csv(uploaded_file, api_key: str):
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Parse CSV and get timing
        csv_start = time()
        parsed_data, summary, csv_time = custom_csv_parser(uploaded_file)
        if not parsed_data:
            return None, summary, (0, 0)
        
        df = pd.DataFrame(parsed_data)
        
        # Language detection timing
        lang_start = time()
        st.write("Checking language columns... This may take a few minutes.")
        
        # Add progress bar for language detection
        total_cells = len(df) * len(LANGUAGE_COLUMNS)
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        time_container = st.empty()
        start_time = time()
        
        # Process in larger batches for GPT
        batch_size = 20
        cells_processed = 0
        
        try:
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]
                
                # Collect texts that need GPT processing
                gpt_texts = []
                text_locations = []  # Store (idx, col) pairs for mapping back results
                
                for idx, row in batch.iterrows():
                    for col in LANGUAGE_COLUMNS:
                        cells_processed += 1
                        progress = cells_processed / total_cells
                        progress_bar.progress(progress)
                        
                        # Calculate time remaining
                        elapsed_time = time() - start_time
                        if progress > 0:
                            estimated_total = elapsed_time / progress
                            remaining_time = estimated_total - elapsed_time
                            time_container.text(f"Estimated time remaining: {remaining_time:.1f} seconds")
                        
                        if pd.notna(row[col]) and isinstance(row[col], str) and len(row[col].strip()) > 0:
                            text = row[col].strip()
                            if len(text) < 3:  # Skip very short texts
                                continue
                            
                            try:
                                # Try langdetect first (faster)
                                detected_lang = detect(text)
                            except LangDetectException:
                                # Only use GPT for longer texts
                                if len(text) > 10:
                                    gpt_texts.append(text)
                                    text_locations.append((idx, col))
                                continue
                            
                            # Process langdetect results immediately
                            if detected_lang:
                                correct_col = find_correct_column(detected_lang)
                                if correct_col and correct_col != col:
                                    if pd.isna(df.at[idx, correct_col]) or df.at[idx, correct_col] == '':
                                        df.at[idx, correct_col] = text
                                        df.at[idx, col] = ''
                                        summary.add_message(
                                            f"Moving text from {col} to {correct_col} (row {idx+2})",
                                            idx+2
                                        )
                                    else:
                                        summary.add_message(
                                            f"Could not move text from {col} to {correct_col} (row {idx+2}) - target column already contains content",
                                            idx+2
                                        )
                
                # Process GPT batch if we have any texts
                if gpt_texts:
                    detected_langs = detect_language_gpt_batch(gpt_texts, api_key)
                    
                    # Process GPT results
                    for (idx, col), detected_lang in zip(text_locations, detected_langs):
                        if detected_lang:
                            correct_col = find_correct_column(detected_lang)
                            if correct_col and correct_col != col:
                                text = df.at[idx, col]
                                if pd.isna(df.at[idx, correct_col]) or df.at[idx, correct_col] == '':
                                    df.at[idx, correct_col] = text
                                    df.at[idx, col] = ''
                                    summary.add_message(
                                        f"Moving text from {col} to {correct_col} (row {idx+2})",
                                        idx+2
                                    )
                
                # Add a small sleep to prevent API rate limiting
                sleep(0.1)
            
            # Ensure progress elements are always cleared
            progress_placeholder.empty()
            time_container.empty()
            
            # Ensure required columns are present and preserved
            preserved_columns = ['Key', 'Description', 'Customizable', 'Can Be Empty']
            missing_columns = set(preserved_columns) - set(df.columns)
            if missing_columns:
                summary.add_message(f"Missing required columns: {', '.join(missing_columns)}")
                return None, summary, (0, 0)
            
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
            
            lang_time = time() - lang_start
            
            # Add timing information to summary
            summary.add_message(f"CSV cleaning completed in {csv_time:.2f} seconds")
            summary.add_message(f"Language detection completed in {lang_time:.2f} seconds")
            
            return new_df, summary, (csv_time, lang_time)
        finally:
            # Ensure progress elements are always cleared
            progress_placeholder.empty()
            time_container.empty()
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        summary = ProcessingSummary()
        summary.add_message(f"Error processing CSV: {str(e)}")
        return None, summary, (0, 0)

def main():
    st.title("📋 Clean Locale Configuration File")
    st.write("Upload a CSV file to clean and validate its contents.")
    
    # Add API key (in production, use st.secrets or environment variables)
    api_key = st.secrets["OPENAI_KEY"]
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Process CSV and get timing information
        df, summary, (csv_time, lang_time) = process_csv(uploaded_file, api_key)
        
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

            # Timing Information Container
            with st.container():
                st.markdown("### ⏱️ Processing Times")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CSV Cleaning", f"{csv_time:.2f}s")
                with col2:
                    st.metric("Language Detection", f"{lang_time:.2f}s")
                with col3:
                    st.metric("Total Time", f"{(csv_time + lang_time):.2f}s")
                st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
