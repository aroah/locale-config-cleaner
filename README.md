# CSV Locale Configuration Cleaner

A web application built with Streamlit for cleaning and validating locale configuration CSV files.

## Features

- CSV file upload and processing
- Robust error handling for inconsistent columns
- Preservation of key columns (Key, Description, Customizable, Can Be Empty)
- Data validation and error reporting
- Export to CSV and JSON formats
- Interactive data preview

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run main.py
   ```

3. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose a CSV file" button to upload your locale configuration CSV file
2. The application will process the file and display:
   - Data preview
   - Processing results with any errors or warnings
   - Download options for the processed file
3. Use the download buttons to get your cleaned data in CSV or JSON format

## File Requirements

The input CSV file should have the following characteristics:
- Must contain the columns: Key, Description, Customizable, Can Be Empty
- Expected to have 48 total columns
- UTF-8 encoded

## Error Handling

The application handles various CSV issues:
- Missing columns
- Extra columns
- Inconsistent column counts
- Data validation errors

## Deployment

The application is configured to run on port 5000 and listen on all interfaces (0.0.0.0).
Configuration can be modified in `.streamlit/config.toml`.
