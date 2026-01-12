# HEMA Ratings Scraper

A Python scraper for [hemaratings.com](https://hemaratings.com).

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the scraper:
```bash
python scraper.py
```

This will:
1. Fetch all available rating sets.
2. Scrape the first rating set (Mixed & Men's Steel Longsword) by default.
3. Save the results to a CSV file (e.g., `rankings_1.csv`).

## Structure
- `src/scraper.py`: Main script containing `HEMARatingsScraper` class.
- `requirements.txt`: Python dependencies.


## Data Pipeline
1. Scrape fighter ratings and match histories
2. Reconstruct fighter state chronologically
3. Generate pre-match feature dataset
4. Train and evaluate predictive models

