import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "../data/raw"

CURRENT_RANKINGS_FILE = RAW_DATA_DIR / "rankings_1.csv"
MATCH_HISTORY_FILE = RAW_DATA_DIR / "tournament_histories.csv"
RATINGS_HISTORY_FILE = RAW_DATA_DIR / "fighter_ratings_history.csv"
class HEMARatingsScraper:
    BASE_URL = "https://hemaratings.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def get_rating_sets(self):
        """
        Fetches the list of available rating sets from the homepage.
        Returns a list of dicts with 'name' and 'url'.
        """
        print(f"Fetching rating sets from {self.BASE_URL}...")
        response = self.session.get(self.BASE_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        rating_sets = []
        # Rating sets are listed in the 'Ratings' section. 
        # Inspecting the HTML structure from previous research:
        # headers are likely in a container. Let's look for links containing "periods/details/?ratingsetid="
        
        links = soup.find_all('a', href=re.compile(r'/periods/details/\?ratingsetid='))
        for link in links:
            name = link.text.strip()
            url = self.BASE_URL + link['href'] if not link['href'].startswith('http') else link['href']
            rating_set_id = link['href'].split('ratingsetid=')[1]
            rating_sets.append({
                'name': name,
                'url': url,
                'id': rating_set_id
            })
            print(f"Found rating set: {name} (ID: {rating_set_id})")
        
        return rating_sets

    def get_rankings(self, ratingset_id):
        """
        Fetches the rankings for a specific rating set.
        """
        url = f"{self.BASE_URL}/periods/details/?ratingsetid={ratingset_id}"
        print(f"Fetching rankings from {url}...")
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        fighters = []
        
        # Rankings are typically in a table
        table = soup.find('table', class_='table')
        if not table:
            print("No ranking table found.")
            return []

        rows = table.find_all('tr')[1:] # Skip header
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            
            # Identify columns based on standard layout (Rank, Fighter, Club, Rating, etc.)
            # This might need adjustment if logic is complex, but usually:
            # 0: Rank, 1: Fighter Name (link), 2: Club (link), 3: Rating, ...
            
            try:
                rank = cols[0].text.strip().strip('.')
                
                fighter_link = cols[2].find('a')
                fighter_name = fighter_link.text.strip()
                fighter_url = self.BASE_URL + fighter_link['href']
                fighter_id = fighter_link['href'].split('/')[3] 
                
                club_link = cols[4].find('a')
                club_name = club_link.text.strip() if club_link else cols[4].text.strip()
                
                rating = cols[5].text.strip()

                confidence = ""
                if len(cols) > 7:
                    icon = cols[7].find('i')
                    if icon and icon.get('title'):
                        confidence = icon['title']
                        confidence = re.findall(r'\d+\.?\d*', confidence)[0]
                
                fighters.append({
                    'rank': rank,
                    'name': fighter_name,
                    'id': fighter_id,
                    'club': club_name,
                    'rating': rating,
                    'confidence': confidence,
                    'url': fighter_url
                })
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

        print(f"Found {len(fighters)} fighters in this set.")
        return fighters

    def get_match_history(self, fighter_id):
        """
        Fetches detailed stats for a fighter.
        """
        url = f"{self.BASE_URL}/fighters/details/{fighter_id}/"
        print(f"Fetching details for fighter {fighter_id}...")
        response = self.session.get(url)
        if response.status_code != 200:
            return {}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        try:
            longsword_tournament_history = soup.find('div', id='accordion_results_0.')
            if not longsword_tournament_history:
                return []
            tournaments = longsword_tournament_history.find_all('div', id=re.compile(r'accordion_tournament_\d+_\d+\.'))
            tournament_histories = []
            for t in tournaments:
                tournament_name = t.find('h4', class_='panel-title').find('span')
                tournament_name = tournament_name.text.strip()
                tournament_date, tournament_name = tournament_name.split('-')[0], tournament_name.split('-')[1]
                table = t.find('table')
                if not table:
                    return []
                
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if not cols:
                        continue # Header row or empty
                        
                    try:
                        tournament_histories.append({
                        'fighter_id': fighter_id,
                        'division': cols[0].text.strip() if cols[0].text.strip() else None,
                        'stage': cols[1].text.strip() if cols[1].text.strip() else None,
                        'tournament_name': tournament_name,
                        'tournament_date': tournament_date,
                        'opponent': cols[2].text.strip() if cols[2].text.strip() else None,
                        'opponent_id': cols[2].find('a')['href'].split('/')[-2] if cols[2].find('a') else None,
                        'outcome': cols[3].text.strip() if cols[3].text.strip() else None,
                        'opponent_rating': cols[4].text.strip() if cols[4].text.strip() else None,
                        'win_chance': cols[5].text.strip() if cols[5].text.strip() else None
                        })
                    except (IndexError, AttributeError) as row_err:
                        print(f"Skipping malformed row: {row_err}")
                    continue
            return tournament_histories
        except Exception as e:
            print(f"Error parsing match history: {e}")
            tournament_histories = [] 
        return tournament_histories
        
    def get_ratings_history(self, fighter_id):
        ratings_history = []
        url = f"{self.BASE_URL}/fighters/details/{fighter_id}/"
        response = self.session.get(url)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        print('getting ratings history for fighter', fighter_id)     
        longsword_ratings_history_data = soup.find_all('div', class_='rating-history-data')[0]
        if not longsword_ratings_history_data:
            return []
        for rating_history_element in longsword_ratings_history_data.find_all('div', class_='rating-history-data-element'):
            try:
                date = rating_history_element.find('div', class_='rating-history-data-element-period').text.strip() if rating_history_element.find('div', class_='rating-history-data-element-period') else None
                rank = rating_history_element.find('div', class_='rating-history-data-element-rank').text.strip() if rating_history_element.find('div', class_='rating-history-data-element-rank') else None
                rating = rating_history_element.find('div', class_='rating-history-data-element-rating').text.strip() if rating_history_element.find('div', class_='rating-history-data-element-rating') else None
            
                ratings_history.append({
                    'fighter_id': fighter_id,
                    'date': date,
                    'rank': rank,
                    'rating': rating,
                })
            except Exception as e:
                print(f"Error parsing rating history element: {e}")
                continue            
        return ratings_history            
        
    
    def save_to_csv(self, data, filename):
        if not data:
            print("No data to save.")
            return

        keys = data[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
        print(f"Saved data to {filename}")

if __name__ == "__main__":
    # if sets:
    #     # Example: Scrape the first set (usually Longsword)
    #     first_set = sets[0]
    #     print(f"\nScraping {first_set['name']}...")
    #     rankings = scraper.get_rankings(first_set['id'])
        
    # if rankings:
    #     scraper.save_to_csv(rankings, f"rankings_{first_set['id']}.csv")
 
    scraper = HEMARatingsScraper()
    fighter_ids = []
    with open(CURRENT_RANKINGS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rankings = list(reader)
        for fighter in rankings:
            fighter_ids.append(fighter['id'])
    
    aggregate_tournament_histories = []
    aggregate_ratings_history = []
    for fighter_id in fighter_ids:
        # aggregate_tournament_histories.extend(scraper.get_match_history(fighter_id))
        aggregate_ratings_history.extend(scraper.get_ratings_history(fighter_id))
    # scraper.save_to_csv(aggregate_tournament_histories, "../data/raw/tournament_histories.csv")
    scraper.save_to_csv(aggregate_ratings_history, RATINGS_HISTORY_FILE)
    # sets = scraper.get_rating_sets()
    
    # scraper.save_to_csv(aggregate_tournament_histories, MATCH_HISTORY_FILE) 
    # Example: Fetch details for top 3 fighters
    # for fighter in rankings[:3]:
    #     details = scraper.get_match_history(fighter['id'])
    #     print(f"Details for {fighter['name']}: {details}")
