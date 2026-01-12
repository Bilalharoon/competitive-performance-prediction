from src.scraper import HEMARatingsScraper

web_scraper = HEMARatingsScraper()

# match_history = web_scraper.get_match_history(711)
# print(match_history)

ratings_history = web_scraper.get_ratings_history(711)
print(ratings_history)

