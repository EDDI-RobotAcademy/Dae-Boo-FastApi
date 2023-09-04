import random
from fastapi import APIRouter

request_receiver = APIRouter()

@request_receiver.get("/age-recommend-card")
def request_age_recommend():
    age_card_numbers = []
    # age_card_numbers = [random.randrange(1, 300) for _ in range(5)]
    age_card_numbers.append(1)
    age_card_numbers.append(2)
    return age_card_numbers
