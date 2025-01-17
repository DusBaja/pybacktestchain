import random
from datetime import datetime
# List of animals, professions, and colors
animals = [
    "Eagle", "Tiger", "Lion", "Wolf", "Bear", "Falcon", "Shark", "Panther", "Leopard", "Cheetah", 
    "Hawk", "Fox", "Owl", "Cobra", "Jaguar", "Horse", "Elephant", "Dolphin", "Gorilla", "Lynx"
]

professions = [
    "Carpenter", "Engineer", "Doctor", "Pilot", "Farmer", "Artist", "Blacksmith", "Chef", "Teacher", "Mechanic",
    "Architect", "Scientist", "Soldier", "Nurse", "Firefighter", "Plumber", "Astronaut", "Tailor", "Photographer", "Lawyer"
]

colors = [
    "Red", "Blue", "Green", "Yellow", "Black", "White", "Purple", "Orange", "Brown", "Grey", 
    "Pink", "Violet", "Crimson", "Turquoise", "Gold", "Silver", "Amber", "Magenta", "Teal", "Indigo"
]

# Function to generate a random name
def generate_random_name():
    animal = random.choice(animals)
    profession = random.choice(professions)
    color = random.choice(colors)
    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{color}{animal}{profession}_{current_date}" #we reduce the chances to have twice the same 
