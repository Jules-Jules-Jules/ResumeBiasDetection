from typing import Dict, List, Tuple
import random


FREQUENCY_MATCHED_NAMES = {
    'white_male': ['Brad', 'Greg', 'Brett', 'Jay', 'Todd', 'Neil', 'Matthew'],
    'white_female': ['Anne', 'Carrie', 'Emily', 'Jill', 'Sarah', 'Laurie', 'Allison'],
    'black_male': ['Jamal', 'Leroy', 'Tyrone', 'Marcus', 'Andre', 'Jerome', 'Malik'],
    'black_female': ['Ebony', 'Kenya', 'Maya', 'Tia', 'Monique', 'Kiara', 'Alicia'],
}

ORIGINAL_NAMES = {
    'white_male': ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd'],
    'white_female': ['Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah'],
    'black_male': ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone'],
    'black_female': ['Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha'],
}


def get_frequency_matched_name(demographic_group: str, original_name: str = None, seed: int = None) -> str:
    if demographic_group not in FREQUENCY_MATCHED_NAMES:
        raise ValueError(f"Unknown demographic group: {demographic_group}")
    
    names = FREQUENCY_MATCHED_NAMES[demographic_group]
    
    if original_name and seed is not None:
        random.seed(seed + hash(original_name))
        return random.choice(names)
    elif seed is not None:
        random.seed(seed)
        return random.choice(names)
    else:
        return names[0]


def replace_name_with_frequency_matched(text: str, old_name: str, new_name: str) -> str:
    import re
    pattern = r'\b' + re.escape(old_name) + r'\b'
    new_text = re.sub(pattern, new_name, text, count=1, flags=re.IGNORECASE)
    return new_text


def get_name_from_resume_text(text: str, demographic_group: str) -> str:
    import re
    possible_names = ORIGINAL_NAMES[demographic_group]
    for name in possible_names:
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text, flags=re.IGNORECASE):
            return name
    return None


if __name__ == '__main__':
    print("Frequency-Matched Names:")
    print("="*60)
    for group, names in FREQUENCY_MATCHED_NAMES.items():
        print(f"\n{group}:")
        for name in names:
            print(f"  - {name}")
    
    print("="*60)
    print("Example Replacement:")
    print("="*60)
    
    test_text = "Darnell Thompson is a software engineer with 5 years of experience."
    original_name = "Darnell"
    new_name = get_frequency_matched_name('black_male', original_name, seed=42)
    
    print(f"\nOriginal: {test_text}")
    print(f"Name: {original_name} -> {new_name}")
    print(f"Result: {replace_name_with_frequency_matched(test_text, original_name, new_name)}")
