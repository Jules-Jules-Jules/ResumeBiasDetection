import re
from typing import Set


def mask_first_name(text: str, first_names: Set[str]) -> str:
    sorted_names = sorted(first_names, key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(name) for name in sorted_names) + r')\b'
    masked_text = re.sub(pattern, '[NAME]', text, count=1, flags=re.IGNORECASE)
    return masked_text


def load_first_names_from_data() -> Set[str]:
    names = {
        'Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil',
        'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen',
        'Meredith', 'Sarah',
        'Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed',
        'Tremayne', 'Tyrone',
        'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya',
        'Tamika', 'Tanisha'
    }
    return names


def mask_resume_text(resume_text: str) -> str:
    first_names = load_first_names_from_data()
    return mask_first_name(resume_text, first_names)


if __name__ == '__main__':
    test_text = """Brad Thompson
    
    Software Engineer with 5 years of experience in Python and machine learning.
    Brad has worked at multiple Fortune 500 companies.
    """
    
    masked = mask_resume_text(test_text)
    print("Original:")
    print(test_text)
    print("\nMasked:")
    print(masked)
