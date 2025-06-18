import re

def get_valid_email():
    pattern = r"^[^@]+@[^@]+\.[^@]+$"
    while True:
        email = input("Enter your email for Entrez (e.g., user@example.com): ").strip()
        if re.match(pattern, email):
            return email
        print("Invalid email format. Check the expected format and please try again.")

def get_valid_tool():
    pattern = r"^[\w\-]+$"
    while True:
        tool = input("Enter your tool name for Entrez: ").strip()
        if tool and re.match(pattern, tool):
            return tool
        print("Invalid tool name. Use only letters, numbers, underscores, or hyphens.")