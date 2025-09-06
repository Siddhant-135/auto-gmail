import os
import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time
import json
from urllib.parse import unquote
# Import 'packaging' to compare library versions
from packaging.version import parse as parse_version

# Gmail API imports
import base64
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Selenium imports for dynamic web scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# --- CONFIGURATION ---
load_dotenv()
# Make sure you have a .env file with your GEMINI_API_KEY
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
else:
    print("Error: GEMINI_API_KEY not found. Please create a .env file and add it.")
    exit()


# If modifying these scopes, delete the file token.json.
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.compose']
# The user-provided URL for NLP professors in Europe
CSRANKINGS_URL = "https://csrankings.org/#/index?nlp&europe"

# --- GMAIL API FUNCTIONS ---
def get_gmail_service():
    """Shows basic usage of the Gmail API."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("Error: credentials.json not found. Please download it from the Google Cloud Console.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def create_draft(service, user_id, message_body):
    """Create and insert a draft email."""
    try:
        message = {'message': message_body}
        draft = service.users().drafts().create(userId=user_id, body=message).execute()
        print(f'Draft id: {draft["id"]}\nDraft message: {draft["message"]}')
        return draft
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def create_message(sender, to, subject, message_text):
  """Create a message for an email."""
  message = MIMEText(message_text, 'plain', 'utf-8')
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject
  raw_message = base64.urlsafe_b64encode(message.as_bytes())
  return {'raw': raw_message.decode()}


# --- WEB SCRAPING & RESEARCH FUNCTIONS ---
def scrape_csrankings(url, num_professors=20):
    """Scrapes the top professors from csrankings.org for a given area using Selenium."""
    print("Scraping CSRankings for top professors...")
    
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        location_params = {
            "latitude": 47.3769, "longitude": 8.5417, "accuracy": 100
        }
        driver.execute_cdp_cmd("Emulation.setGeolocationOverride", location_params)
        
        driver.get(url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#ranking > tbody > tr"))
        )
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        professors = []
        
        university_rows = soup.select("#ranking > tbody > tr")

        for uni_row in university_rows:
            university_div = uni_row.select_one('td > div[id*="-faculty"]')
            if not university_div:
                continue

            university_name_encoded = university_div['id'].replace('-faculty', '')
            university_name = unquote(university_name_encoded)

            professor_rows = university_div.select("table > tbody > tr")
            
            for prof_row in professor_rows:
                name_cell = prof_row.select_one('td:nth-of-type(2) > small > a')
                
                if name_cell:
                    name = name_cell.get_text(strip=True)
                    professors.append({'name': name, 'university': university_name})
                    
                    if len(professors) >= num_professors:
                        break
            
            if len(professors) >= num_professors:
                break
                            
    except TimeoutException:
        print("Error: Timed out waiting for the professor table to load on CSRankings.")
        return []
    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return []
    finally:
        driver.quit()

    print(f"Found {len(professors)} professors.")
    return professors

def research_professor(professor_name, university):
    """
    Uses a multi-step, iterative Gemini API process to research a professor and find their email.
    """
    print(f"\n--- Researching Professor: {professor_name} ({university}) using Gemini ---")
    
    try:
        # --- STEP 1: Get a detailed research summary for the professor ---
        print("Step 1: Getting detailed research summary...")
        research_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        
        summary_prompt = f"""
        You are an expert academic research assistant with access to Google Search.
        Use your search capabilities to find the official homepage of Professor {professor_name} from {university}.
        Read through their research interests, biography, and publications.
        Provide a detailed, extended summary (3-4 paragraphs) of their primary research focus, key themes, and recent work.
        """
        summary_response = research_model.generate_content(summary_prompt)
        detailed_summary = summary_response.text
        print("Successfully generated detailed summary.")

        # --- STEP 2: Cross-link summary with student's profile to find relevant themes and papers ---
        print("Step 2: Finding most relevant research theme and paper...")
        
        student_profile_summary = """
        I am a Computer Science sophomore at IIT Delhi with a strong foundation in mathematics and a keen interest in the core challenges of Natural Language Processing. 
        My current focus is on the creation and improvement of Large Language Models, including their architecture, efficiency, and the mitigation of inherent biases—an interest sparked by my research analyzing LLMs in academic peer review. 
        I am also deeply interested in the complexities of speech processing, particularly audio-to-text conversion. 
        My motivation stems from a desire to build more accessible technologies, whether through enabling multilingual NLP for vernacular languages, developing more capable and nuanced AI systems, or exploring how these systems intersect with other scientific fields.
        """

        paper_schema = {
            "type": "OBJECT",
            "properties": {
                "most_relevant_theme": {"type": "STRING"},
                "representative_paper_title": {"type": "STRING"},
                "reasoning": {"type": "STRING"}
            },
            "required": ["most_relevant_theme", "representative_paper_title", "reasoning"]
        }

        crosslink_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config={"response_mime_type": "application/json", "response_schema": paper_schema}
        )

        crosslink_prompt = f"""
        You are a research advisor matching a student to a professor's work.
        
        Professor's Research Summary:
        ---
        {detailed_summary}
        ---

        Student's Profile:
        ---
        {student_profile_summary}
        ---

        Based on the synergy between the professor's summary and the student's profile, identify the SINGLE most relevant **research theme**.
        Then, use your search capabilities to find a **representative paper title** for that theme by Professor {professor_name}. The paper does not have to be from the last year.
        Provide a one-sentence reasoning for why this theme is the best fit.
        Return the result in the specified JSON format. If no specific paper title can be found, return "N/A" for the title but still provide the theme.
        """
        
        paper_response = crosslink_model.generate_content(crosslink_prompt)
        paper_data = json.loads(paper_response.text)
        
        most_relevant_theme = paper_data.get('most_relevant_theme', 'General NLP Research')
        representative_paper = paper_data.get('representative_paper_title', 'N/A')
        
        print(f"Most relevant theme found: {most_relevant_theme}")
        print(f"Representative paper found: {representative_paper}")

        # --- STEP 3: Find the professor's email address ---
        print("Step 3: Finding professor's email address...")
        email_prompt = f"""
        Use Google Search to find the professional/academic email address for Professor {professor_name} at {university}. 
        Return ONLY the email address as a plain string. For example: "example.prof@university.edu". If you cannot find it, return "N/A".
        """
        email_response = research_model.generate_content(email_prompt)
        found_email = email_response.text.strip()
        print(f"Email found: {found_email}")

        # --- STEP 4: Consolidate information ---
        final_summary = f"Key Research Theme: {most_relevant_theme}\n\n{detailed_summary}"

        return {
            'name': professor_name,
            'university': university,
            'summary': final_summary,
            'papers': [representative_paper] if representative_paper != "N/A" else [],
            'email': found_email if "N/A" not in found_email else None
        }

    except Exception as e:
        print(f"An error occurred during the multi-step research process: {e}")
        return None

# --- GEMINI & PROMPT FUNCTIONS ---
def load_persona():
    """Loads the persona file."""
    try:
        with open('persona.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print("ERROR: persona.md not found. Please create it based on the template.")
        return None

def generate_email(persona, professor_info):
    """Generates an email using the Gemini API."""
    print("Generating email draft with Gemini...")

    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                  generation_config=generation_config)

    prompt = f"""
    You are an AI assistant helping a student, Siddhant Agrawal, draft emails for research internships.
    Your task is to write a personalized email to a professor. You must follow the style, tone, and narrative of the provided exemplar.

    Here is all the information about Siddhant (his persona, resume, and an exemplar email he wrote):
    --- PERSONA START ---
    {persona}
    --- PERSONA END ---

    Here is the information about the professor he wants to email:
    --- PROFESSOR INFO START ---
    Name: {professor_info['name']}
    University: {professor_info['university']}
    Research Summary: {professor_info['summary']} 
    Representative Paper Title: {', '.join(professor_info['papers'])}
    --- PROFESSOR INFO END ---

    Follow these rules STRICTLY:
    1. Adopt Siddhant's tone from the exemplar: formal, confident, and story-driven.
    2. **Crucial Tone Guidance:** Maintain a humble yet capable tone. The student is expressing strong, informed interest and familiarity, not claiming to be an expert peer. Avoid overhyping his skills; frame them as a solid foundation for learning and contributing under the professor's guidance.
    3. Use the exemplar "Cotterell Email" as a template for structure and flow.
    4. Your main hook should be the **Key Research Theme** mentioned at the start of the professor's summary. 
    5. If a representative paper is provided, cite it as a specific example of that theme. If no paper is provided, do not invent one; instead, discuss the research theme more broadly.
    6. Create a custom paragraph that logically connects one of Siddhant's specific projects/skills to that theme/paper.
    7. Begin the output with "Subject: [Your Subject Line]" and then the full email text. Do not include any other preamble.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    persona = load_persona()
    if not persona:
        exit()

    professors = scrape_csrankings(CSRANKINGS_URL, num_professors=2)
    if not professors:
        print("Could not retrieve professor list. Exiting.")
        exit()

    gmail_service = get_gmail_service()
    if not gmail_service:
        print("Could not connect to Gmail. Drafts will not be created.")

    approved_emails = []

    for prof in professors:
        prof_info = research_professor(prof['name'], prof['university'])
        if not prof_info:
            print(f"Could not research {prof['name']}. Skipping.")
            continue
        
        generated_email_text = generate_email(persona, prof_info)
        if not generated_email_text:
            print("Failed to generate email. Skipping professor.")
            continue

        print("\n" + "="*50)
        print("GENERATED DRAFT")
        print("="*50)
        print(f"For: {prof_info['name']} ({prof_info['university']})")
        print(f"Research Info Found: {prof_info['summary'][:200]}...")
        print(f"Papers Found: {prof_info['papers']}")
        print(f"Email Found: {prof_info.get('email')}")
        print("-"*50)
        print(generated_email_text)
        print("="*50)

        while True:
            action = input("Approve this draft? (y/n/retry): ").lower()
            if action in ['y', 'n', 'retry']:
                break
            print("Invalid input.")

        if action == 'y':
            prof_email = prof_info.get('email')
            # Fallback to manual input if email wasn't found automatically
            if not prof_email:
                prof_email = input("Could not find email automatically. Please enter it now: ")

            try:
                subject = generated_email_text.split('\n')[0].replace("Subject: ", "")
                body = "\n".join(generated_email_text.split('\n')[1:]).strip()
                
                if gmail_service:
                    message = create_message('me', prof_email, subject, body)
                    create_draft(gmail_service, 'me', message)
                    print(f"Draft for {prof_info['name']} created in your Gmail account.")
                else:
                    approved_emails.append({'to': prof_email, 'subject': subject, 'body': body})
                    print("Gmail service not available. Saved for later.")
            except IndexError:
                print("Error: Generated email was not in the expected format. Could not parse subject/body.")


        elif action == 'retry':
            # In a more advanced version, you could modify the prompt here
            print("Retrying... (functionality not implemented, skipping for now)")
            continue
        else:
            print(f"Skipping draft for {prof_info['name']}.")

    print("\nProcess finished.")
    if approved_emails and not gmail_service:
        print("Here are the approved emails that were not drafted:")
        for email in approved_emails:
            print("\n" + "-"*20)
            print(f"To: {email['to']}")
            print(f"Subject: {email['subject']}")
            print(email['body'])

