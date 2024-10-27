from utils.assistant import UniversityAssistant
from dotenv import load_dotenv


load_dotenv()
assistant = UniversityAssistant(index_name="ustc-rag-2048")

while True:
    quesetion = input('\nYou: ')
    response = assistant.get_response(quesetion)
    print('\nBot: ', response)
    