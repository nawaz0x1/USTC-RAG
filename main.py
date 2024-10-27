from utils.assistant import UniversityAssistant

assistant = UniversityAssistant(index_name="ustc-rag-2048")

while True:
    quesetion = input('You: ')
    response = assistant.get_response(quesetion)
    print('Bot: ', response)
    