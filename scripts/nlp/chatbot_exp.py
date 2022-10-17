from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

import logging 
logger = logging.getLogger() 
logger.setLevel(logging.CRITICAL)
# Create a new chat bot named Charlie
bot1 = ChatBot('Charlie')

trainer = ChatterBotCorpusTrainer(bot1)

trainer.train(
    #"chatterbot.corpus.english.conversations",
    "chatterbot.corpus.english",
    #"chatterbot.corpus.english.greetings",
)

bot2 = bot1

class textclass:
    def __init__(self,text):
        self.text=text

response2 = textclass('what is basketball')
# Get a response to the input text 'I would like to book a flight.'

for e in range(5):
    response1 = bot1.get_response(response2.text)
    print('1:',response1)
    response2 = bot2.get_response(response1.text)
    print('2:',response2)
    



