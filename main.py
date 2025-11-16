import test
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("DISCORD_TOKEN")
bot = test.DiscordBot(token=token)
bot.run()