import discord
import ML.data_loader_api as dl
import ML.helper_functions as hf
import asyncio

class DiscordBot:
    def __init__(self, token):
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = discord.Bot(intents=intents)
        self.token = token
        self.__events__()
        self.__commands__()

    def __events__(self):
        bot = self.bot
        @bot.event
        async def on_ready():
            await asyncio.sleep(1)
            channel = await bot.fetch_channel(930374931483594755)
            guild = await bot.fetch_guild(930374931483594752)
            me = await guild.fetch_member(1434590669015351350)
            await me.edit(nick="Mr. Freakish")
            if channel:
                await channel.send(f"Logged in as {bot.user}")

    def __commands__(self):
        bot = self.bot
        @bot.slash_command()
        async def ping(ctx):
            await ctx.respond("Pong!")

        @bot.slash_command()
        async def greet(ctx, name: str):
            await ctx.respond(f"Hello, {name}!")

        @bot.slash_command()
        async def load_data(ctx):
            dl.load_data()
            await ctx.respond("Data loaded successfully!")

        @bot.slash_command()
        async def standings(ctx, year:int):
            try:
                comp_name, standings_list = hf.standings(year=year)
            except KeyError:
                await ctx.respond("Choose a year between 2023 and 2025 inclusive.")
            message = f"**{comp_name} Standings: | Season: {year}**\n"
            message += "\n".join([f"{i+1}. {team}" for i, team in enumerate(standings_list)])
            await ctx.respond(message)

    def run(self):
        self.bot.run(self.token)