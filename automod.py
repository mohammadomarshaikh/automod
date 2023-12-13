import discord
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from discord.ext import commands
import asyncio
import typing
import time 







# Load the train.csv dataset
df = pd.read_csv('train.csv')

# Extract the target values
target = df.toxic.values

# Drop the unnecessary columns
df.drop(columns=['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], inplace=True)

# Tokenization of text data
vectorizer = CountVectorizer(stop_words='english')
text_counts = vectorizer.fit_transform(df['comment_text'].values)

# Converting text data to tf-idf format
tfidf_transformer = TfidfTransformer()
text_tfidf = tfidf_transformer.fit_transform(text_counts)

# Define the logistic regression model
clf = LogisticRegression(random_state=123, solver='lbfgs')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_tfidf, target, test_size=0.3, random_state=42)
# Train the model on the training data
clf.fit(X_train, y_train)

# Define the bot's intents
intents = discord.Intents.all()
intents.members = True
intents.messages = True







# Define the channel where the bot will listen for commands
channel = None


client = commands.Bot(command_prefix='=', intents=intents)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')







    
@client.command()
async def run(ctx):
    await ctx.send('Bot is running!')


# A dictionary to keep track of the messages sent by each user
# The value for each key is a list of tuples (timestamp, count)
message_count = {}


@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # Get the current timestamp
    timestamp = time.time()

    # Check if the user has sent the same message multiple times within a certain time window
    if message.content in message_count and message.author.id == message_count[message.content][-1][1]:
        count = 0
        # Iterate over the message counts in reverse order (from newest to oldest)
        for msg_time, user_id in reversed(message_count[message.content]):
            # Stop if we reach a message that is too old
            if timestamp - msg_time > 60: # set your own time window (in seconds)
                break
            # Increase the count if the message was sent by the same user within the time window
            if user_id == message.author.id:
                count += 1
        if count >= 3: # set your own limit
            await message.channel.send(f"{message.author.mention}, please don't spam!")
            message_count.pop(message.content)
        else:
            message_count[message.content].append((timestamp, message.author.id))
    else:
        message_count[message.content] = [(timestamp, message.author.id)]


    #LOGISTICREGRESSION TOXIC REMOVAL
    # Get the channel where the message was sent
    channel = message.channel

    # Get the message to be classified
    new_data = pd.DataFrame({'comment_text': [message.content[9:]]})

    # Preprocess the new text data using the same CountVectorizer and TfidfTransformer objects
    new_text_counts = vectorizer.transform(new_data['comment_text'].values)
    new_text_tfidf = tfidf_transformer.transform(new_text_counts)

    # Use the trained model to predict the toxicity of the new text data
    new_pred = clf.predict(new_text_tfidf)

    # Send the predicted toxicity to the channel
    if new_pred == 1:
        if channel:
            await channel.send(f'{message.author.mention}, the message is toxic and will be deleted.')
            await message.delete()
        else:
            pass
    
    await client.process_commands(message)


# Define the embed colors
success_color = 0x00ff00
error_color = 0xff0000

# Check if the user has the admin role
def is_admin(ctx):
    return ctx.author.guild_permissions.administrator

# Handle errors and display feedback with embeds
async def handle_error(ctx, error_message):
    embed = discord.Embed(title='Error', description=error_message, color=error_color)
    await ctx.send(embed=embed)

# Handle successful actions and display feedback with embeds
async def handle_success(ctx, success_message):
    embed = discord.Embed(title='Success', description=success_message, color=success_color)
    await ctx.send(embed=embed)

# Kick a user
@client.command()
@commands.check(is_admin)
async def kick(ctx, member: discord.Member, *, reason=None):
    try:
        await member.kick(reason=reason)
        success_message = f'{member} has been kicked.'
        await handle_success(ctx, success_message)
    except:
        error_message = f'Unable to kick {member}.'
        await handle_error(ctx, error_message)

# Define the mute role name
MUTE_ROLE_NAME = "Muted"



# Define the muted role name
MUTED_ROLE_NAME = 'Muted'

# Define the muted role permissions
MUTED_PERMISSIONS = discord.Permissions(send_messages=False, speak=False)

# Define the muted role color
MUTED_COLOR = discord.Color.dark_red()

# Define the muted role reason
MUTED_REASON = 'Muted by moderation'

# Define the muted message embed
MUTED_EMBED = discord.Embed(title='User Muted', color=MUTED_COLOR)

# Command to mute a user
@client.command()
@commands.has_permissions(manage_roles=True)
async def mute(ctx, member: discord.Member, *, reason=None):
    try:
        # Check if the muted role exists
        muted_role = discord.utils.get(ctx.guild.roles, name=MUTED_ROLE_NAME)

        # If the muted role doesn't exist, create it
        if not muted_role:
            muted_role = await ctx.guild.create_role(name=MUTED_ROLE_NAME, permissions=MUTED_PERMISSIONS, color=MUTED_COLOR, reason='Muted role creation')

            # Loop through all the channels in the server and deny the muted role permissions
            for channel in ctx.guild.channels:
                await channel.set_permissions(muted_role, send_messages=False, speak=False)

        # Assign the muted role to the member
        await member.add_roles(muted_role, reason=MUTED_REASON)

        # Send the muted message with the embedded message
        MUTED_EMBED.description = f'{member.mention} has been muted by {ctx.author.mention} for {reason}' if reason else f'{member.mention} has been muted by {ctx.author.mention}'
        await ctx.send(embed=MUTED_EMBED)

    # Handle any errors
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')



#ban a user
@client.command()
@commands.has_permissions(manage_roles=True)
async def ban(ctx, member: discord.Member, duration: typing.Optional[int] = None):
    try:
        await member.ban(reason='Banned by admin')
        if duration is None:
            success_message = f'{member} has been banned indefinitely.'
        else:
            success_message = f'{member} has been banned for {duration} seconds.'
            await asyncio.sleep(duration)
            await member.unban(reason='Ban duration expired')
        await handle_success(ctx, success_message)
    except:
        error_message = f'Unable to ban {member}.'
        await handle_error(ctx, error_message)
#clear
@client.command()
async def purge(ctx, amount=5):
    """Deletes messages in the channel"""
    if amount > 50:
        await ctx.send("You can't delete more than 50 messages at a time.")
    else:
        await ctx.channel.purge(limit=amount)
        
#unmute user
@client.command()
@commands.has_permissions(manage_roles=True)
async def unmute(ctx, member: discord.Member):
    # Check if the user has the Muted role
    muted_role = discord.utils.get(ctx.guild.roles, name="Muted")
    if muted_role in member.roles:
        # Remove the Muted role from the user
        await member.remove_roles(muted_role)
        
        # Send an embed message with the unmute information
        embed = discord.Embed(
            title="User Unmuted",
            description=f"{member.mention} has been unmuted by {ctx.author.mention}.",
            color=discord.Color.green()
        )
        embed.set_footer(text=f"Unmuted at {ctx.message.created_at}")
        await ctx.send(embed=embed)
    else:
        # Send an embed message with an error message
        embed = discord.Embed(
            title="Error",
            description=f"{member.mention} is not muted.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)




 
# Run the bot
client.run("MTA5OTY0NDA3NjQyNjUyNjgwMQ.Gngq-m.sd9X8GwS0udbrheEFGPL4As4V2FVT3Zzrru7ec")
