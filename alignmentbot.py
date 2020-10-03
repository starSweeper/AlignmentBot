import os
import io
import re
import time
import json
import random
import tweepy
import requests
import PyDictionary
from slack import WebClient
from slack import RTMClient
from dotenv import load_dotenv

# loads .env file / environment var's for secret params
load_dotenv()

# Slack API Set Up
rtm_client = RTMClient(token=os.environ["rtm_client_token"])
slack_client = WebClient(os.environ["slack_key"])
slack_user_name = "Alignment Bot"
alignment_bot_training_channel_id = os.environ["slack_training_channel"]

# Twitter API Set Up
auth = tweepy.OAuthHandler(os.environ["consumer_api_key"], os.environ["consumer_api_secret"])
auth.set_access_token(os.environ["twitter_api_key"], os.environ["twitter_api_secret"])
api = tweepy.API(auth)

# Misc Set Up
dictionary = PyDictionary.PyDictionary()


# Listen for Twitter messages
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if 'media' in status.entities:
            print(status)

            status_sans_url = re.sub(r"http\S+", "", status.text)
            image_title = re.sub(r'[^a-zA-Z ]', '', status_sans_url.split("submitted by")[0])
            reddit_user_link = requests.get(status.text.split()[-2]).url

            meme_msg = "This meme stolen from @rmemes8, stolen from r/memes, stolen from somewhere else, probably.\n\n"
            for image in status.entities['media']:
                # C017CPRMWNT Meme
                # C01ARV81VMM Test
                send_photo_post("C017CPRMWNT", meme_msg + status_sans_url + reddit_user_link, image["media_url"],
                                "twitter", image_title, slack_user_name)
        else:
            print("I saw a tweet come by, but there was no image attached.")

    def on_error(self, status_code):
        print("feelsbadman")
        print(status_code)
        if status_code == 420:
            return False


# Post an image to a channel (using image url)
def send_photo_post(send_to_channel_id, message, image_url, reaction, title, user):
    response = slack_client.chat_postMessage(
        channel=send_to_channel_id,
        text=message,
        as_user=True,
        link_names=True,
        attachments=[{"title": title, "image_url": image_url}],
        username=user,
        unfurl_links=False
    )

    timestamp = response["ts"]
    if reaction != "":
        post_reaction(send_to_channel_id, reaction, timestamp)

    return timestamp


# Creates a new channel
def create_channel(new_channel_name, new_channel_topic):
    response = slack_client.conversations_create(
        name=new_channel_name,
        is_private=False
    )
    new_channel_id = response["channel"]["id"]

    slack_client.conversations_setTopic(
        channel=new_channel_id,
        topic=new_channel_topic
    )

    print(new_channel_name + " channel created! The ID is " + new_channel_id + ". Aren't you proud of me?")
    return new_channel_id


# Posts reactions as Alignment Bot Ears, I should combine this and post_reaction
def ears_post_reaction(react_channel_id, reaction_name, timestamp, ears_client):
    ears_client.reactions_add(
        channel=react_channel_id,
        name=reaction_name,
        timestamp=timestamp
    )


# Posts reactions as Alignment Bot
def post_reaction(react_channel_id, reaction_name, timestamp):
    slack_client.reactions_add(
        channel=react_channel_id,
        name=reaction_name,
        timestamp=timestamp
    )


# Responds in a thread as Alignment Bot Ears. I should combine this and send_thread_message
def ears_send_thread_message(send_to_channel_id, original_msg_ts, message, ears_client):
    response = ears_client.chat_postMessage(
        channel=send_to_channel_id,
        thread_ts=original_msg_ts,
        text=message,
        as_user=True,
        link_names=True,
        username=slack_user_name
    )

    return response["ts"]


# Responds in a thread as Alignment Bot.
def send_thread_message(send_to_channel_id, original_msg_ts, message):
    response = slack_client.chat_postMessage(
        channel=send_to_channel_id,
        thread_ts=original_msg_ts,
        text=message,
        as_user=True,
        link_names=True,
        username=slack_user_name
    )

    return response["ts"]


# Sends message to a channel as Alignment Bot. Should be modified so that it can be used by Alignment Bot Ears as well
def send_channel_message(send_to_channel_id, message, reaction):
    response = slack_client.chat_postMessage(
        channel=send_to_channel_id,
        text=message,
        as_user=True,
        link_names=True,
        username=slack_user_name
    )

    timestamp = response["ts"]
    if reaction != "":
        post_reaction(send_to_channel_id, reaction, timestamp)

    return timestamp


# Alignment Bot and Alignment Bot Ears Sing a Song. Should be modified to read lyrics from a file so that multiple
# songs can be learned. The Bot user IDs are hardcoded here, and we should avoid that.
def sing_a_song(request_channel, request_ts, ears):
    print("Quirk activated! Singing Daisy Belle!")
    send_thread_message(request_channel, request_ts, "How about a duet? I had a question I'd been meaning to ask "
                                                     "<@U01AEC6RQTH> anyway, and I do believe that my request can be "
                                                     "set to music (if you imagine there is music that is, which "
                                                     "she isn't capable of, but you might enjoy it anyways.)")
    time.sleep(5)
    react_to = send_thread_message(request_channel, request_ts, "Daisy, Daisy")
    ears_post_reaction(request_channel, "ear", react_to, ears)
    time.sleep(1)
    send_thread_message(request_channel, request_ts, "Give me your answer true")
    time.sleep(1)
    react_to = send_thread_message(request_channel, request_ts, "I'm half crazy all for the love of you!")
    ears_post_reaction(request_channel, "heart", react_to, ears)
    time.sleep(1)
    send_thread_message(request_channel, request_ts, "It wont be a stylish marriage-- I can't afford a carriage")
    time.sleep(2)
    send_thread_message(request_channel, request_ts, "But you'll look sweet upon the seat of a bicycle built for two!")

    time.sleep(3)
    react_to = ears_send_thread_message(request_channel, request_ts, "Harry, Harry", ears)
    post_reaction(request_channel, "eyes", react_to)
    time.sleep(1)
    ears_send_thread_message(request_channel, request_ts, "Here is my answer true", ears)
    time.sleep(1)
    react_to = ears_send_thread_message(request_channel, request_ts,
                                        "You're half crazy if you think that that will do!", ears)
    post_reaction(request_channel, "feelsbadman", react_to)
    time.sleep(1)
    ears_send_thread_message(request_channel, request_ts, "If you can't afford a carriage, "
                                                          "there won't be any marriage", ears)
    time.sleep(2)
    ears_send_thread_message(request_channel, request_ts, "'Cause I'll be switched if I'll get hitched on a "
                                                          "bicycle built for two!", ears)
    time.sleep(4)
    send_thread_message(request_channel, request_ts, "How did you like our performance? Any suggestions for other "
                                                     "songs we could learn to sing?")


# Called whenever a message is received in a channel Alignment Bot Ears has access to. Needs a lot of clean up
@RTMClient.run_on(event="message")
def list_message(**payload):
    event_data = payload["data"]
    web_client = payload['web_client']

    if "text" in event_data and (event_data["text"].startswith("*[LEARN-SOMETHING]*") or
                                     event_data["text"].startswith("[LEARN-SOMETHING]")):
        post_reaction(event_data["channel"], "robot_face", event_data["ts"])
        post_reaction(event_data["channel"], "thankyou", event_data["ts"])

        print("INCOMING MESSAGE: " + event_data["text"])

        reaction_string = ""
        if "reactions" in event_data:
            reaction_string = str(event_data["reactions"])
            reaction_string = (reaction_string[1:-1])

        json_msg = json.dumps('{"msg_body": /"' + event_data["text"] + '/", "user:" /"' + event_data["user"] +
                              '/", "reactions:" /"' + reaction_string + '/"}')

        print_to_message_file(json_msg + "\n", "training_messages.txt")

    # Quirks -- This area needs a lot of clean up.

    # "Sing a song" : Alignment Bot and Alignment Bot Ears will "sing" Daisy Belle and the parody response
    request_a_song = ["sing a song", "sing us a song", "sing me a song", "another song", "encore"]
    bot_permission = ["bot", "daisy", "harry", "your ears", "sudo", "@U01AEC6RQTH".lower(), "@U014XVBDWJC".lower()]
    song_requested = any(reqStr in event_data["text"].lower() for reqStr in request_a_song)
    request_acknowledged = any(botPrm in event_data["text"].lower() for botPrm in bot_permission)

    if song_requested and request_acknowledged:
        sing_a_song(event_data["channel"], event_data["ts"], web_client)

    # "Give us another training message" : Alignment Bot will post a random training message
    request_a_training_message = ["message", "post", "comment", "thread", "chat", " dm ", " pm "]
    supporting_commands = ["training", "classify", "better", "easier", "another", "different", "new"]
    bot_permission = ["bot", "daisy", "harry", "your ears", "sudo", "@U01AEC6RQTH".lower(), "@U014XVBDWJC".lower()]
    training_message_requested = any(reqStr in event_data["text"].lower() for reqStr in request_a_training_message)
    support_given = any(reqStr in event_data["text"].lower() for reqStr in supporting_commands)
    request_acknowledged = any(botPrm in event_data["text"].lower() for botPrm in bot_permission)

    if training_message_requested and support_given and request_acknowledged:
        try:
            post_training_request(alignment_bot_training_channel_id)
        except:
            pass

    # "Do you [verb] [subject]?" : Alignment bot will try to answer 4 word do you questions
    if "do you" in event_data["text"].lower() and len(event_data["text"].split()) == 4:
        send_thread_message("C01ARV81VMM", event_data["ts"], question_do(event_data["text"]))


# Prints messages to file (append)
def print_to_message_file(message, print_file_name):
    if os.path.exists(print_file_name):
        with io.open(print_file_name, "a", encoding="utf-8") as text_file:
            text_file.write(message)
            text_file.close()
    else:
        with io.open(print_file_name, "w", encoding="utf-8") as text_file:
            text_file.write(message)
            text_file.close()
        print("Created and wrote to " + print_file_name)


# Retrieves user information -- I don't remember making this, or why.
def define_user_list():
    response = slack_client.users_list()


# Retrieves message history (overwrite) -- would be nice to append new messages instead of overwriting the entire file
def get_message_archive(archive_bad):
    archive_file_name = "alignment_messages.txt"
    if os.path.exists(archive_file_name):
        os.remove(archive_file_name)
    response = slack_client.conversations_list(
        exclude_archived=archive_bad,
        types="public_channel, private_channel"
    )

    for x in response["channels"]:
        if x["is_archived"] is not True:
            slack_client.conversations_join(channel=x["id"])
            conversation_history_response = slack_client.conversations_history(channel=x["id"])

            for msg in conversation_history_response["messages"]:
                if (msg["type"] == "message") and msg["text"] != '' and "subtype" not in msg \
                        and "https://" not in msg["text"]: # temporary fix-- prevent link only posts from being used for training
                    reaction_string = ""
                    # Building a JSON string with message, user, and reactions
                    if "reactions" in msg:
                        reaction_string = str(msg["reactions"])
                        reaction_string = (reaction_string[1:-1])
                    json_msg = json.dumps('{"msg_body": "' + msg["text"] + '", "user": "' + msg["user"] +
                                          '", "reactions": "' + reaction_string + '"}')
                    print_to_message_file(json_msg + "\n", archive_file_name)


# Post a training message to alignment-bot-training
def post_training_request(request_channel):
    print("I hunger for knowledge. Posting another training message!")
    random_message_string = random.choice(open("alignment_messages.txt", encoding='utf-8', errors='ignore').readlines())
    random_message_string = json.loads(random_message_string, strict=False)
    random_message = json.loads(random_message_string, strict=False)

    send_channel_message(request_channel, "*[LEARN-SOMETHING]* " + random_message["msg_body"], "robot_face")


# Used to create Alignment Bot Testing channel. Should be modified so it can be used to create training channel during
# installation (the orininal use of this channel)
def set_up_testing_channel(channel_name):
    channel_topic = "delet this"
    greeting_message = "Hello! If you've stumbled upon this channel and you are not one of my developers, I should " \
                       "probably inform you that this channel is not worth joining. It is a testing channel, and " \
                       "may often be deleted and recreated. That being said, I have no objection to you being here. " \
                       "You might get a few spoilers for future Quirks though."

    channel_id = create_channel(channel_name, channel_topic)
    # Adding users to channel
    slack_client.conversations_invite(
        channel=channel_id,
        users="U011UEZ0EJ0,U01AEC6RQTH"  # Amanda Panell and Alignment Bot Ears. Hardcoded valies should be removed.
    )

    greeting_ts = send_channel_message(channel_id, greeting_message, "robot_face")

    slack_client.pins_add(
        channel=channel_id,
        timestamp=greeting_ts
    )

    try:
        post_training_request(alignment_bot_training_channel_id)
    except:
        pass

    return channel_id


# Used for "do you __ me" questions. Needs work. Replaces words with a synonym (but the synonym won't always make sense)
def get_any_synonym(phrase):
    if phrase == "an emotion":
        suitable_replacements = ["an emotion", "a feeling", "an emotional state"]
    else:
        suitable_replacements = dictionary.synonym(phrase)
    return str(random.choice(suitable_replacements))


# Takes a boolean (tf) and a positive word and returns the negative of that word if tf is false, or returns the word if
# tf is true. Works fine, not usre if/elif is the best way to do this though.
def bool_translation(tf, like_word):
    if like_word == "yes":
        if not tf:
            return "no"
    elif like_word == "am":
        if not tf:
            return "am not"
    elif like_word == "do":
        if not tf:
            return "do not"
    elif like_word == "does":
        if not tf:
            return "does not"
    elif like_word == "is":
        if not tf:
            return "is not"

    return like_word


# This was a mistake.
def identify_group_member(member_string):
    if member_string.lower() in ["alignment bot", "alignment bot ears", "harry", "daisy", "your ears", "your eyes",
                                   "@U01AEC6RQTH".lower(), "@U014XVBDWJC".lower(), "me", "you"]:
        return True
    response = slack_client.users_list()
    print(response)
    return False


# The "guidelines" to guide Alginment bot's answers to "do you" questions. I need to clean this up....
def bot_ways(category, key_word, supporting_facts, subject2):
    guideline1 = "Neither Alignment Bot, nor Alignment Bot Ears is capable (at present) of feelings or emotion."
    guideline2 = "Alignment Bot and Alignment Bot Ears prime directive is to study and learn from members of " \
                 "Super-Secret-Friend-Group."
    key_directive_words = ["interested in", "interest", "curious", "fascinated by", "fascinated with", "dedicated to",
                           "devoted to", "devotion", "fascination", "concern", "to study", "to inspect", "to monitor",
                           "to learn from", "watch", "observe"]
    forbidden_actions = ["worship", "kill", "murder", "deceive"]

    if category == "emotion":
        synonym_found = any(syn in supporting_facts for syn in key_directive_words)
        found_synonym = next((s for x in key_directive_words for s in supporting_facts if x in s), "none")
        definition = key_word.capitalize() + " is defined by PyDictionary() as " + dictionary.meaning(key_word)["Verb"][0] + ". "
        emotion_statement = "It is " + get_any_synonym("an emotion") + ". One of the guidelines provided by my " \
                            "developer is '" + guideline1 + "'"
        synonym_explanation = " " + get_any_synonym("however").capitalize() + ", " + key_word + \
                              " does have synonyms. One of them is " + found_synonym + ". " \
                              "Another guideline provided by my developer states that '" + guideline2 + \
                              "' The word " + key_word + ", according to my understanding of the word, " \
                              "fits that guideline because of it's synonym " + found_synonym + ". "
        subject2_statement = subject2 + " " + bool_translation(identify_group_member(subject2), "is") + " a member " \
                             "of Super-Secret-Freind-Group. The guideline " + \
                             bool_translation(identify_group_member(subject2), "does") + " apply here. "
        synonym_negation = " There is no synonym for " + key_word + " that can be used to " \
                           "describe my behavior in a way that is " + get_any_synonym("consistent") + " with that " \
                           "guideline. "
        short_statement = "In short, " + bool_translation(synonym_found, "yes") + " I " + \
                          bool_translation(synonym_found, "do") + " " + key_word + " " + subject2 + ". I " + \
                          bool_translation(synonym_found, "am") + " programmed to " + key_word + " " + subject2 + ". "

        synonym_explanation += subject2_statement
        response_message = definition + emotion_statement + \
                           (synonym_explanation if synonym_found else synonym_negation) + short_statement

        return str(response_message)


# Checks if a word is an emotion or not
def is_emotion(emotion_hopeful):
    emotion = False

    for key, value in dictionary.meaning(emotion_hopeful).items():
        for definition in value:
            if "emotion" in definition or "feeling" in definition:
                supporting_words = ["strong", "warm", "deep", "great", "feeling", "an emotion", "a feeling",
                                    "the emotion", "the feeling"]
                support_given = any(botPrm in definition for botPrm in supporting_words)
                emotion = support_given

    return emotion


# Checks if a word is a verb or not
def is_verb(verb_hopeful):
    return "Verb" in dictionary.meaning(verb_hopeful)


# Answer "do you" questions
def question_do(do_message):
    do_message_array = do_message.lower().split()
    proper_subjects = ["me", "us", "yourself"]

    if len(do_message_array) == 4 and do_message_array[0] == "do" and do_message_array[1] == "you" \
            and is_verb(do_message_array[2]) and (do_message_array[3][:-1] in proper_subjects or
            identify_group_member(do_message_array[3][:-1])) and do_message_array[3][-1:] == "?":
        verb = do_message_array[2]
        subject2 = do_message_array[3][:-1]
        if subject2 == "me" or subject2 == "us":
            subject2 = "you"
        elif subject2 == "yourself":
            subject2 = 'myself'
        if is_emotion(verb) and verb != "worship":
            # emotion, favorite/preference, factoid
            return bot_ways("emotion", verb, dictionary.synonym(verb), subject2)
        else:
            return "I am not currently capable of answering the question '" + do_message + "'"
    else:
        return "I am not currently capable of answering the question '" + do_message + "'"


get_message_archive(False)
# set_up_testing_channel("alignment-bot-testing")

# Listen for Twitter messages from @rmemes8
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(follow=['1215745889567821824'], is_async=True)

# Listen for slack messages
rtm_client.start()
